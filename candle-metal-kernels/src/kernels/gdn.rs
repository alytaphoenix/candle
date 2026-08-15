use crate::utils::{BufferOffset, EncoderProvider};
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// Fused gated-DeltaNet decode step -- one Metal dispatch replacing
/// ratatoskr's `sequential_step`'s ~9 serialized candle tensor ops. See
/// `metal_src/gdn.metal`'s own doc comment for the exact math and the
/// functional-state-write correctness argument (`state_out` must be a
/// fresh buffer, `state_in` is read-only and untouched).
///
/// Shapes (all contiguous F32): `q`/`k`: `[b, h, hk]`; `v`: `[b, h, hv]`;
/// `g`/`beta`: `[b, h]` (`g` is **not** `exp`'d -- the kernel takes the raw
/// decay-gate log directly and exponentiates it internally, Phase 4 of the
/// fused-kernel design: this used to be a separate candle-side `.exp()?`
/// dispatch on the caller's side, now folded in; `sequential_step`'s own
/// convention still needs the pre-exp'd value for its own call, since it
/// doesn't take this shortcut); `state_in`/`state_out`: `[b, h, hk, hv]`;
/// `out`: `[b, h, hv]`.
///
/// Every read input takes a `BufferOffset`, not a bare `Buffer` -- a real,
/// found-live bug (ratatoskr's `qwen35_decode_step_matches_hf` differential,
/// 2026-08-15) confirmed that at the real decode call site, `v` is a
/// `narrow()`'d slice of a shared QKV-split buffer with a genuine nonzero
/// byte offset (128 elements in the failing case), silently ignored by an
/// earlier version of the ratatoskr-side wrapper that discarded each
/// tensor's `Layout` and always bound offset 0 -- reading the wrong region
/// of the buffer entirely. `q`/`k` happened to be offset-0 in that same
/// failure (they pass through `l2_normalize`/scaling first, which
/// materializes fresh contiguous tensors), which is exactly why this needs
/// a real offset on *every* read input, not just the one that first
/// exposed it.
///
/// Caller must bind `state_out` and `out` via the write path (this
/// function already does, via `Output::new`) -- binding them read-only
/// would leave the *next* decode step's read of `state_out` without a
/// barrier under this fork's `HazardTrackingModeUntracked` convention, the
/// same bug class as the mm_id-counts incident (see
/// yggdrasil/ratatoskr/DESIGN.md's "Fused Metal kernel for
/// `sequential_step`" section).
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_decode_step_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    h: usize,
    hk: usize,
    hv: usize,
    q: &BufferOffset,
    k: &BufferOffset,
    v: &BufferOffset,
    g: &BufferOffset,
    beta: &BufferOffset,
    state_in: &BufferOffset,
    state_out: &Buffer,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GdnStepArgs {
        hk: u32,
        hv: u32,
        h: u32,
    }

    impl EncoderParam for GdnStepArgs {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_decode_step_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_decode_step b={b} h={h} hk={hk} hv={hv}");

    let args = GdnStepArgs {
        hk: hk as u32,
        hv: hv as u32,
        h: h as u32,
    };
    set_params!(
        encoder,
        (
            q,
            k,
            v,
            g,
            beta,
            state_in,
            Output::new(state_out),
            Output::new(out),
            args
        )
    );

    let grid_dims = MTLSize {
        width: hv,
        height: h,
        depth: b,
    };
    let group_dims = MTLSize {
        width: hv.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnConv1dArgs {
    seq_len: u32,
    hist_len: u32,
    channels: u32,
    kernel_size: u32,
}

impl EncoderParam for GdnConv1dArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Fused causal depthwise conv1d + silu -- the output half. See
/// `metal_src/gdn.metal`'s own doc comment for the exact math (operates
/// conceptually on `history ++ x` without ever materializing the
/// concatenation) and the dispatch-count arithmetic this replaces.
///
/// Shapes (all contiguous F32): `x`: `[b, seq_len, channels]`; `history`:
/// `[b, hist_len, channels]`; `weight`: `[channels, kernel_size]` --
/// **caller must canonicalize to this exact layout** (ratatoskr's own GGUF
/// tensor can arrive as `[channels, kernel]` or `[kernel, channels]`; do the
/// transpose once at load time, not per call, and never pass the
/// non-canonical layout here). `out`: `[b, seq_len, channels]`.
///
/// Every read input takes a `BufferOffset`, not a bare `Buffer` -- same
/// discipline as `call_gdn_decode_step_f32` above, after the real
/// nonzero-offset bug that kernel found live: `history` in particular is
/// exactly the kind of tensor (a restored/cloned cache view, or a
/// `narrow()`'d slice) most likely to carry a real nonzero byte offset in
/// production, not just in a synthetic test.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_causal_conv1d_output_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    hist_len: usize,
    channels: usize,
    kernel_size: usize,
    x: &BufferOffset,
    history: &BufferOffset,
    weight: &BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_causal_conv1d_output_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_causal_conv1d_output b={b} seq_len={seq_len} hist_len={hist_len} channels={channels} kernel_size={kernel_size}");

    let args = GdnConv1dArgs {
        seq_len: seq_len as u32,
        hist_len: hist_len as u32,
        channels: channels as u32,
        kernel_size: kernel_size as u32,
    };
    set_params!(encoder, (x, history, weight, Output::new(out), args));

    let grid_dims = MTLSize {
        width: channels,
        height: seq_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: channels.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

/// Fused causal depthwise conv1d -- the next-conv-state half, companion to
/// `call_gdn_causal_conv1d_output_f32` above (same inputs, no `weight`,
/// different output). `new_state` is written functionally (a fresh buffer,
/// `history` read-only and untouched) -- same correctness discipline as
/// `state_out` in `call_gdn_decode_step_f32`: a prior session-checkpoint
/// clone of `history` must survive this call unchanged, and the caller must
/// bind `new_state` via the write path (this function already does) so the
/// *next* call's read of it gets a barrier under this fork's
/// `HazardTrackingModeUntracked` convention.
///
/// Shapes: `x`: `[b, seq_len, channels]`; `history`: `[b, hist_len,
/// channels]`; `new_state`: `[b, hist_len, channels]`. Correct (a no-op
/// grid) when `hist_len == 0`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_causal_conv1d_state_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    hist_len: usize,
    channels: usize,
    x: &BufferOffset,
    history: &BufferOffset,
    new_state: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_causal_conv1d_state_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_causal_conv1d_state b={b} seq_len={seq_len} hist_len={hist_len} channels={channels}");

    let args = GdnConv1dArgs {
        seq_len: seq_len as u32,
        hist_len: hist_len as u32,
        channels: channels as u32,
        kernel_size: 0,
    };
    set_params!(encoder, (x, history, Output::new(new_state), args));

    let grid_dims = MTLSize {
        width: channels,
        height: hist_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: channels.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnL2NormArgs {
    seq_len: u32,
    heads: u32,
    dim: u32,
    scale: f32,
    eps: f32,
}

impl EncoderParam for GdnL2NormArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Fused L2-normalize + scale -- DeltaNet-preprocessing fusion Kernel B,
/// the q/k half. See `metal_src/gdn.metal`'s own doc comment for the exact
/// math (eps is added to the sum of squares, matching
/// ratatoskr's `SSMWeights::l2_normalize` exactly -- do not substitute a
/// generic RMS-norm kernel, the epsilon placement differs). Called once for
/// q (`scale = 1/sqrt(state_size)`) and once for k (`scale = 1.0`).
///
/// Shapes (contiguous F32): `x`/`out`: `[b, seq_len, heads, dim]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_l2_normalize_scale_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    heads: usize,
    dim: usize,
    scale: f32,
    eps: f32,
    x: &BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_l2_normalize_scale_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_l2_normalize_scale b={b} seq_len={seq_len} heads={heads} dim={dim} scale={scale}");

    let args = GdnL2NormArgs {
        seq_len: seq_len as u32,
        heads: heads as u32,
        dim: dim as u32,
        scale,
        eps,
    };
    set_params!(encoder, (x, Output::new(out), args));

    let grid_dims = MTLSize {
        width: heads,
        height: seq_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: heads.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[repr(C)]
struct GdnDecayBetaArgs {
    seq_len: u32,
    heads: u32,
}

impl EncoderParam for GdnDecayBetaArgs {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// Fused decay-gate softplus + beta sigmoid -- DeltaNet-preprocessing
/// fusion Kernel B, the gating half. See `metal_src/gdn.metal`'s own doc
/// comment for the exact math. `dt_bias`/`ssm_a` are indexed directly by
/// head (`[heads]`), not pre-broadcast -- this eliminates the caller's own
/// `broadcast_as` calls entirely, not just the elementwise math around them.
///
/// Shapes (contiguous F32): `alpha_logits`/`beta_logits`/`g_out`/`beta_out`:
/// `[b, seq_len, heads]`; `dt_bias`/`ssm_a`: `[heads]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_decay_beta_gate_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    b: usize,
    seq_len: usize,
    heads: usize,
    alpha_logits: &BufferOffset,
    dt_bias: &BufferOffset,
    ssm_a: &BufferOffset,
    beta_logits: &BufferOffset,
    g_out: &Buffer,
    beta_out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "kernel_gdn_decay_beta_gate_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_decay_beta_gate b={b} seq_len={seq_len} heads={heads}");

    let args = GdnDecayBetaArgs {
        seq_len: seq_len as u32,
        heads: heads as u32,
    };
    set_params!(
        encoder,
        (alpha_logits, dt_bias, ssm_a, beta_logits, Output::new(g_out), Output::new(beta_out), args)
    );

    let grid_dims = MTLSize {
        width: heads,
        height: seq_len,
        depth: b,
    };
    let group_dims = MTLSize {
        width: heads.min(64),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}
