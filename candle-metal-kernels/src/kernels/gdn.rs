use crate::utils::EncoderProvider;
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
/// `g`/`beta`: `[b, h]` (`g` already `exp`'d -- the actual decay gate, same
/// convention `sequential_step` itself expects, not `log_g`); `state_in`/
/// `state_out`: `[b, h, hk, hv]`; `out`: `[b, h, hv]`.
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
    q: &Buffer,
    k: &Buffer,
    v: &Buffer,
    g: &Buffer,
    beta: &Buffer,
    state_in: &Buffer,
    state_out: &mut Buffer,
    out: &mut Buffer,
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
