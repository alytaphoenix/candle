#include <metal_stdlib>
using namespace metal;

// Fused gated-DeltaNet decode step. One dispatch replaces the candle-side
// ratatoskr::sequential_step's ~9 serialized, barrier-separated tensor ops
// (decay, kv_mem read, delta, state write, output read) with a single
// kernel -- see yggdrasil/ratatoskr/DESIGN.md's "Fused Metal kernel for
// `sequential_step`" section for the design and the math this must match.
//
// Per output column j (over the value_head_dim axis), for a fixed
// (batch, head), and summing over i (the state_size axis):
//   g           = exp(g_log)                         [Phase 4: folded in here, not a separate candle dispatch]
//   s_dec[i][j] = g * s_in[i][j]
//   kv_mem[j]   = sum_i s_dec[i][j] * k[i]
//   delta[j]    = (v[j] - kv_mem[j]) * beta
//   s_out[i][j] = s_dec[i][j] + k[i] * delta[j]
//   out[j]      = sum_i s_out[i][j] * q[i]
//
// One thread per (batch, head, j) -- no cross-thread communication, no
// threadgroup memory. state_out is written functionally (a fresh buffer,
// state_in left untouched) -- deliberate, not an oversight: ratatoskr's
// session-checkpoint mechanism relies on DeltaNet state updates being
// immutable rebinds, never in-place mutations (see the DESIGN.md section
// above for the full correctness argument). state_in is read-only; every
// per-thread write to state_out/out is to a disjoint element, so the
// kernel itself has no internal write hazard -- but the caller MUST bind
// state_out and out via the write/Output path (not read-only), or the
// *next* decode step's read of state_out gets no barrier under this
// fork's HazardTrackingModeUntracked convention (see DESIGN.md's own
// account of the mm_id-counts barrier bug -- same class, must not repeat).

struct gdn_step_args {
    uint hk; // state_size
    uint hv; // value_head_dim
    uint h;  // value_head_count (heads)
};

kernel void kernel_gdn_decode_step_f32(
        device const float * q      [[buffer(0)]],  // [b, h, hk]
        device const float * k      [[buffer(1)]],  // [b, h, hk]
        device const float * v      [[buffer(2)]],  // [b, h, hv]
        device const float * g_log  [[buffer(3)]],  // [b, h], NOT exp'ed -- the raw decay-gate log; this kernel exponentiates it internally
        device const float * beta   [[buffer(4)]],  // [b, h]
        device const float * s_in   [[buffer(5)]],  // [b, h, hk, hv]
        device float       * s_out  [[buffer(6)]],  // [b, h, hk, hv], functional -- fresh buffer, s_in untouched
        device float       * out    [[buffer(7)]],  // [b, h, hv]
        constant gdn_step_args & args [[buffer(8)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint j = gid.x;
    if (j >= args.hv) {
        return;
    }
    const uint hk = args.hk;
    const uint hv = args.hv;
    const uint bh = gid.z * args.h + gid.y; // flattened (batch, head) index

    device const float * qh = q + bh * hk;
    device const float * kh = k + bh * hk;
    device const float * si = s_in  + bh * hk * hv;
    device float       * so = s_out + bh * hk * hv;

    const float gv = exp(g_log[bh]); // one exp() per (batch, head), not a separate candle-side dispatch
    const float bv = beta[bh];

    float kv_mem = 0.0f;
    for (uint i = 0; i < hk; ++i) {
        kv_mem += (gv * si[i * hv + j]) * kh[i];
    }
    const float delta = (v[bh * hv + j] - kv_mem) * bv;

    float acc = 0.0f;
    for (uint i = 0; i < hk; ++i) {
        const float s_new = gv * si[i * hv + j] + kh[i] * delta;
        so[i * hv + j] = s_new;
        acc += s_new * qh[i];
    }
    out[bh * hv + j] = acc;
}

// Fused causal depthwise conv1d + silu for gated-DeltaNet's preprocessing
// pipeline -- replaces ratatoskr's `SSMWeights::apply_conv1d`'s Rust-level
// `for t in 0..seq_len { for k in 0..kernel { ... } }` loop (O(seq_len *
// kernel_size) separate candle tensor-op dispatches -- narrow/broadcast_mul/
// add per tap) with one dispatch for the conv+silu output and one small
// dispatch for the next conv state. See yggdrasil/ratatoskr/DESIGN.md's
// native-MTP section, "DeltaNet-preprocessing fusion" subsection, for the
// full design and the dispatch-count arithmetic this must beat.
//
// Conceptually operates on `padded = history ++ x` (concat along time),
// length `hist_len + seq_len`. For output position t (0 <= t < seq_len):
//   out[t][c] = silu( sum_k padded[t+k][c] * weight[c][k] )
// For next-state position s (0 <= s < hist_len) -- the trailing hist_len
// entries of `padded`, i.e. `padded[seq_len + s]`:
//   new_state[s][c] = padded[seq_len + s][c]
// Both are expressed directly against `history`/`x` (never materializing
// `padded` itself) via the same `idx < hist_len ? history[idx] : x[idx -
// hist_len]` branch -- one thread per (channel, output-position, batch), no
// cross-thread communication, no threadgroup memory. `new_state` is written
// functionally (a fresh buffer) -- same correctness discipline as
// `kernel_gdn_decode_step_f32` above: a prior session-checkpoint clone of
// the old conv_state must survive this call completely unchanged.
//
// weight is `[channels, kernel_size]`, already canonicalized to that exact
// layout by the caller (ratatoskr's own `ssm_conv1d` GGUF tensor can arrive
// as either `[channels, kernel]` or `[kernel, channels]` -- the transpose,
// if needed, happens once at model-load time, not per dispatch, and never
// inside this kernel).

struct gdn_conv1d_args {
    uint seq_len;
    uint hist_len;
    uint channels;
    uint kernel_size; // only read by the _output kernel; harmless unused field on the _state kernel
};

kernel void kernel_gdn_causal_conv1d_output_f32(
        device const float * x        [[buffer(0)]],  // [b, seq_len, channels]
        device const float * history  [[buffer(1)]],  // [b, hist_len, channels]
        device const float * weight   [[buffer(2)]],  // [channels, kernel_size]
        device float       * out      [[buffer(3)]],  // [b, seq_len, channels], silu already applied
        constant gdn_conv1d_args & args [[buffer(4)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= args.channels || t >= args.seq_len) {
        return;
    }
    const uint hist_len = args.hist_len;
    const uint seq_len = args.seq_len;
    const uint channels = args.channels;
    const uint kernel_size = args.kernel_size;
    const uint b = gid.z;

    device const float * xb = x + b * seq_len * channels;
    device const float * hb = history + b * hist_len * channels;
    device const float * wc = weight + c * kernel_size;

    float acc = 0.0f;
    for (uint k = 0; k < kernel_size; ++k) {
        const uint idx = t + k; // index into conceptual padded = history ++ x
        const float val = (idx < hist_len)
            ? hb[idx * channels + c]
            : xb[(idx - hist_len) * channels + c];
        acc += val * wc[k];
    }
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    out[(b * seq_len + t) * channels + c] = acc / (1.0f + exp(-acc));
}

kernel void kernel_gdn_causal_conv1d_state_f32(
        device const float * x           [[buffer(0)]],  // [b, seq_len, channels]
        device const float * history     [[buffer(1)]],  // [b, hist_len, channels]
        device float       * new_state   [[buffer(2)]],  // [b, hist_len, channels], functional
        constant gdn_conv1d_args & args [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint s = gid.y;
    if (c >= args.channels || s >= args.hist_len) {
        return;
    }
    const uint hist_len = args.hist_len;
    const uint seq_len = args.seq_len;
    const uint channels = args.channels;
    const uint b = gid.z;

    device const float * xb = x + b * seq_len * channels;
    device const float * hb = history + b * hist_len * channels;

    const uint idx = seq_len + s; // index into conceptual padded = history ++ x
    const float val = (idx < hist_len)
        ? hb[idx * channels + c]
        : xb[(idx - hist_len) * channels + c];
    new_state[(b * hist_len + s) * channels + c] = val;
}

// DeltaNet-preprocessing fusion, Kernel B (gated-DeltaNet MTP-verify-step
// investigation, see yggdrasil/ratatoskr/DESIGN.md's native-MTP section,
// "DeltaNet-preprocessing fusion" subsection): the elementwise gating tail,
// two independent sub-kernels replacing ~17 separate candle dispatches
// (L2-normalize q and k, scale q, softplus-decay g, sigmoid beta) with two.
//
// kernel_gdn_l2_normalize_scale_f32: out[b][t][h][:] = scale *
// x[b][t][h][:] / sqrt(sum_d x[b][t][h][d]^2 + eps), matching
// SSMWeights::l2_normalize's own eps placement (added to the sum of
// squares, not the mean) exactly -- do not substitute a generic RMS-norm
// kernel here, the epsilon semantics differ. One thread per (b, t, h),
// looping over `dim` twice (sum-of-squares, then the normalized write) --
// re-reads `x` rather than caching it in a local array, matching this
// file's existing preference for simple per-thread loops over
// threadgroup-memory tricks (dim is small, ~64-256, so the extra read is
// cheap). Called once for q (scale = 1/sqrt(state_size)) and once for k
// (scale = 1.0) -- two dispatches of the same kernel, not two kernels.
struct gdn_l2norm_args {
    uint seq_len;
    uint heads;
    uint dim;
    float scale;
    float eps;
};

kernel void kernel_gdn_l2_normalize_scale_f32(
        device const float * x    [[buffer(0)]],  // [b, seq_len, heads, dim]
        device float       * out  [[buffer(1)]],  // [b, seq_len, heads, dim]
        constant gdn_l2norm_args & args [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint t = gid.y;
    if (h >= args.heads || t >= args.seq_len) {
        return;
    }
    const uint dim = args.dim;
    const uint b = gid.z;
    const uint row = (b * args.seq_len + t) * args.heads + h;
    device const float * xr = x + row * dim;
    device float       * outr = out + row * dim;

    float sum_sq = 0.0f;
    for (uint d = 0; d < dim; ++d) {
        const float v = xr[d];
        sum_sq += v * v;
    }
    const float inv_norm = args.scale / sqrt(sum_sq + args.eps);
    for (uint d = 0; d < dim; ++d) {
        outr[d] = xr[d] * inv_norm;
    }
}

// kernel_gdn_decay_beta_gate_f32: fuses the decay-gate softplus chain and
// the beta sigmoid, matching SSMWeights::forward's own math exactly:
//   g[b][t][h]    = ssm_a[h] * log(exp(alpha_logits[b][t][h] + dt_bias[h]) + 1)
//   beta[b][t][h] = sigmoid(beta_logits[b][t][h])
// `dt_bias`/`ssm_a` are indexed directly by head (`[heads]`, not
// pre-broadcast to `[b, seq_len, heads]` -- eliminates the
// candle-side `broadcast_as` calls entirely, not just the elementwise math
// around them). One thread per (b, t, h), no reduction, purely elementwise.
struct gdn_decay_beta_args {
    uint seq_len;
    uint heads;
};

kernel void kernel_gdn_decay_beta_gate_f32(
        device const float * alpha_logits [[buffer(0)]],  // [b, seq_len, heads]
        device const float * dt_bias      [[buffer(1)]],  // [heads]
        device const float * ssm_a        [[buffer(2)]],  // [heads]
        device const float * beta_logits  [[buffer(3)]],  // [b, seq_len, heads]
        device float       * g_out        [[buffer(4)]],  // [b, seq_len, heads]
        device float       * beta_out     [[buffer(5)]],  // [b, seq_len, heads]
        constant gdn_decay_beta_args & args [[buffer(6)]],
        uint3 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint t = gid.y;
    if (h >= args.heads || t >= args.seq_len) {
        return;
    }
    const uint b = gid.z;
    const uint idx = (b * args.seq_len + t) * args.heads + h;

    const float softplus = log(exp(alpha_logits[idx] + dt_bias[h]) + 1.0f);
    g_out[idx] = ssm_a[h] * softplus;
    beta_out[idx] = 1.0f / (1.0f + exp(-beta_logits[idx]));
}
