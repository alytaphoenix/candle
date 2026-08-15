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
