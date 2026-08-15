use candle_core::{
    bail,
    quantized::{self, GgmlDType},
    test_device,
    test_utils::to_vec2_round,
    DType, Device, IndexOp, Module, Result, Tensor, Var,
};
use quantized::{k_quants, GgmlType};
use rand::prelude::*;
use std::borrow::Cow;

const GGML_TEST_SIZE: usize = 32 * 128;

const GGML_MAX_QUANTIZATION_TOTAL_ERROR: f32 = 0.002;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS: f32 = 0.0040;
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

fn test_matmul(
    device: &Device,
    (b, m, n, k): (usize, usize, usize, usize),
    dtype: GgmlDType,
) -> Result<()> {
    if (device.is_cuda() || device.is_metal())
        && (dtype == GgmlDType::Q8_1 || dtype == GgmlDType::Q8K)
    {
        return Ok(());
    }

    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device)?;
    let rhs = Tensor::from_slice(&rhs, (k, n), device)?;
    let mm = lhs.matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;
    let error = error / (b * m * n) as f32;
    assert!(
        error <= 0.02,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_matmul_mm() -> Result<()> {
    let dtype = GgmlDType::Q8_0;
    let device = Device::new_metal(0)?;

    let m = 32;
    let n = 32;
    let k = 32;
    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (1, 1, k, n), &device)?.repeat((5, 20, 1, 1))?;
    let mm = lhs.broadcast_matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&lhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&rhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;

    let error = error / res.elem_count() as f32;
    assert!(
        error <= 0.001,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

// Covers the Metal arm of QTensor::indexed_moe_forward (the CUDA-only-until-
// now MoE expert-dispatch entry point candle_metal_kernels::
// call_quantized_matmul_mm_id feeds into) with a *real* quantized weight,
// not the F32 dtype the kernel-level tests use -- this is the layer that
// extracts shapes/strides/dtypes from real QTensor/Tensor storage, which
// the kernel-level tests can't exercise. Compares against dequantize-then-
// manual-index-then-matmul, the same style of reference candle's own
// quantized matmul tests use, so real Q4_0 quantization error is expected
// and tolerated -- this isn't testing quantization accuracy, just that the
// indexed routing lands on the right rows.
#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal() -> Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = GgmlDType::Q4_0;

    let n_expert = 3usize;
    let n_out = 64usize;
    let n_in = 64usize;
    let batch = 5usize;
    let topk = 2usize;

    let w_data: Vec<f32> = (0..n_expert * n_out * n_in)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let weight = Tensor::from_slice(&w_data, (n_expert, n_out, n_in), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, dtype)?;
    // Reference uses the dequantized (i.e. quantization-error-including)
    // weight, not w_data, so this isn't also asserting quantization
    // accuracy -- that's covered elsewhere.
    let dequant = qweight.dequantize(&device)?.to_vec3::<f32>()?;

    let x_data: Vec<f32> = (0..batch * topk * n_in)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.02)
        .collect();
    let x = Tensor::from_slice(&x_data, (batch, topk, n_in), &device)?;

    let ids_data: Vec<u32> = (0..batch * topk)
        .map(|i| ((i * 7 + i / topk) % n_expert) as u32)
        .collect();
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;

    let got = qweight.indexed_moe_forward(&x, &ids)?;
    assert_eq!(got.dims(), &[batch, topk, n_out]);
    let got = got.to_vec3::<f32>()?;

    for t in 0..batch {
        for s in 0..topk {
            let e = ids_data[t * topk + s] as usize;
            for j in 0..n_out {
                let mut acc = 0f32;
                for k in 0..n_in {
                    acc += dequant[e][j][k] * x_data[t * topk * n_in + s * n_in + k];
                }
                let diff = (got[t][s][j] - acc).abs();
                assert!(
                    diff <= 1e-2 + 1e-2 * acc.abs(),
                    "mismatch at token {t} slot {s} out {j}: got {}, expected {acc} (diff {diff})",
                    got[t][s][j]
                );
            }
        }
    }

    Ok(())
}

// Same shape/reference discipline as indexed_moe_forward_metal above, but
// batch=1 -- the real decode shape, and the one that now routes through
// call_quantized_matmul_mv_id instead of mm_id for a dtype
// `candle_metal_kernels::mv_id_eligible` covers (see
// QMetalStorage::indexed_moe_forward's `use_mv` branch, ratatoskr/DESIGN.md
// section 15 "Phase 2"). Exercises the new kernel through the real public
// API with real quantized weights, not just the kernel-level differential
// spike.
#[cfg(feature = "metal")]
fn indexed_moe_forward_metal_decode_uses_mv_id(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_metal(0)?;

    let n_expert = 3usize;
    let n_out = 64usize;
    // 256, not 64: K-quant dtypes (Q4K/Q6K/Q2K/Q5K/Q3K) require their last
    // dim divisible by their own block size (256) -- a QTensor::quantize
    // constraint, unrelated to and stricter than mv_id's own nth0*nth1
    // minimum (64 at most across all eight covered dtypes), which 256
    // clears comfortably too.
    let n_in = 256usize;
    let batch = 1usize;
    let topk = 2usize;

    let w_data: Vec<f32> = (0..n_expert * n_out * n_in)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let weight = Tensor::from_slice(&w_data, (n_expert, n_out, n_in), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, dtype)?;
    let dequant = qweight.dequantize(&device)?.to_vec3::<f32>()?;

    let x_data: Vec<f32> = (0..batch * topk * n_in)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.02)
        .collect();
    let x = Tensor::from_slice(&x_data, (batch, topk, n_in), &device)?;

    let ids_data: Vec<u32> = (0..batch * topk)
        .map(|i| ((i * 7 + i / topk) % n_expert) as u32)
        .collect();
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;

    let got = qweight.indexed_moe_forward(&x, &ids)?;
    assert_eq!(got.dims(), &[batch, topk, n_out]);
    let got = got.to_vec3::<f32>()?;

    for t in 0..batch {
        for s in 0..topk {
            let e = ids_data[t * topk + s] as usize;
            for j in 0..n_out {
                let mut acc = 0f32;
                for k in 0..n_in {
                    acc += dequant[e][j][k] * x_data[t * topk * n_in + s * n_in + k];
                }
                let diff = (got[t][s][j] - acc).abs();
                assert!(
                    diff <= 1e-2 + 1e-2 * acc.abs(),
                    "{dtype:?} mismatch at token {t} slot {s} out {j}: got {}, expected {acc} (diff {diff})",
                    got[t][s][j]
                );
            }
        }
    }

    Ok(())
}

// mv_id_eligible's exact eight dtypes -- not a sample of them. Q4_K and
// Q6_K are the two ratatoskr/DESIGN.md section 15 calls mandatory (this
// stack's real models); Q4_0/Q2_K round out tuning-class coverage. Q5_K,
// Q3_K, Q5_0, Q5_1 joined in "Decode throughput optimization" (Phase 1) --
// every currently-cached target model's MoE down- or up-projection uses one
// of these four, so this is the correctness gate that unblocks each dtype's
// entry in mv_id_eligible's allow-list. Each is its own #[test] (rather
// than a loop inside one) so a failure names the specific dtype instead of
// requiring a debugger to find it.
#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q4k() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q4K)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q6k() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q6K)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q4_0() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q4_0)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q2k() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q2K)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q5k() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q5K)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q3k() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q3K)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q5_0() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q5_0)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_decode_uses_mv_id_q5_1() -> Result<()> {
    indexed_moe_forward_metal_decode_uses_mv_id(GgmlDType::Q5_1)
}

// Metal MoE Phase 3 (chunked mm_id, see ratatoskr/DESIGN.md section 15):
// covers the real production path this phase exists for -- a prefill batch
// large enough that a single call_quantized_matmul_mm_id dispatch would
// exceed the device's threadgroup-memory budget (found live via a real
// ~2594-token prompt against qwen3moe-test) -- through the real public
// indexed_moe_forward API with real quantized weights, not just the
// kernel-level bit-exact differential spike (candle-metal-kernels' own
// tests.rs). batch=2000 at topk=8 (this stack's real top-k) gives
// nei0*nei1=16000, comfortably above the ~6144 ceiling every Apple Silicon
// device reports to date (32KB threadgroup budget) -- this exact test would
// fail before Phase 3's chunking with "required threadgroup memory ...
// exceeds this device's max". Spot-checks a few tokens rather than the
// full batch (the chunking mechanism itself is already validated row-by-row,
// bit-exact, at the kernel level) -- this test's job is proving the real
// public API, with real quantized weights, produces correct values at a
// batch size that would have failed outright before this phase.
#[cfg(feature = "metal")]
fn indexed_moe_forward_metal_prefill_above_ceiling_uses_chunking(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_metal(0)?;

    let n_expert = 16usize;
    let n_out = 8usize;
    let n_in = 256usize; // K-quant block-size requirement, see the mv_id test above
    let batch = 2000usize;
    let topk = 8usize;

    let w_data: Vec<f32> = (0..n_expert * n_out * n_in)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let weight = Tensor::from_slice(&w_data, (n_expert, n_out, n_in), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, dtype)?;
    let dequant = qweight.dequantize(&device)?.to_vec3::<f32>()?;

    let x_data: Vec<f32> = (0..batch * topk * n_in)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.02)
        .collect();
    let x = Tensor::from_slice(&x_data, (batch, topk, n_in), &device)?;

    let ids_data: Vec<u32> = (0..batch * topk)
        .map(|i| ((i * 7 + i / topk) % n_expert) as u32)
        .collect();
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;

    let got = qweight.indexed_moe_forward(&x, &ids)?;
    assert_eq!(got.dims(), &[batch, topk, n_out]);
    let got = got.to_vec3::<f32>()?;

    for &t in &[0usize, batch / 2, batch - 1] {
        for s in 0..topk {
            let e = ids_data[t * topk + s] as usize;
            for j in 0..n_out {
                let mut acc = 0f32;
                for k in 0..n_in {
                    acc += dequant[e][j][k] * x_data[t * topk * n_in + s * n_in + k];
                }
                let diff = (got[t][s][j] - acc).abs();
                assert!(
                    diff <= 1e-2 + 1e-2 * acc.abs(),
                    "{dtype:?} mismatch at token {t} slot {s} out {j}: got {}, expected {acc} (diff {diff})",
                    got[t][s][j]
                );
            }
        }
    }

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_prefill_above_ceiling_uses_chunking_q4k() -> Result<()> {
    indexed_moe_forward_metal_prefill_above_ceiling_uses_chunking(GgmlDType::Q4K)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_forward_metal_prefill_above_ceiling_uses_chunking_q6k() -> Result<()> {
    indexed_moe_forward_metal_prefill_above_ceiling_uses_chunking(GgmlDType::Q6K)
}

/// `QTensor::fwd`'s own small-multi-token-batch dispatch (extends the
/// existing `batch == 1` mat-vec fast path to `batch <= 8`, routing through
/// `fwd_mv`'s single batched `call_quantized_matmul_mv_t` dispatch instead
/// of `mm_t`): compares a real `batch`-row `QMatMul::forward` call against
/// `batch` independent single-row calls on the same data, which must be
/// bit-exact -- every row is computed identically either way (the kernel's
/// own `src1 + r1*ne10 + ...` indexing, confirmed against the actual
/// `quantized.metal` source for these dtypes, is per-row-independent by
/// construction), so any difference is a real stride/offset bug in the
/// batched dispatch's own plumbing, not an accumulation-order artifact.
#[cfg(feature = "metal")]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(
    dtype: GgmlDType,
) -> Result<()> {
    let device = Device::new_metal(0)?;
    let (n, k) = (256usize, 512usize); // block-aligned for every dtype under test (min block size 256)

    let weight: Vec<f32> = (0..n * k).map(|v| ((v * 37 + 11) % 997) as f32 * 0.001).collect();
    let weight = Tensor::from_slice(&weight, (n, k), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qweight)?;

    for batch in [2usize, 3, 4, 8] {
        let x: Vec<f32> = (0..batch * k).map(|v| ((v * 13 + 5) % 991) as f32 * 0.001).collect();
        let x = Tensor::from_slice(&x, (batch, k), &device)?;

        let batched = matmul.forward(&x)?.to_vec2::<f32>()?;
        let mut sequential = Vec::with_capacity(batch);
        for row in 0..batch {
            let x_row = x.narrow(0, row, 1)?;
            let out_row = matmul.forward(&x_row)?.to_vec2::<f32>()?;
            sequential.push(out_row[0].clone());
        }

        assert_eq!(
            batched, sequential,
            "dtype={dtype:?} batch={batch}: batched QMatMul::forward diverges from {batch} \
             independent single-row calls on the same data -- must be bit-exact"
        );
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q4k() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q4K)
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q6k() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q6K)
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q4_0() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q4_0)
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q5k() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q5K)
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q3k() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q3K)
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q4_1() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q4_1)
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q5_0() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q5_0)
}

// Regression guard, not a "safe dtype" test: `Q5_1` is excluded from
// `fwd()`'s own `m > 1` routing entirely (not just from `fwd_mv`'s internal
// batched-dispatch dtype list) -- a real differential test found
// `fwd_mv`'s pre-existing, unmodified per-row loop itself silently zeroes
// row 0's output for this dtype at `m == 4` (rows 1..m unaffected, a
// distinct failure shape from `Q2K`'s own scattered-zero pattern). So at
// `m > 1`, `Q5_1` now always uses `mm_t` -- comparing that against `m`
// independent `fwd_mv` (`mv_t`) calls is a genuine cross-kernel comparison
// (different accumulation order), hence tolerance-based here, unlike the
// bit-exact dtype tests above which compare the *same* kernel batched vs.
// looped.
#[cfg(feature = "metal")]
#[test]
fn qmatmul_mm_t_matches_sequential_single_row_mv_calls_within_tolerance_q5_1() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (n, k) = (256usize, 512usize);
    let weight: Vec<f32> = (0..n * k).map(|v| ((v * 37 + 11) % 997) as f32 * 0.001).collect();
    let weight = Tensor::from_slice(&weight, (n, k), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, GgmlDType::Q5_1)?;
    let matmul = quantized::QMatMul::from_qtensor(qweight)?;

    for batch in [2usize, 4, 8] {
        let x: Vec<f32> = (0..batch * k).map(|v| ((v * 13 + 5) % 991) as f32 * 0.001).collect();
        let x = Tensor::from_slice(&x, (batch, k), &device)?;
        let got = matmul.forward(&x)?.to_vec2::<f32>()?; // mm_t (batch > 1 always routes here for Q5_1)

        let mut expected = Vec::with_capacity(batch);
        for row in 0..batch {
            let x_row = x.narrow(0, row, 1)?;
            let out_row = matmul.forward(&x_row)?.to_vec2::<f32>()?; // fwd_mv (m == 1, always safe)
            expected.push(out_row[0].clone());
        }
        for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            for (c, (gv, ev)) in g.iter().zip(e.iter()).enumerate() {
                let diff = (gv - ev).abs();
                assert!(
                    diff <= 1e-3 + 1e-3 * ev.abs(),
                    "batch={batch} row={r} col={c}: Q5_1 mm_t vs. sequential mv_t mismatch, \
                     got={gv} expected={ev} (diff {diff})"
                );
            }
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact_q8_0() -> Result<()> {
    qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact(GgmlDType::Q8_0)
}

// No qmatmul_batched_..._q8_1 test: `kernel_mul_mv_q8_1_f32` does not exist
// anywhere in this fork's vendored `metal_src/quantized.metal` -- Q8_1's
// mat-vec Metal path is completely non-functional today, independent of
// and pre-dating this change (the identical kernel-load failure occurs at
// `batch == 1` through the untouched loop path too, confirmed while
// building this test). Not a regression this change introduces, not fixed
// here -- Q8_1 isn't used by any real model this dispatch path serves.

// Regression guard, not a "safe dtype" test: `Q2K` is excluded from
// `fwd()`'s own `m > 1` routing entirely (not just `fwd_mv`'s internal
// batched-dispatch dtype list) -- a real differential test found `Q2K`'s
// batched `mv_t` dispatch produces scattered zeros/NaN/garbage at m == 2,
// and separately that its pre-existing, unmodified per-row loop *also*
// breaks at larger m (the pre-existing `qmm_batch` test, stacking up to
// m == 12, failed once `Q2K` could reach `fwd_mv` for m > 1 at all). Root
// cause not chased further -- `Q2K` isn't needed by the target models
// through this dispatch path. So at `m > 1`, `Q2K` now always uses `mm_t`
// -- comparing that against `m` independent `fwd_mv` (`mv_t`) calls is a
// genuine cross-kernel comparison (different accumulation order), hence
// tolerance-based here, same treatment as `Q5_1` above.
#[cfg(feature = "metal")]
#[test]
fn qmatmul_mm_t_matches_sequential_single_row_mv_calls_within_tolerance_q2k() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (n, k) = (256usize, 512usize);
    let weight: Vec<f32> = (0..n * k).map(|v| ((v * 37 + 11) % 997) as f32 * 0.001).collect();
    let weight = Tensor::from_slice(&weight, (n, k), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, GgmlDType::Q2K)?;
    let matmul = quantized::QMatMul::from_qtensor(qweight)?;

    for batch in [2usize, 4, 8] {
        let x: Vec<f32> = (0..batch * k).map(|v| ((v * 13 + 5) % 991) as f32 * 0.001).collect();
        let x = Tensor::from_slice(&x, (batch, k), &device)?;
        let got = matmul.forward(&x)?.to_vec2::<f32>()?; // mm_t (batch > 1 always routes here for Q2K)

        let mut expected = Vec::with_capacity(batch);
        for row in 0..batch {
            let x_row = x.narrow(0, row, 1)?;
            let out_row = matmul.forward(&x_row)?.to_vec2::<f32>()?; // fwd_mv (m == 1, always safe)
            expected.push(out_row[0].clone());
        }
        for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            for (c, (gv, ev)) in g.iter().zip(e.iter()).enumerate() {
                let diff = (gv - ev).abs();
                assert!(
                    diff <= 1e-3 + 1e-3 * ev.abs(),
                    "batch={batch} row={r} col={c}: Q2K mm_t vs. sequential mv_t mismatch, \
                     got={gv} expected={ev} (diff {diff})"
                );
            }
        }
    }
    Ok(())
}

/// Regression guard for the batch-size gate's own boundary: `batch == 9`
/// must fall through to the unchanged `mm_t` path, not `fwd_mv` -- a real
/// tolerance-based (not bit-exact, different accumulation order) check
/// that the two dispatch choices still agree, at the exact boundary where
/// a future threshold change is most likely to introduce an off-by-one.
#[cfg(feature = "metal")]
#[test]
fn qmatmul_mv_and_mm_agree_at_the_batch_size_gate_boundary() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (n, k) = (256usize, 512usize);
    let weight: Vec<f32> = (0..n * k).map(|v| ((v * 37 + 11) % 997) as f32 * 0.001).collect();
    let weight = Tensor::from_slice(&weight, (n, k), &device)?;
    let qweight = quantized::QTensor::quantize(&weight, GgmlDType::Q4K)?;
    let matmul = quantized::QMatMul::from_qtensor(qweight)?;

    for batch in [8usize, 9] {
        let x: Vec<f32> = (0..batch * k).map(|v| ((v * 13 + 5) % 991) as f32 * 0.001).collect();
        let x = Tensor::from_slice(&x, (batch, k), &device)?;
        let got = matmul.forward(&x)?.to_vec2::<f32>()?;

        // Independent per-row reference, computed via the always-available
        // batch=1 path (never routes through the new batched dispatch),
        // so this is a genuine oracle at both sides of the boundary.
        let mut expected = Vec::with_capacity(batch);
        for row in 0..batch {
            let x_row = x.narrow(0, row, 1)?;
            let out_row = matmul.forward(&x_row)?.to_vec2::<f32>()?;
            expected.push(out_row[0].clone());
        }
        for (r, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            for (c, (gv, ev)) in g.iter().zip(e.iter()).enumerate() {
                let diff = (gv - ev).abs();
                assert!(
                    diff <= 1e-3 + 1e-3 * ev.abs(),
                    "batch={batch} row={r} col={c}: mismatch at gate boundary, got={gv} expected={ev} (diff {diff})"
                );
            }
        }
    }
    Ok(())
}

fn quantized_matmul(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0,
            341876.0, 994283.0, 1655709.0, 2301518.0
        ]
    );
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        mm.to_vec2::<f32>()?,
        &[
            [85344.0, 214368.0, 343392.0, 472416.0],
            [214368.0, 605536.0, 996704.0, 1387872.0],
            [343392.0, 996704.0, 1650016.0, 2303328.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84946.0, 214126.0, 344757.0, 473798.0],
                [213458.0, 604350.0, 1000469.0, 1387990.0],
                [341970.0, 994574.0, 1656181.0, 2302182.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84866.0, 214045.0, 344676.0, 473707.0],
                [213425.0, 604313.0, 1000431.0, 1387960.0],
                [342030.0, 994630.0, 1656248.0, 2302250.0]
            ]
        ),
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [85120.0, 214562.0, 345455.0, 474748.0],
                [213475.0, 604465.0, 1000686.0, 1388317.0],
                [341876.0, 994283.0, 1655709.0, 2301518.0]
            ]
        ),
    }
    test_matmul(device, (1, 3, 4, 256), GgmlDType::Q4_0)?;
    Ok(())
}

fn quantized_matmul_neg(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k))
        .map(|v| v as f32 - (m * k) as f32 / 2.0)
        .collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..k * n)
        .map(|v| v as f32 - (k * n) as f32 / 3.0)
        .collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0,
            -196472.0, 63012.0, 324585.0, 587902.0
        ]
    );
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        to_vec2_round(&mm, 0)?,
        &[
            [244064.0, -20128.0, -284320.0, -548512.0],
            [23563.0, 21515.0, 19467.0, 17419.0],
            [-196939.0, 63157.0, 323253.0, 583349.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        // `lhs` is (m=3, k=64), Q4_0 -- now routes through `fwd_mv`'s
        // batched mat-vec dispatch instead of `mm_t` (the `use_mv`-style
        // batch<=8 extension). Different accumulation order than `mm_t`
        // (same class of expected float noise as CUDA's own MMVQ-vs-MMQ
        // divergence noted below, not a correctness regression -- `qmm_b`/
        // `qmatmul_batched_mv_matches_sequential_single_row_calls_bit_exact`
        // are this dispatch choice's own real correctness oracles), so
        // these golden values were re-captured against the new path.
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243666.0, -19714.0, -285433.0, -550453.0],
                [23782.0, 21654.0, 19400.0, 18369.0],
                [-196102.0, 63022.0, 324233.0, 587191.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243740.0, -19762.0, -285476.0, -550498.0],
                [23774.0, 21645.0, 19395.0, 18364.0],
                [-196045.0, 63030.0, 324120.0, 587079.0]
            ]
        ),
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243524.0, -19596.0, -285051.0, -549815.0],
                [23777.0, 21651.0, 19398.0, 18367.0],
                [-196472.0, 63012.0, 324585.0, 587902.0]
            ]
        ),
    }
    let lhs2 = Tensor::stack(&[&lhs, &lhs], 0)?;
    let res2 = matmul.forward(&lhs2)?;
    let res2 = res2.i(1)?;
    let diff = (&res - res2)?.abs()?.mean_all()?.to_vec0::<f32>()? / res.elem_count() as f32;
    if device.is_cuda() {
        assert!(diff < 0.1);
    } else {
        assert!(diff < 0.96);
    }
    Ok(())
}

fn qmm_batch(dev: &Device) -> Result<()> {
    let (lhs, rhs, _mm) = get_random_tensors(2, 256, 6, dev)?;
    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;
    assert_eq!(mm.shape().dims(), [2, 6]);
    let lhs2 = Tensor::cat(&[&lhs, &lhs], 0)?;
    let mm2 = rhs.forward(&lhs2)?;
    assert_eq!(mm2.shape().dims(), [4, 6]);
    let diff2 = (mm2.i(2..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff2, 0.0);
    let lhs3 = Tensor::cat(&[&lhs2, &lhs], 0)?;
    let mm3 = rhs.forward(&lhs3)?;
    assert_eq!(mm3.shape().dims(), [6, 6]);
    let diff3 = (mm3.i(2..4)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff3, 0.0);
    let diff3 = (mm3.i(4..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff3, 0.0);
    let lhs4 = Tensor::cat(&[&lhs3, &lhs3], 0)?;
    let mm4 = rhs.forward(&lhs4)?;
    assert_eq!(mm4.shape().dims(), [12, 6]);
    let diff4 = (mm4.i(..6)? - &mm3)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        // We use different fused kernels (MMVQ for batch<=8, MMQ for batch>8) on CUDA which accumulate differently than dequantize-then-matmul.
        // This can lead to small numerical differences especially for low-bit quants.
        assert!(0. < diff4 && diff4 < 0.5)
    } else {
        assert_eq!(diff4, 0.0)
    };
    let diff4 = (mm4.i(6..)? - &mm4.i(..6)?)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff4, 0.0);
    Ok(())
}

test_device!(quantized_matmul, qmm_cpu, qmm_cuda, qmm_metal);
test_device!(quantized_matmul_neg, qmm_n_cpu, qmm_n_cuda, qmm_n_metal);
test_device!(qmm_batch, qmm_b_cpu, qmm_b_cuda, qmm_b_metal);

fn quantize_q4_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();

    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        dst.to_vec1::<f32>()?,
        &[
            -0.0, -0.0, 3.875, 3.875, 3.875, 3.875, 7.75, 7.75, 7.75, 7.75, 11.625, 11.625, 11.625,
            11.625, 15.5, 15.5, 15.5, 15.5, 19.375, 19.375, 19.375, 19.375, 23.25, 23.25, 23.25,
            23.25, 27.125, 27.125, 27.125, 27.125, 31.0, 31.0, 31.5, 31.5, 31.5, 31.5, 39.375,
            39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 47.25, 47.25, 47.25, 47.25,
            47.25, 47.25, 47.25, 47.25, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125,
            55.125, 63.0, 63.0, 63.0, 63.0, 59.375, 59.375, 71.25, 71.25, 71.25, 71.25, 71.25,
            71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 83.125, 83.125, 83.125, 83.125,
            83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 95.0, 95.0, 95.0, 95.0,
            95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0,
            127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q4_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 0.0, 2.066, 2.066, 4.133, 4.133, 6.199, 6.199, 8.266, 8.266, 10.332, 10.332,
            12.398, 12.398, 14.465, 14.465, 16.531, 16.531, 18.598, 18.598, 20.664, 20.664, 22.73,
            22.73, 24.797, 24.797, 26.863, 26.863, 28.93, 28.93, 30.996, 30.996, 32.0, 32.0,
            34.066, 34.066, 36.133, 36.133, 38.199, 38.199, 40.266, 40.266, 42.332, 42.332, 44.398,
            44.398, 46.465, 46.465, 48.531, 48.531, 50.598, 50.598, 52.664, 52.664, 54.73, 54.73,
            56.797, 56.797, 58.863, 58.863, 60.93, 60.93, 62.996, 62.996, 64.0, 64.0, 66.066,
            66.066, 68.133, 68.133, 70.199, 70.199, 72.266, 72.266, 74.332, 74.332, 76.398, 76.398,
            78.465, 78.465, 80.531, 80.531, 82.598, 82.598, 84.664, 84.664, 86.73, 86.73, 88.797,
            88.797, 90.863, 90.863, 92.93, 92.93, 94.996, 94.996, 96.0, 96.0, 98.066, 98.066,
            100.133, 100.133, 102.199, 102.199, 104.266, 104.266, 106.332, 106.332, 108.398,
            108.398, 110.465, 110.465, 112.531, 112.531, 114.598, 114.598, 116.664, 116.664,
            118.73, 118.73, 120.797, 120.797, 122.863, 122.863, 124.93, 124.93, 126.996, 126.996
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            -0.0, 1.938, 1.938, 3.875, 3.875, 5.813, 5.813, 7.75, 7.75, 9.688, 9.688, 11.625,
            11.625, 13.563, 13.563, 15.5, 15.5, 17.438, 17.438, 19.375, 19.375, 21.313, 21.313,
            23.25, 23.25, 25.188, 25.188, 27.125, 27.125, 29.063, 29.063, 31.0, 31.5, 31.5, 35.438,
            35.438, 35.438, 35.438, 39.375, 39.375, 39.375, 39.375, 43.313, 43.313, 43.313, 43.313,
            47.25, 47.25, 47.25, 47.25, 51.188, 51.188, 51.188, 51.188, 55.125, 55.125, 55.125,
            55.125, 59.063, 59.063, 59.063, 59.063, 63.0, 63.0, 65.313, 65.313, 65.313, 65.313,
            65.313, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 77.188, 77.188, 77.188, 77.188,
            77.188, 77.188, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 89.063, 89.063, 89.063,
            89.063, 89.063, 89.063, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 103.188, 103.188,
            103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 119.063, 119.063, 119.063, 119.063,
            119.063, 119.063, 119.063, 119.063, 127.0, 127.0, 127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
            86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn get_test_vector2(bound: f32, size: usize, device: &Device) -> Result<Tensor> {
    assert!(
        size.is_multiple_of(crate::quantized::k_quants::QK_K),
        "size must be a multiple of {}",
        crate::quantized::k_quants::QK_K
    );

    let src = (0..size)
        .map(|v| (v as f32 - size as f32 / 2.) * bound / (size as f32 / 2.))
        .collect::<Vec<_>>();
    assert_eq!([src[0], src[size / 2]], [-bound, 0.0]);
    Tensor::from_vec(src, (size,), device)
}

/// Round a vector
fn round_vector(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>()
}

fn compare_with_error(values: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (value, expected_value)) in values.iter().zip(expected.iter()).enumerate() {
        let difference = (value - expected_value).abs();

        assert!(
            difference < tolerance,
            "Error at index {i}: value = {value}, expected = {expected_value}. Difference = {difference} exceeds tolerance = {tolerance}."
        );
    }
}

/// Creates a vector similar to the ones used in GGML unit tests:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// Calculates the root mean square error between two vectors
fn calculate_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    sum / a.len() as f32
}

/// Similar to the GGML quantization unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L43-L50
fn ggml_quantization_error_test(dtype: GgmlDType, device: &Device, max_error: f32) -> Result<()> {
    let src = create_ggml_like_vector(0.0);
    let src = Tensor::from_slice(&src, (GGML_TEST_SIZE,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    let error = calculate_rmse(&src.to_vec1::<f32>()?, &dst.to_vec1::<f32>()?);
    if error > max_error {
        bail!(
            "Quantization error {} exceeds max error {}",
            error,
            max_error
        );
    }
    Ok(())
}

#[test]
fn imatrix_quantize_q6k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q6K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q6K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q5k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q5K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q5K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q4k() -> Result<()> {
    // let data =
    //     quantized::imatrix_file::load_imatrix("../Llama-3.2-3B-Instruct.imatrix").unwrap();
    // for (name, weights) in &data {
    //     println!("{name}, {} elems", weights.len());
    // }
    // dbg!(&data["blk.0.attn_q.weight"].len());

    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q4K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q4K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q3k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q3K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q3K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q2k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q2K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q2K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

fn quantize_q2k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q2K;

    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.1);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.499, -0.366, -0.249, 0.0, 0.295, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 6.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS)?;
    Ok(())
}

fn quantize_q3k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q3K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.03);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.493, -0.37, -0.243, -0.0, 0.292, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 3.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS)?;
    Ok(())
}

fn quantize_q4k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q4K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.017);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.288, 0.498]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 4.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q5K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.009);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.279, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q6k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q6K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.497, -0.372, -0.25, -0.0, 0.284, 0.5]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q8k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q8K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.375, -0.25, -0.0, 0.281, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 0.6);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

test_device!(
    quantize_q4_0,
    quantize_q4_0_cpu,
    quantize_q4_0_cuda,
    quantize_q4_0_metal
);
test_device!(
    quantize_q4_1,
    quantize_q4_1_cpu,
    quantize_q4_1_cuda,
    quantize_q4_1_metal
);
test_device!(
    quantize_q5_0,
    quantize_q5_0_cpu,
    quantize_q5_0_cuda,
    quantize_q5_0_metal
);
test_device!(
    quantize_q5_1,
    quantize_q5_1_cpu,
    quantize_q5_1_cuda,
    quantize_q5_1_metal
);
test_device!(
    quantize_q2k,
    quantize_q2k_cpu,
    quantize_q2k_cuda,
    quantize_q2k_metal
);
test_device!(
    quantize_q3k,
    quantize_q3k_cpu,
    quantize_q3k_cuda,
    quantize_q3k_metal
);
test_device!(
    quantize_q4k,
    quantize_q4k_cpu,
    quantize_q4k_cuda,
    quantize_q4k_metal
);
test_device!(
    quantize_q5k,
    quantize_q5k_cpu,
    quantize_q5k_cuda,
    quantize_q5k_metal
);
test_device!(
    quantize_q6k,
    quantize_q6k_cpu,
    quantize_q6k_cuda,
    quantize_q6k_metal
);
test_device!(
    quantize_q8k,
    quantize_q8k_cpu,
    quantize_q8k_cuda,
    quantize_q8k_metal
);

/// Very simple dot product implementation
fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns the error achieved by the GGML matmul unit test.
fn ggml_reference_matmul_error(dtype: GgmlDType) -> Result<f32> {
    let err = match dtype {
        GgmlDType::F32 => 0.000000,
        GgmlDType::F16 => 0.000010,
        GgmlDType::BF16 => 0.000200,
        GgmlDType::Q2K => 0.004086,
        GgmlDType::Q3K => 0.016148,
        GgmlDType::Q4K => 0.002425,
        GgmlDType::Q5K => 0.000740,
        GgmlDType::Q6K => 0.000952,
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4_1 => 0.008,
        GgmlDType::Q5_0 => 0.001353,
        GgmlDType::Q5_1 => 0.00149,
        GgmlDType::Q8_0 => 0.000092,
        GgmlDType::Q8_1 => 0.000092,

        // Not from the ggml repo.
        GgmlDType::Q8K => 0.00065,
    };
    Ok(err)
}

/// Similar to the GGML matmul unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
fn ggml_matmul_error_test<T: GgmlType>() -> Result<()> {
    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 1.0)?;
    // Another example that is more likely to trigger the overflow reported in #1526
    let a = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    let b = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 2.0)?;
    Ok(())
}

fn ggml_matmul_error_test_<T: GgmlType>(a: &[f32], b: &[f32], err_m: f32) -> Result<()> {
    let length = a.len();

    let mut a_quant = vec![T::zeros(); length / T::BLCK_SIZE];
    let mut b_quant = vec![T::VecDotType::zeros(); length / T::VecDotType::BLCK_SIZE];
    T::from_float(a, &mut a_quant);
    T::VecDotType::from_float(b, &mut b_quant);

    let result = T::vec_dot(length, &a_quant, &b_quant);
    let result_unopt = T::vec_dot_unopt(length, &a_quant, &b_quant);

    if (result - result_unopt).abs() / length as f32 > 1e-6 {
        bail!(
            "the opt and unopt vec-dot returned different values, opt: {result} vs unopt: {result_unopt}"
        )
    }

    let mut dst = vec![0.0f32; 1];
    crate::k_quants::matmul((1, length, 1), b, &a_quant, &mut dst)?;
    let result_matmul = dst[0];

    if (result_matmul - result).abs() / length as f32 > 1e-6 {
        bail!(
            "calling matmul vs calling vec-dot directly returned different values, matmul: {result_matmul} vs vec-dot: {result}"
        )
    }

    let reference_result = vec_dot_reference(a, b);

    let verify_result = |result: f32, source: &str| {
        let error = (result - reference_result).abs() / length as f32;
        let ggml_error = ggml_reference_matmul_error(T::DTYPE)? * err_m;
        if !error.is_finite() || error > GGML_MAX_DOT_PRODUCT_ERROR {
            bail!("Dot product with dtype {:?} error {error} exceeds max error {GGML_MAX_DOT_PRODUCT_ERROR}. Source: {source}", T::DTYPE);
        }
        // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
        // => we use a slightly higher error threshold
        const ERROR_LENIENCY: f32 = 0.00001;
        if error - ERROR_LENIENCY > ggml_error {
            bail!(
                "Dot product with dtype {:?} error {error} exceeds ggml reference error {ggml_error}. Source: {source}",
                T::DTYPE,
            );
        }
        Ok(())
    };

    verify_result(result, "vec-dot")?;
    verify_result(result_matmul, "matmul")?;
    Ok(())
}

#[test]
fn quantized_mm() -> Result<()> {
    ggml_matmul_error_test::<f32>()?;
    ggml_matmul_error_test::<half::f16>()?;
    ggml_matmul_error_test::<half::bf16>()?;
    ggml_matmul_error_test::<k_quants::BlockQ4_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ4_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_1>()?;
    Ok(())
}

/// generates random tensors of size `m x k` and `n x k` and calculates their expected matrix multiplication result.
fn get_random_tensors(
    m: usize,
    k: usize,
    n: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(314159265358979);

    let lhs = (0..m * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();
    let rhs = (0..n * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_vec(lhs, (m, k), device)?;
    let rhs = Tensor::from_vec(rhs, (n, k), device)?;

    let mm = lhs.matmul(&rhs.t()?)?;
    Ok((lhs, rhs, mm))
}

#[macro_export]
macro_rules! quantized_matmul {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $fn_name_cpu: ident, $fn_name_cuda: ident, $fn_name_metal: ident, $dtype: expr) => {
        fn $fn_name(device: &Device) -> Result<()> {
            test_matmul(device, (1, 3, 4, 256), $dtype)?;
            Ok(())
        }

        test_device!($fn_name, $fn_name_cpu, $fn_name_cuda, $fn_name_metal);
    };
}

quantized_matmul!(
    quantized_matmul_q4_0_bis,
    quantized_matmul_q4_0_cpu,
    quantized_matmul_q4_0_cuda,
    quantized_matmul_q4_0_metal,
    GgmlDType::Q4_0
);
quantized_matmul!(
    quantized_matmul_q4_1_bis,
    quantized_matmul_q4_1_cpu,
    quantized_matmul_q4_1_cuda,
    quantized_matmul_q4_1_metal,
    GgmlDType::Q4_1
);
quantized_matmul!(
    quantized_matmul_q5_0_bis,
    quantized_matmul_q5_0_cpu,
    quantized_matmul_q5_0_cuda,
    quantized_matmul_q5_0_metal,
    GgmlDType::Q5_0
);
quantized_matmul!(
    quantized_matmul_q5_1_bis,
    quantized_matmul_q5_1_cpu,
    quantized_matmul_q5_1_cuda,
    quantized_matmul_q5_1_metal,
    GgmlDType::Q5_1
);
quantized_matmul!(
    quantized_matmul_q8_0_bis,
    quantized_matmul_q8_0_cpu,
    quantized_matmul_q8_0_cuda,
    quantized_matmul_q8_0_metal,
    GgmlDType::Q8_0
);
quantized_matmul!(
    quantized_matmul_q8_1_bis,
    quantized_matmul_q8_1_cpu,
    quantized_matmul_q8_1_cuda,
    quantized_matmul_q8_1_metal,
    GgmlDType::Q8_1
);
quantized_matmul!(
    quantized_matmul_q2k_bis,
    quantized_matmul_q2k_cpu,
    quantized_matmul_q2k_cuda,
    quantized_matmul_q2k_metal,
    GgmlDType::Q2K
);
quantized_matmul!(
    quantized_matmul_q3k_bis,
    quantized_matmul_q3k_cpu,
    quantized_matmul_q3k_cuda,
    quantized_matmul_q3k_metal,
    GgmlDType::Q3K
);
quantized_matmul!(
    quantized_matmul_q4k_bis,
    quantized_matmul_q4k_cpu,
    quantized_matmul_q4k_cuda,
    quantized_matmul_q4k_metal,
    GgmlDType::Q4K
);
quantized_matmul!(
    quantized_matmul_q5k_bis,
    quantized_matmul_q5k_cpu,
    quantized_matmul_q5k_cuda,
    quantized_matmul_q5k_metal,
    GgmlDType::Q5K
);
quantized_matmul!(
    quantized_matmul_q6k_bis,
    quantized_matmul_q6k_cpu,
    quantized_matmul_q6k_cuda,
    quantized_matmul_q6k_metal,
    GgmlDType::Q6K
);
// Not implemented on metal
quantized_matmul!(
    quantized_matmul_q8k_bis,
    quantized_matmul_q8k_cpu,
    quantized_matmul_q8k_cuda,
    quantized_matmul_q8k_metal,
    GgmlDType::Q8K
);

#[test]
fn quantized_matmul_q2k() -> Result<()> {
    use k_quants::BlockQ2K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [0.916, 0.422, 0.215, 1.668]);

    ggml_matmul_error_test::<BlockQ2K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q3k() -> Result<()> {
    use k_quants::BlockQ3K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q3K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.029, 1.418, -0.314, 1.495]);

    ggml_matmul_error_test::<BlockQ3K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q4k() -> Result<()> {
    use k_quants::BlockQ4K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q4K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.125, 1.435, -0.201, 1.589]);

    ggml_matmul_error_test::<BlockQ4K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q5k() -> Result<()> {
    use k_quants::BlockQ5K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q5K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.192, 1.491, -0.18, 1.743]);

    //Expected: 0.000740408897
    ggml_matmul_error_test::<BlockQ5K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q6k() -> Result<()> {
    use k_quants::BlockQ6K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q6K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.324, 1.49, -0.164, 1.741]);

    ggml_matmul_error_test::<BlockQ6K>()?;
    Ok(())
}

#[test]
fn quantized_matmul_q8k() -> Result<()> {
    use k_quants::BlockQ8K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q8K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.266, 1.504, -0.204, 1.7]);

    ggml_matmul_error_test::<BlockQ8K>()?;
    Ok(())
}

fn from_data_dequant_matches_canonical_when_caller_passes_cow_owned(device: &Device) -> Result<()> {
    let cpu = Device::Cpu;
    let n = 1024usize;
    let src_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let src = Tensor::from_vec(src_data, (n,), &cpu)?;
    let qt_canonical = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let canonical_dequant = qt_canonical.dequantize(&cpu)?.to_vec1::<f32>()?;

    let bytes_owned: Vec<u8> = qt_canonical.data()?.to_vec();
    let storage = quantized::QStorage::from_data(Cow::Owned(bytes_owned), device, GgmlDType::Q4_0)?;
    let qt_via_from_data = quantized::QTensor::new(storage, (n,))?;
    let observed_dequant = qt_via_from_data.dequantize(device)?.to_vec1::<f32>()?;

    let max_diff = canonical_dequant
        .iter()
        .zip(observed_dequant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "QStorage::from_data dequant mismatch on {device:?} (max |Δ| = {max_diff})"
    );
    Ok(())
}

test_device!(
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned,
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned_cpu,
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned_cuda,
    from_data_dequant_matches_canonical_when_caller_passes_cow_owned_metal
);

/// `QMetalStorage::from_buffer` is how a zero-copy (e.g. mmap-backed) loader
/// wires a tensor's storage to a *view* -- possibly at a nonzero offset --
/// into a buffer shared with other tensors, instead of a buffer this storage
/// owns outright. This must produce identical dequantize and matmul output
/// to the normal owned-buffer path, at both a zero and a nonzero offset --
/// an offset-arithmetic bug here would silently corrupt weights rather than
/// crash.
#[cfg(feature = "metal")]
#[test]
fn qmetalstorage_from_buffer_view_matches_owned_buffer() -> Result<()> {
    use candle_core::quantized::{metal::QMetalStorage, QStorage, QTensor};

    let device = Device::new_metal(0)?;
    let metal_device = match &device {
        Device::Metal(d) => d.clone(),
        _ => unreachable!(),
    };
    let dtype = GgmlDType::Q4K;
    // A (n_out, k) weight, matching quantized_matmul's own convention
    // (quantize a (n,k)-shaped tensor, matmul against a (m,k) activation).
    let k = dtype.block_size();
    let n_out = 4usize;
    let n = n_out * k;

    let src_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let src = Tensor::from_vec(src_data, (n_out, k), &device)?;
    let canonical = QTensor::quantize(&src, dtype)?;
    let canonical_dequant = canonical.dequantize(&device)?.to_vec2::<f32>()?;
    let raw_bytes: Vec<u8> = canonical.data()?.to_vec();

    // QTensor isn't Clone and QMatMul::from_qtensor takes it by value, so
    // compute the canonical matmul reference once, up front -- it doesn't
    // depend on front_padding, only `view` needs rebuilding per iteration.
    let activation = Tensor::ones((1, k), DType::F32, &device)?;
    let canonical_matmul = quantized::QMatMul::from_qtensor(canonical)?;
    let canonical_out = canonical_matmul.forward(&activation)?.to_vec2::<f32>()?;

    for front_padding in [0usize, 4096] {
        let mut combined = vec![0xABu8; front_padding];
        combined.extend_from_slice(&raw_bytes);
        combined.extend_from_slice(&[0xCDu8; 4096]); // trailing padding, must never be read

        let shared_buffer = metal_device.new_buffer_with_data(&combined)?;
        let storage = QMetalStorage::from_buffer(
            shared_buffer,
            front_padding,
            raw_bytes.len(),
            metal_device.clone(),
            dtype,
        );
        let view = QTensor::new(QStorage::Metal(storage), (n_out, k))?;

        let view_dequant = view.dequantize(&device)?.to_vec2::<f32>()?;
        let max_diff = canonical_dequant
            .iter()
            .flatten()
            .zip(view_dequant.iter().flatten())
            .map(|(a, b): (&f32, &f32)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(
            max_diff, 0.0,
            "from_buffer view (offset={front_padding}) dequant must be bit-identical to the owned-buffer path"
        );

        // Exercise the offset-aware matmul path (call_quantized_matmul_mv_t),
        // not just dequantize -- these are two independent call sites.
        let view_matmul = quantized::QMatMul::from_qtensor(view)?;
        let view_out = view_matmul.forward(&activation)?.to_vec2::<f32>()?;
        let mm_max_diff = canonical_out
            .iter()
            .flatten()
            .zip(view_out.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(
            mm_max_diff, 0.0,
            "from_buffer view (offset={front_padding}) matmul must be bit-identical to the owned-buffer path"
        );
    }

    Ok(())
}
