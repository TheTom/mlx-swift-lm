//
//  SSM.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/10/01.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ssm.py

import Foundation
import MLX
import MLXNN

public func computeDt(_ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float))
    -> MLXArray
{
    let dt = softplus(dt + dtBias)
    return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)
}

private func makeSSMKernel() -> MLXFast.MLXFastKernel? {
    // Batched-decode correctness fix (2026-05-16):
    //
    // The original Python kernel computes `g_idx = n / G` where `n` is a flat
    // index over `B * H`. For B=1 (the only case Python's mlx_lm ever exercises
    // this fast path with — Python prefill goes through ssm_attn instead),
    // `n == h_idx` and `n / G` happens to equal `h_idx / G == g_idx`. For
    // batched decode (B > 1), `n = b_idx * H + h_idx`, so `n / G` includes a
    // bogus batch term and overshoots the valid group range. With `numGroups
    // > 1` (e.g. Nemotron Cascade 2 has `n_groups=8`) every slot beyond slot 0
    // indexes out-of-range C / B and produces garbage.
    //
    // Three fixes vs the Python kernel (see also the equivalent fix needed
    // upstream in mlx-lm if Mamba2 batched-decode lands there):
    //   1. Compute `b_idx = n / H` explicitly.
    //   2. Compute `g_idx = h_idx / G` (group within batch), NOT `n / G`.
    //   3. Offset C and B by the per-batch stride `b_idx * num_groups * Ds`
    //      where `num_groups = H / G`. Without this every batch slot reads
    //      C[0] / B[0] from slot 0.
    let source = """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto b_idx = n / H;
            auto g_idx = h_idx / G;
            constexpr int n_per_t = Ds / 32;
            constexpr int num_groups = H / G;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            // C and B have shape [batch, T=1, group, state_dim]; offset by
            // batch (b_idx * num_groups * Ds) and then by group (g_idx * Ds).
            auto C_ = C + (b_idx * num_groups + g_idx) * Ds;
            auto B_ = B + (b_idx * num_groups + g_idx) * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            auto dt_ = static_cast<float>(dt[n]);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto idx = d_idx * Ds + s_idx;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                auto state = dA * static_cast<float>(i_state[idx]) + dB_by_x;
                o_state[idx] = static_cast<U>(state);
                acc += state * static_cast<float>(C_[s_idx]);
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[d_idx] = static_cast<T>(acc + x_ * static_cast<float>(D[h_idx]));
            }
        """

    return MLXFast.metalKernel(
        name: "ssm_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: source
    )
}

private final class SSMKernelManager: Sendable {
    static let shared = SSMKernelManager()

    let ssmKernel: MLXFast.MLXFastKernel?

    private init() {
        ssmKernel = makeSSMKernel()
    }
}

func ssmUpdateKernel(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    let (n, _, h, d) = hiddenStates.shape4
    let inputType = hiddenStates.dtype
    let stateType = state.dtype
    let (hb, ds) = (B.dim(-2), B.dim(-1))

    // Match Python `mlx_lm/models/ssm.py`'s `compute_dt`: promote dt to fp32
    // BEFORE softplus + clip, then keep fp32 through the kernel dispatch.
    // The Swift `computeDt` does not include the upcast (Python: `dt =
    // dt.astype(mx.float32)` before `softplus`), so without this the
    // kernel receives bf16 dt and the recurrence loses precision — the
    // visible symptom is cross-run non-determinism at decode (the
    // recurrence accumulates small bf16 rounding error differently each
    // run depending on uninitialised pool buffer contents the kernel
    // happens to read in adjacent grid threads).
    let dt = computeDt(dt.asType(.float32), dtBias, timeStepLimit)

    guard let kernel = SSMKernelManager.shared.ssmKernel else {
        fatalError("SSM kernel not available")
    }

    // Two template types (matches Python `mlx_lm/models/ssm.py`):
    //   T = inputType  — used for x / out / D / C / B / dt buffer dtype
    //   U = stateType  — used for state_in / state_out buffer dtype
    //
    // The Swift version pre-2026-05-16 used a single `T` template and
    // `outputDTypes: [inputType, inputType]`. That declared `state_in` as
    // `T*` (= bf16*) while the kernel actually received fp32-dtype state
    // from the per-request `SSMStateCache` (Python keeps state fp32 across
    // the recurrence to dodge precision loss). The kernel then
    // reinterpreted fp32 bytes as bf16 — producing garbage / non-
    // deterministic output across runs (cross-process memory layout
    // varied which uninit reads landed in valid float range).
    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, dt, state],
        template: [
            ("T", inputType),
            ("U", stateType),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h / hb),
        ],
        grid: (32, d, h * n),
        threadGroup: (32, 8, 1),
        outputShapes: [[n, 1, h, d], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

public func segsum(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let l = x.dim(-1)
    var x = x

    if let mask = mask {
        let mask = MLX.expandedDimensions(mask, axis: 1)
        x = x * mask
    }

    x = MLX.repeated(x[.ellipsis, .newAxis], count: l, axis: -1)
    x = MLX.tril(x, k: -1)
    var xSegsum = MLX.cumsum(x, axis: -2)

    if let mask = mask {
        // Match xSegsum dtype to avoid fp32 promotion.
        // A/B tested: bf16 segsum produces identical output to fp32 at 128-1024 context.
        xSegsum = which(
            mask[.ellipsis, .newAxis, 0...] * mask[.ellipsis, .newAxis],
            xSegsum,
            MLXArray(Float(-Float.infinity), dtype: xSegsum.dtype)
        )
    }

    return xSegsum
}

public func ssmAttn(
    x: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let (_, _, g, d) = B.shape4

    // Match Python `mlx_lm/models/ssm.py` `compute_dt`: promote dt to fp32
    // before softplus + clip so the SSM recurrence holds full fp32
    // precision (the matmul `dtxdecay @ B` then produces fp32 state, which
    // the per-request `SSMStateCache` carries into the next forward).
    let dt = computeDt(dt.asType(.float32), dtBias, timeStepLimit)
    let repeats = h / g
    let A = -MLX.exp(ALog)
    var B = MLX.transposed(B, axes: [0, 2, 3, 1])

    // A * s + B * C
    var CB = MLX.swappedAxes(C, 1, 2).matmul(B)
    CB = MLX.repeated(CB, count: repeats, axis: 1)

    let dtA = dt * A.reshaped(1, 1, -1)
    var decay = MLX.exp(segsum(dtA.swappedAxes(1, 2), mask: mask))

    let surrogateAttentionMatrix = MLX.tril(CB * decay, k: 0)

    let dtx = dt.reshaped(b, l, h, 1) * x
    var y = surrogateAttentionMatrix.matmul(dtx.swappedAxes(1, 2))
    y = MLX.swappedAxes(y, 1, 2)

    decay = decay[0..., 0..., (-1)..., 0...].transposed(0, 3, 1, 2)
    B = MLX.repeated(B, count: h / g, axis: 1).swappedAxes(2, 3)
    var dtxdecay = dtx * decay
    dtxdecay = dtxdecay.swappedAxes(1, 2).swappedAxes(2, 3)

    var nextState = dtxdecay.matmul(B)

    if var state = state {
        let expDtACumsum = MLX.exp(MLX.cumsum(dtA, axis: -2))
        nextState = nextState + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * state
        state = state.reshaped(b, 1, g, repeats, dh, d)
        let C = C.reshaped(b, l, g, 1, d, 1)
        let yPrev = (state.matmul(C)).squeezed(axis: -1).flattened(start: 2, end: 3)
        y = y + expDtACumsum[.ellipsis, .newAxis] * yPrev
    }

    y = y + x * D.reshaped(1, 1, h, 1)
    return (y, nextState)
}

// Diagnostic: set VSM_FORCE_SSM_ATTN=1 to force the matmul-based ssmAttn
// path even at seqLen=1. Used to isolate whether the SSM kernel is the
// source of B>1 numerical drift on Nemotron.
private let _forceSSMAttn: Bool =
    ProcessInfo.processInfo.environment["VSM_FORCE_SSM_ATTN"] == "1"

public func ssmUpdate(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let seqLen = hiddenStates.dim(1)

    if seqLen == 1,
        let state = state,
        !_forceSSMAttn,
        SSMKernelManager.shared.ssmKernel != nil
    {
        return ssmUpdateKernel(
            hiddenStates: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit
        )
    } else {
        return ssmAttn(
            x: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit,
            mask: mask
        )
    }
}
