# F-81 Phase D V2 — Real RA × TurboQuant+ Composition Design

**Status**: Design.
**Date**: 2026-05-14.
**Blocked by**: nothing — F-80 cliff fix unblocks long-context decode.
**Estimated cost**: 2-4 days kernel-and-API work + 1 day testing.

## What the V1 scaffold gave us

`RetrievalAttentionKVCache.inner` widened to `BaseKVCache`. New init takes
`valueBits` and constructs a `TurboQuantizedKVCache` in rawKeyMode. RA's
`update()` routes to `tqInner.updateAndDequant` when inner is TQ.

**What it does NOT do**: actually compress V. TQ's `updateAndDequant` is
the A path — raw FP16 round-trip, no codec invocation. Result: `RA-vs-RA+TQ`
cosine = 1.0 (bit-identical). No memory savings.

The test (`raTurboCompose_32K_14B1M`) asserts this round-trip stays at 1.0
as a scaffold-state regression catch.

## What Phase D V2 must deliver

V compressed at 4 bits, K stays raw FP16, RA's selector + gather operate
on the compressed-V cache without round-tripping through full dequant per
decode step.

### Target numbers (PRD Phase D)
- `cosine(dense, RA+TQ4) >= 0.99` (preserve RA quality with V compression)
- V memory: 1/4 of raw (V at 4bit vs FP16)
- Decode latency: ≤ RA-alone + ~2-3ms (gather + per-position dequant cost)
- Validated at 32K minimum, ideally 65K-128K.

## API surface (proposed)

Three new methods on `TurboQuantizedKVCache`:

```swift
/// Trigger compression of the raw cache. No-op if already compressed.
/// Must be called once at the prefill → decode boundary.
public func prepareForDecode()

/// Encode a new decode-step token into the compressed value store.
/// Keys: appended to rawKeys (rawKeyMode). Values: MSE-encoded.
/// L must be 1.
public func appendDecodeToken(keys: MLXArray, values: MLXArray)

/// Gather and dequantize values at the given positions.
/// - positions: [N] int32 indices into the seq axis
/// - returns: [B, H, N, D] dequantized values (FP16)
public func gatherAndDequantValues(positions: MLXArray) -> MLXArray
```

The first two are lifecycle hooks. The third is the load-bearing op.

### `gatherAndDequantValues` implementation

```swift
public func gatherAndDequantValues(positions: MLXArray) -> MLXArray {
    guard isCompressed, let valueMSECodec, let valPackedMSE, let valNorms else {
        fatalError("call prepareForDecode() before gatherAndDequantValues")
    }
    // valPackedMSE: [B, H, seq, vpw]
    // valNorms:     [B, H, seq]
    let packedAt = valPackedMSE.take(positions, axis: 2)
    let normsAt = valNorms.take(positions, axis: 2)
    let state = MSECodecState(packedIndices: packedAt, norms: normsAt)
    return valueMSECodec.decode(state)
}
```

`MSECodec.decode` at `TurboQuantKVCache.swift:728`:
- Unpacks low-bit indices → `[N, dim]` int8
- Codebook lookup → `[N, dim]` FP32 approx vectors
- Inverse rotates via Π^T → original V basis
- Rescales by per-position norms
- Returns FP16/FP32 `[B, H, N, D]`

Cost per gather (N positions, D=128, bits=4):
- unpack: ~50 µs for N=4K
- codebook lookup: ~30 µs
- matmul Π^T: ~80 µs
- rescale: ~10 µs
- Total: ~170 µs per layer per gather call. At 40 sparse layers × 1 step =
  ~7 ms additional decode latency. Acceptable given F-79 ship is 65 ms.

## RA integration

`RetrievalAttentionKVCache.update`:

```swift
public override func update(...) -> (MLXArray, MLXArray) {
    if let tq = inner as? TurboQuantizedKVCache {
        if keys.dim(2) > 1 {
            // Prefill: raw store, no compression yet
            return tq.update(keys: keys, values: values)
        } else {
            // Decode step: ensure compressed, append, return synthetic
            // (raw_K, packed_V) handle — but we can't return a packed V
            // through the same interface. Need to return (rawKeys[..<off],
            // null V placeholder) and let the dispatcher route to a new
            // ra-tq gather path.
            tq.prepareForDecode()  // idempotent
            tq.appendDecodeToken(keys: keys, values: values)
            ...
        }
    } else {
        return inner.update(keys: keys, values: values)
    }
    // ... selector index update on raw K (unchanged)
}
```

Then in `AttentionUtils.attentionWithCacheUpdate` case `.retrievalSparse`,
detect `raCache.inner is TurboQuantizedKVCache` and route through a new
`raCache.gatherAndAttendTQ(...)` that:
1. Picks gather positions via selector (unchanged)
2. Gathers raw K from `tq.rawKeys[..., positions, :]`
3. Gathers + dequants V via `tq.gatherAndDequantValues(positions)`
4. Runs `MLXFast.scaledDotProductAttention(q, gatheredK, gatheredV, ...)`

## Open questions

- **Q1**: Does TQ's compression need to happen ONLY at prefill end, or can it
  be lazy/incremental? Currently `compressRawCache` is a batch operation.
  For incremental decode-time encoding, `encodeNewToken` already exists at
  TurboQuantKVCache.swift:1577. We can reuse it.
- **Q2**: At decode, RA returns `(cachedKeys, cachedValues)` from `update()`.
  When inner is TQ-compose, what's `cachedValues`? Today it's a full
  dequantized V. We could return raw K and a "lazy V handle" but that
  changes signature. Cleanest: return `(rawKeys[..<offset], dummy_V_view)`
  and have the dispatcher know to call `gatherAndDequantValues` instead of
  reading `cachedValues` directly.
- **Q3**: Sliding-window caches (rotating K/V) — TQ supports `rotatingMaxSize`.
  RA's window math vs TQ's wrap-around: need careful unification. For v1
  compose, gate on unbounded mode only.

## Implementation milestones

1. **M1 (1 day)**: Add `prepareForDecode`, `appendDecodeToken`,
   `gatherAndDequantValues` to TurboQuantizedKVCache. Unit-test the
   round-trip: encode random V → gather → dequant, compare to original
   with quant-noise tolerance.

2. **M2 (1 day)**: New `RetrievalAttentionKVCache.gatherAndAttendTQ` path.
   Reuses existing selector + retrievalAttentionGatherAndAttend, swap V
   source from `cachedValues` to `tq.gatherAndDequantValues(positions)`.

3. **M3 (1 day)**: Dispatcher routing. `AttentionUtils.attentionWithCacheUpdate`
   detects TQ inner on `.retrievalSparse` and routes accordingly. Disable
   `bypassSelectorDecode` for this path (no fallback to dense without
   real V dequant).

4. **M4 (1 day)**: Validation. Update `raTurboCompose_32K_14B1M`:
   - cosine(dense, RA+TQ4) >= 0.99 → real compression confirmed
   - cosine(RA, RA+TQ4) >= 0.97 (NOT 1.0 — must differ from quant noise)
   - decode latency report
   - Memory profile (peak active during decode)
   - Extend to 49K, 65K with cliff fix.

## What we'll NOT do in V2

- Full K compression (rawKeyMode stays — K precision matters for selector)
- B-path `compressedAttention` integration (RA does its own SDPA)
- Sinks-using model support
- Multi-batch (B>1)
- Codec swaps mid-decode

These are V3 territory.

## References

- F-79 ship doc: `Open Sparse Stack — Near-Parity with Dense.md`
- F-80 cliff diagnosis: `F80_CLIFF_DIAGNOSIS.md`
- F-81 V1 scaffold commit: `feature/retrieval-attention@22ab42d`
- TQ codec: `TurboQuantKVCache.swift:610` (MSECodec class)
- RA cache: `RetrievalAttentionKVCache.swift`
- Dispatcher: `AttentionUtils.swift:229` (case `.retrievalSparse`)
