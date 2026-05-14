# RetrievalAttention research scratchpad

Python prototypes for the [[Open Sparse Stack PRD]] spec 034 work. Lives on `feature/retrieval-attention` only — does not merge to `main` / `vllm-swift-stable`. Swift backend is the actual deliverable; this directory holds:

- Pure-math validations (V3-trig basis, JL projection sanity).
- Pure-logic unit tests for the dedupe + block-centroid + sentinel design.
- Selector recall@k harness skeleton (Python prototype before Swift port).
- Calibration scripts that export `.safetensors` for Swift to load (per Decision 2).

See `/Users/tom/Documents/obsidian/Self Study/Open Sparse Stack — Experiment Log.md` for the live findings log.

## Run

```bash
cd research/retrieval_attention
python3 -m pytest -v
```

All tests must be runnable with just `mlx`, `numpy`, `pytest`. No model weights required for the v0 tests.
