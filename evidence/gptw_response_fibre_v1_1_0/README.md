# GPTW response-fibre evidence snapshot v1.1.0

Frozen, repository-ready evidence for the GPT-2 + native-LoRA branch of *Moving Response Fibres in Neural Networks*.

## Confirmed and supported results

| Stage | Frozen decision | Result |
|---|---|---:|
| `v1_r1` | Formal initial audit | 0/8 because the FD gate divided by near-zero `||Jv||`; effect gates were 7/8 |
| `v1_r2a_precision` | Same-seed precision repair | 7/8, required 6/8 |
| `v1_r3_newseed` | Prospective adaptive-value confirmation | 8/8, required 6/8 |
| `v2_current_vs_fixed` | Prospective moving-fibre realizability audit | 6/6 instances and 18/18 nodes |
| `v3_natural_text` | Prospective disjoint natural-English current-vs-fixed support | 8/8 instances and 32/32 nodes |

The r3 result confirms selected response-kernel tangent adaptive value in a declared GPT-2 rank-2 LoRA-B domain. The v2 result supports a finite local current-fibre realizability advantage over source-fixed replay along constructed response-retracted paths. Its 2-versus-1 exponent split is analytically forced by the arm definitions and retained as a numerical-correctness check; the substantive v2 evidence is the finite-radius cost ratio 8.60-25.70, current/source-fixed principal angle 0.0161-0.0713 rad, active-J residual amplification about 1.4e4-1.4e5, numerical-precision path-response error, and 6/6 seeds with 18/18 nodes passing. V3 prospectively confirms that the separation is not confined to the original codeword prompts when the development continuations and response-defining continuations are disjoint frozen natural-English prompts: all eight new seeds and all 32 interior nodes pass, with finest-radius cost ratio 8.87-35.56, active-J residual amplification 1.41e7-1.12e8 and maximum path-response error about 1.08e-15. This does not change the full-model or semantic boundaries stated in `CLAIM_BOUNDARY.md`.

## Layout

Each experiment contains the exact `run.py` and unmodified result JSON files. V3 also includes its Colab launcher and the frozen prompt sets inside `results/protocol.json`. `development/` retains the r2 dtype-failure log. `CLAIM_BOUNDARY.md` states permitted and prohibited interpretations; `DEVELOPMENT_RECORD.md` records the repair history. `paper/` contains the manuscript snapshot used when assembling this release.

## Reproduction

Use Python 3.10-3.13, PyTorch with CUDA, and:

```bash
python -m pip install -r requirements.txt
python experiments/v1_r3_newseed/run.py --output v1_r3_reproduction
python experiments/v2_current_vs_fixed/run.py --output v2_reproduction
python experiments/v3_natural_text/run.py --output v3_reproduction
```

An A100 is recommended. Each program disables TF32 for audited operations, converts the model to float64, downloads GPT-2 from `openai-community/gpt2` or `gpt2`, uses no external dataset, and enforces a 55-minute hard limit.

Before citing numerical values, verify the snapshot:

```bash
sha256sum -c SHA256SUMS
```

Model weights are not redistributed in this archive. Run `python verify_snapshot.py` for semantic cohort checks in addition to the byte-level checksum verification.
