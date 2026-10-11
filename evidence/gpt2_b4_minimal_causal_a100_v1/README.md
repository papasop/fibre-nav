# GPT-2 B4 minimal causal development — A100 V1

**Recorded decision: B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS.**
Evidence date: 11 October2026. Single development seed82931; fixed XOR input01.
This stronger-control experiment supplements the earlier V1 archive; it does
not replace historical results or establish complete B4/B5 confirmation.

## Recorded causal sequence
| Stage | Actual four-cell bits | Required effect |
|---|---|---|
| Prepare | [0,1,1,0] | Output deliberately incorrect |
| Normal writeback | [0,1,1,1] | Actual reads01; external XOR1; output flips |
| Wrong-read branch | [0,1,1,0] | Substitute controller input11; actual output flips to0 |
| Stored-input substitution | [1,1,1,1] | Change stored A only; output intentionally stale |
| Substituted writeback | [1,1,1,0] | Actual reads11; external XOR0; output flips |

Resetting actual LoRA-B gives read bits00 and computed XOR0; restoring the
saved normal endpoint gives read bits01 and computed XOR1. Nineteen recorded
case checks pass, including five endpoint reload checks and no-write controls
using the measured pre-write outputs.

Five endpoints have exact and original development-expression accuracy1.
Minimum original-development margin is0.347885 (gate0.05); maximum recorded
averaged protected-response drift is1.52587890625e-5 (budget0.002); maximum mean
anchor KL is0.004150287714 (budget0.005). A reused V2 expression family also
passes auxiliary diagnostics; it is NOT fresh held-out confirmation and is
not an acceptance gate. Response and KL constraints are averaged quantities,
not uniform preservation of all prompts or capabilities.

XOR and operation sequencing are external; GPT-2 supplies parameter-memory
reads and the constrained writer updates frozen-chart LoRA-B coordinates.
Acceptance supports single-input, single-development-seed causal parameter
read-compute-write evidence. Four-input coverage, new-seed confirmation,
independent replication, geometry-superiority baselines and B5 remain open.
Full XOR V2 retains its own results-pending status.

## Files and verification
- `gpt2_b4_minimal_causal_a100_results.zip`: untouched original result archive,
  including five saved endpoint tensors and frozen execution source.
- `source/`: byte-identical execution source copied from the result archive;
  its original README is historical protocol documentation.
- `recorded/`: byte-identical JSON records from that archive.
- `GPT2_B4_MINIMAL_CAUSAL_A100_V1.py`: regenerated one-cell Colab launcher
  embedding the exact archived execution source. The wrapper is newly generated;
  it is not claimed byte-identical to the original launcher's outer wrapper.
- `GPT2_B4_MINIMAL_CAUSAL_A100_V1.zip`: exact archived source packaged for use.
- `PROVENANCE.json`: repository base, original archive hash, scope and runtime.
- `verify_archive.py`: stdlib-only checks; no model loading or training.

Run `python3 verify_archive.py` in this directory. In Colab, select A100 and
paste the full launcher into one cell. Fresh reconstruction uses2520 training
steps; default Drive backup, verified atomic copies and compatible continuation
are retained. Old V1, full V2 and quick-smoke result ZIPs are incompatible.
Do not launch while another B4 worker is running. No wall-clock limit.
