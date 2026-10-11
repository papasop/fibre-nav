# GPT-2 B4: minimal causal read–compute–write development test

Run GPT2_B4_CAUSAL_XOR_A100_V1.py in a separate Colab runtime. It is self-contained.
Default: Google Drive backup enabled, NVIDIA A100, float32, no session time budget.
Upload is unnecessary for a first run. For continuation, set RESUME_ZIP to a result ZIP path.
The launcher searches the B4 backup directory by verified actual completed-phase/step progress.
A resume never silently falls back to fresh training when a selected archive is invalid.

## Parent and evidence scope
Uses Library-archived moving_fibre_intelligence_l5_v2_2_4 source; old L5 corresponds to current B3.
This is a separately declared A100 development run; CPU result ZIPs cannot resume into it.
Original writer preserved verbatim as writer_core_original.py with parent_manifest.json hashes.
GitHub's current HEAD has not been fetched or verified; this is not a claim about its current state.
The archived v2.2.4 single-seed L5 result passed; unified v2.4.3 passed 2/3 seeds overall.
Historical result ZIPs contain no recoverable weights. This run rebuilds a shared state from zero B.
Seed82931 remains development; previously inspected expression prompts are development checks.

## What is tested
Four cells: input A / input B / unselected sentinel / output.
1. Prepare [0,0,1,1] using the archived B3 writer (600 steps).
2. Read inputs through actual GPT-2 forward passes, externally compute XOR, write output (480 steps).
3. Restore preparation state and substitute input A in parameters: [1,0,1,1] (480 steps).
4. Read again, compute XOR, write output (480 steps), reload and check both endpoints.
Controller receives a reader callback, no desired task answers. Existing contents for unselected
cells are decoded, not taken from oracle task labels. Ground-truth labels occur in validation
and in the explicitly prescribed preparation/intervention stages.
Controls: content substitution, blocked reads, zero-B reset, no-write counterfactual,
endpoint reload, retention of unselected cells. The blocked/no-write controls are computed
counterfactuals; they do not train separate control models.
Output is already correct in the substituted branch; the normal branch tests a required output change.
This is a minimal causal integration test of two input states, not exhaustive memory capacity testing.

## Acceptance and limitations
All stages require exact and previously inspected expression accuracy=1; minimum margins>=0.05;
mean selected-response drift<=0.002; mean anchor KL<=0.005. No universal preservation claim.
Selection uses archived training-only checkpoint criteria. Full training histories are retained.
Initial/replacement failure stops with explicit NOT_READY instead of claiming causal success.
A passing run reports B4_MINIMAL_DEVELOPMENT_PASS, not independent B4 confirmation or B5.
XOR execution and numerical updates are external. No model-selected instruction execution,
internal parameter updating, lifelong learning, new-seed confirmation or fresh generalization claim.
Protected-response definitions are the parent averaged-anchor definitions (12 anchors, 6 coordinates).

## Persistence
Every five training steps: actual B, last selected B, start B, full loop state, RNG and histories
are serialized into a CRC-checked, per-member SHA256 result ZIP. Drive copies are read back and
SHA256 verified before publishing. Only this new B4 directory retains the latest three backups;
older version directories are untouched. Interrupted steps after the last checkpoint may be replayed.
At stage completion the selected endpoint, evaluation and history are saved.
Resume requires matching protocol, writer, model revision, LoRA-A, versions, device and chart.
A Colab interruption may prevent the final export, but verified Drive checkpoint ZIPs remain.
Snapshots store only mutable B; reconstruct frozen model/A from recorded revision and seed.

## Validation before delivery
7 standard-library tests: XOR truth table and access calls, blocked-read isolation,
parameter-content contrast, invalid read rejection, all endpoint gates, archive corruption rejection,
and original parent-source identity.
A tiny PyTorch constrained endpoint executes the actual patched writer loop; interrupted/saved/
restored continuation matches uninterrupted parameters, history and selected checkpoint exactly.
The sequence driver also passes a synthetic-actuator integration test including all causal controls,
archive restoration and skipping completed phases. This checks recovery mechanics, not GPT-2 B4. Real GPT-2 weights were not loaded in delivery QA.

## Files
writer_core_original.py: unmodified parent.
writer_core.py: parent plus explicit revision, disabled KV cache and resumable loop hooks.
worker_b4.py / protocol.py: B4 sequence and causal controller.
archive_io.py: safe verified restore.
test_b4.py / test_resume.py: executable checks.
Result: gpt2_b4_a100_results.zip with SUMMARY.json, causal_traces.json, records, checkpoints and source.

## A100 configuration
Select A100 in Colab Runtime settings before executing. Both launcher and worker refuse CPU or
other GPUs. Colab CUDA PyTorch is retained; pinned Hugging Face packages go into a dedicated
directory. The child process sees system site-packages (CUDA torch) plus that directory.
Forward/backward are float32; TF32 and autocast are disabled; attention is eager.
CUBLAS_WORKSPACE_CONFIG is set before the independent worker imports torch.
Runtime versions, GPU name, CUDA version and precision are recorded and must match on resume.
Acceptance gates, four-cell interface, seed and phase budgets match the CPU development design.
This package has no arbitrary session cap; Colab can still interrupt or reclaim the runtime.
Only the new GPT2_B4_CAUSAL_XOR_A100_V1_RECOVERY directory prunes its own old backups.
Actual A100 execution has not been tested in the delivery environment; tests use CPU synthetic
endpoints and mocked hardware detection for workflow checks. No speed or passing-result guarantee.
