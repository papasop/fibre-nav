# GPT-2 B4 full XOR A100 V2

One-click Colab A100 protocol. Use the complete `GPT2_B4_FULL_XOR_A100_V2.py` in one Colab cell. Drive backups are enabled by default. Select A100 first; existing CUDA PyTorch is retained. Installation logs and process heartbeats are visible, with bounded dependency installation. Training has no session-time cap; Colab can still reclaim the runtime.

This is a new frozen protocol, not continuation of B4 V1. No old ZIP is needed. It reuses the archived B3 writer/selection mechanism and starts from zero LoRA-B for every seed and every independent truth-table preparation. Base GPT-2 and LoRA-A stay frozen. Model revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

Fixed seeds: development 82931; new seeds 82961, 82991. All seeds and four input cases run even if a previous case fails scientific acceptance. Hardware/dependency/identity errors stop the process. Recorded failures remain in the cohort denominator.

## Five phases per input case

1. PREPARE: set inputs A/B, sentinel=1 and output to the opposite XOR answer (600 maximum steps).
2. NORMAL WRITEBACK: read actual model inputs, external deterministic XOR, write correct output (480 steps).
3. BLOCKED WRITEBACK: reload the normal endpoint, replace read channel A with its complement, actually write the resulting incorrect answer (480 steps). This is an explicit wrong-read counterfactual, not just a trace comparison.
4. INPUT SUBSTITUTION: reload normal endpoint, flip stored A while preserving B, sentinel and the stale output (480 steps).
5. SUBSTITUTED WRITEBACK: read actual new inputs and write new XOR output (480 steps).

Normal, blocked and substituted writeback must actually flip output. Unselected cells must retain their decoded values. All saved endpoints are reloaded and evaluated with actual forwards. Zero-B reset is measured for each case; aggregate reset XOR accuracy is compared with the four intact correct answers, not a Boolean comparison with either of two opposite outputs. No-write counterfactuals are explicit saved pre-write states, with no optimizer applied.

Four XOR cases are tested from separately prepared shared parameter states. They do not constitute a continuous arbitrary-program execution test. The external controller computes XOR; GPT-2 stores and retrieves bits. This is B4 interface integration, not B5 model-selected execution or an internal neural XOR computation claim.

## Fixed gates and new expression family

Per endpoint: exact and original development-expression accuracy=1; target-relative min margin>=0.05; six selected response logits averaged over 12 anchors drift<=0.002; anchor-average KL<=0.005. These protect declared averaged responses, not all outputs or all individual anchors.

Four new prefixes (16 prompts) are frozen in `protocol.py`. They are not passed into writing, gradients or checkpoint selection. Each seed's evaluation is performed only after all its training phases finish. Fresh evaluation applies to these newly specified expressions within this frozen protocol; it is not external preregistration or third-party replication. After inspecting results, do not tune on them and continue calling them untouched holdouts.

A cohort PASS requires every case and endpoint gate, the new-expression gates and reset task contrast on all three seeds. Failures are reported, not skipped. There is no geometry-superiority claim; matched-gradient/random-update controls are outside this protocol.

## Save and resume

One result ZIP: `gpt2_b4_full_xor_a100_results.zip`, containing sources, identities, records, final endpoint tensors, in-progress writer state, causal traces, evaluation and hashes.

Every five training steps and at each phase/case/seed boundary: local ZIP + verified Drive copy. Only the three latest verified V2 backups in `MyDrive/GPT2_B4_FULL_XOR_A100_V2_RECOVERY` are retained. Other versions are not changed. Automatic resume selects the greatest actual progress among valid V2 backups. An explicit invalid/incompatible archive raises an error; no silent fresh fallback.

Interrupted optimization restores the real tensor, writer selection state and RNG history. Completed phases/cases/seeds are skipped. Resume requires matching protocol, engine, model revision, fixed LoRA-A, software and device identity. Set `RESUME_ZIP` to a compatible V2 result path only; never an old V1 result or launcher ZIP.

## Local checks

`python test_b4.py` (stdlib), `python test_a100.py` (mocked guard), `python test_resume.py` (small actual Torch writer), `python test_sequence.py` (synthetic actuator, no model evidence).

The delivered package has not itself run GPT-2 on A100 in the authoring workspace. These checks establish protocol logic and resume behavior, not an experimental PASS. Run the launcher and inspect `SUMMARY.json` before claiming results.

Parent repository evidence: `papasop/fibre-nav`, commit `d15b97d75f8541bf092352e2b7e8a2b16b3e720a`, B4 V1 minimal archive and B3 v2.2.4. This protocol extends causal controls and coverage; it does not replace the historical evidence.
