# GPT-2 B5 XOR Continuation A100 V5

Development continuation of verified V4 2/4 results. No independent B5 confirmation.

- Inherit xor_00/xor_01 passing archived evidence, do not rerun or count as new V5 experiments.
- xor_10: restore actual preparation checkpoint at step 600; run original writer to total 1200 steps, then execute model-selected read/XOR/write up to 1080 writer steps.
- xor_11: load original passing prepared state 1111; restart original write procedure to maximum 1080 steps. V4 deleted its rejected write optimizer/candidate, so this is NOT continuation from step 480.
- Fixed V3 controller, zero new controller training. One frozen GPT-2 backbone, exclusive controller/memory adapters. Runtime routes adapters, computes XOR and numerical updates.
- Original B4 writer source and configuration are byte-identical. Fixed access rules, exact/development read margins >=0.05, 100% read accuracy, anchor-average response drift <=0.002 and anchor-average KL <=0.005.
- Development verification prompts remain outside the gradient objective and checkpoint selection. Previously inspected prompts are not fresh holdouts.
- Retain every V5 writer optimizer and selected candidate including rejected attempts. Rejected writes roll back the committed runtime state.
- Test controller full adapter parameter hash and probe at the actual write prompt, independently from memory candidate gates.
- Every 5 steps: atomic local result ZIP and verified Drive copy. Keep only latest 3 V5 backups in its own recovery directory. V4 backups unmodified.
- Resume incomplete V5 from actual checkpoints; skip completed cases, including recorded failures. Complete failures require a separately specified next protocol.

Run the self-contained .py in Colab A100, or execute launch_b5.py from the source directory. The default mounts Drive and selects verified V5 progress first, then a compatible complete V4 result. Set INPUT_ZIP in the self-contained launcher to select an explicit result. Do not upload the source ZIP as a result.

Output: one gpt2_b5_xor_continuation_a100_results.zip containing immutable parent V4 archive, inherited records, real checkpoints, source identity and all new success/failure evidence.

Local validation: 18 stdlib checks plus syntax/package validation. No local torch, GPU or GPT-2 numerical execution was performed. No prediction of acceptance.
