# CPU entry points: C196, C197, C198 and C199

`launchers/` and `original_bundles/` retain four original byte-identical Colab
launcher/BUNDLE pairs. `source_snapshot/` is the unchanged complete C199 source
tree, with the exact earlier roots preserved in its vendored chain. `RUN_INDEX.json`
maps each stage to its original source root, hashes, protocol and result status.

| Stage | Function | Current evidence |
| --- | --- | --- |
| C196 | Eight fixed offline training steps, fixed-final qualification | Supplied execution log reports84/84 in both rounds |
| C197 | Pure NumPy read-only C196 audit | Supplied execution log reports completion; no new model measurements |
| C198 | Frozen native GPT-2 training replay, including fresh feature acquisition | Supplied execution log reports84/84 in both paths and rounds |
| C199 | Fixed48-state new-direction seen-class confirmation | Source/protocol only; execution pending |

## Colab

Use CPU. Paste or run the chosen original `Cnnn_CPU_LAUNCHER.py`, then upload
its **matching original BUNDLE ZIP once**, without manual extraction. Original
scientific inputs are located at pinned Drive paths, or supplied through the
launcher's optional `INPUT_PATHS`. A source BUNDLE, GitHub import, log summary or
similarly named result cannot replace a scientific input.

The complete scientific dependency chain and actual archived model-state arrays
remain external. Obtain every original result ZIP and match its whole SHA256
from [the dependency ledger](../../evidence/gpt2_lc2_c196_c199/REPLAY_DEPENDENCIES.json).
Owner-side Drive paths are not public download links. Fixed model downloads and
dependencies are still required for actual GPT-2 runs. C197 loads no model.

## Local entry point

Print one stage's prerequisites without importing model code:

```bash
python3 external_tests/gpt2_lc2_c196_c199/run_stage.py --stage C199
```

To execute with an existing complete local input chain:

```bash
python3 external_tests/gpt2_lc2_c196_c199/run_stage.py --stage C199 \
  --inputs-json /absolute/path/to/inputs.json --base /absolute/path/to/output-base
```

`inputs.json` must map every required stage input key to its exact local result
ZIP. Both execution options are required together. The unchanged original
launcher handles self-tests, dependency checks, fixed budgets, one success or
failure result package and explicit result status. This wrapper disables Drive
mounting and browser download; it does not alter the scientific protocol.

C199's conditional preflight failure is a completed stopped branch, not new-state
confirmation. Read `decision`, `known_preflight_passed`,
`confirmation_evaluation_complete` and `local_confirmation_passed` together.
Do not count repeated exposed states as unseen; LC2 remains open.
