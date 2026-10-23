# CPU reproduction entry points: C141, C142, C148, C150, C154, C155

`source_snapshot/` is the unchanged delivered C155 source tree, containing the
earlier six stage roots through its original vendored dependency chain. It is
stored once to avoid six expanded duplicate trees. `RUN_INDEX.json` maps each
stage to its exact source root, launcher, source-file hashes and original BUNDLE.
The original Colab launchers and matching BUNDLE ZIPs are also retained byte for
byte under `launchers/` and `original_bundles/`.

From the repository root, inspect prerequisites without loading a model:

```bash
python3 external_tests/gpt2_lc2_c141_c155/run_stage.py --stage C150
```

An actual execution requires a JSON mapping of **all** original input keys to
local result ZIP paths, and an explicit output directory:

```bash
python3 external_tests/gpt2_lc2_c141_c155/run_stage.py --stage C150 \
  --inputs-json /absolute/path/to/inputs.json --out /absolute/path/to/results
```

Alternatively paste the chosen original launcher into CPU Colab and upload its
matching original BUNDLE. Do not upload the GitHub import ZIP as a scientific
input or source BUNDLE. C141 uses the R1 launcher/BUNDLE, retaining its unchanged
scientific protocol. C142 is an archive audit; the other stages execute real GPT-2.

The compact evidence import does **not** include the full predecessor result ZIP
chain. Consult `evidence/gpt2_lc2_c141_c155/REPLAY_DEPENDENCIES.json`; hashes and
exact input identities remain mandatory. No dependency is replaced with a
synthetic fixture, similarly named ZIP, success-only subset or newer checkpoint.

Read-only verification is separate:

```bash
python3 evidence/gpt2_lc2_c141_c155/verify_records.py
```

This checks compact file identities, included actual tensors, reconstructed
proposals and original gate/count arithmetic. It does not replay GPT-2,
independently verify gradients, reconstruct every omitted offline preparation,
or establish excluded-input generalisation.
