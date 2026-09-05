# Repository direction audit

Audit target: `papasop/fibre-nav`, default branch `main`, inspected at commit
`5e5225dcd06cc9c22d07c55b191e7c338e9d0778`.

## Recommendation

Make L1-L4 the default scientific interface, but do not erase the historical
evidence. Git history already records the development chain, and the older
ResNet, GPT-2/Pythia/Picard, GPTW and Fibre-Qwen records remain useful for
provenance and claim boundaries.

Recommended target layout:

```text
README.md                    L1-L4 overview and status matrix
src/mfi/                     reusable current implementation
protocols/l1/                exact memory
protocols/l2/                semantic access
protocols/l3/                concept/category geometry
protocols/l4/                transport and connection tests
evidence/confirmed/          frozen multi-seed evidence
evidence/developmental/      frozen single-seed and negative results
archive/legacy/              superseded launchers and import notes
paper/                       current paper plus paper/archive
```

## Keep

- Frozen configs, raw result archives, console logs and SHA-256 manifests.
- Negative and partial milestones such as v1.8.0 and v1.8.1.
- Claim-boundary documents and external reproduction entry points.
- The current paper and one explicit paper archive.

## Move out of the repository root

- `GITHUB_IMPORT*.md`, `GITHUB_REPAIR*.md`, `README_*_INSERT.md` and old
  one-off ingestion instructions: move to `archive/legacy/import_notes/`.
- Version-specific root manifests: move beside their corresponding evidence;
  retain one root manifest only if it verifies the entire current tree.
- Version-specific root verification scripts: move beside the evidence they
  verify, leaving one top-level `verify.py` dispatcher.

## Do not do yet

- Do not rewrite Git history or physically purge old evidence.
- Do not rename every historical directory in one large commit.
- Do not place v1.8.1 under `evidence/confirmed/`.
- Do not present L4 as implemented until a frozen transport/connection protocol
  has passed its declared gates.

Use a separate housekeeping commit after v1.8.2, so scientific changes and
repository reorganization remain independently reviewable.
