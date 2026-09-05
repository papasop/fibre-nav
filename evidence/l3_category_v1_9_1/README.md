# Moving Fibre Intelligence — L3 category geometry

This snapshot freezes the first fully gated single-seed L3 development result
(`v1.9.0`) and the prospectively specified independent check (`v1.9.1`).

## Current status

- `v1.9.0`: all declared L3 gates passed on development seed `82001`.
- `v1.9.1`: **PROSPECTIVE_CONFIRMATION_PENDING** on fresh seed `82101` and a
  new balanced whole-concept holdout split.
- L4 cross-fibre transport is not run or claimed here.

The v1.9.0 mechanism combines a ridge router with category prototypes fitted
only on training concepts. A low-confidence ridge decision switches to the
training-only prototype. Geometry is evaluated in predeclared response-kernel
category coordinates. In the frozen development result, held-out category
accuracy was 8/8, minimum held-out distance margin was `+1.9900`, and the
within/between ratio was `0.0301`; shuffled-category, pair-only and no-graph
controls failed at least one corresponding gate.

These are mechanism results in GPT-2 with a restricted LoRA-B parameter fibre,
not evidence for universal concept geometry. The development split had already
influenced mechanism design, so v1.9.0 is not an independent confirmation.

## Repository layout

- `src/v1_9_1/`: frozen prospective source and configuration.
- `colab/`: robust notebook launcher.
- `release_assets/`: upload-ready PY and ZIP.
- `evidence/v1_9_0/`: complete frozen development evidence.
- `CLAIM_BOUNDARY.md`: permitted and excluded interpretations.
- `REPRODUCE.md`: one-cell Colab procedure.

## Citation title

Until L4 is tested, the evidence-compatible title is:

**Moving Fibre Intelligence: Reusable Geometric Addressing in a Neural Response Fibre**

`Transportable Concept Geometry` remains reserved for a structure-preserving
cross-fibre transport result.
