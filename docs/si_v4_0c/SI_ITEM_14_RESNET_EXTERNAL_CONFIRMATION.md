# Supplementary Item 14: ResNet-18/CIFAR-10 external confirmation

## Frozen scope

v4.0c-r1 is a prospective sixteen-seed external confirmation of selected
response-fibre tangent value. It uses a frozen ImageNet-pretrained ResNet-18
backbone, CIFAR-10, and the complete 5,130-parameter final classifier.

```text
seeds: 64726--64741
anchors: 4, 16, 32
primary anchors: 16
matched random kernel controls: 32 per seed and anchor count
step radius: 0.08
```

The development direction is projected into the exact centered-anchor-logit
response kernel. Confirmation labels are used only to evaluate frozen
directions. The primary controls are sign reversal, shuffled-target projection
and the best of 32 random kernel directions. The unconstrained ambient gradient
is a positive control, never a feasible competition arm.

## Frozen gates

- base confirmation accuracy at least 0.70;
- primary 16-anchor candidate in at least 12/16 seeds;
- 4- and 32-anchor sensitivity candidate in at least 10/16 seeds each;
- dimensionless kernel residual at most 1e-12;
- finite centered-logit leakage at most 1e-9;
- true confirmation-loss improvement at least 0.001;
- anti-minus-true margin at least 0.002;
- best-random-minus-true margin at least 0.001;
- shuffled-minus-true margin at least 0.001;
- unconstrained ambient response leakage at least 0.01.

## Result

```text
RESNET_CIFAR_EXTERNAL_FIBRE_V40C_CONFIRMED
```

All 16 seeds passed every gate at 4, 16 and 32 anchors. At the primary 16-anchor
setting, median confirmation-loss changes were -0.01842 for the selected
tangent, +0.02307 for sign reversal, -0.00099 for shuffled target and -0.00072
for the best random tangent. The minimum per-seed margins over those three
controls were respectively 0.03059, 0.01366 and 0.01224.

The maximum primary kernel residual was 2.28e-13 and maximum primary finite
response leakage was 2.98e-13. Across 4, 16 and 32 anchors, the median projected
gradient share decreased from 0.453 to 0.338 to 0.282, while all functional
gates remained 16/16.

## Development and repair history

v4.0a remains unsupported; v4.0a-r1 remains developmental; v4.0b remains
formally unconfirmed. v4.0c-r0 produced no seed result because of a pre-result
float32/float64 dtype mismatch. v4.0c-r1 changed only the dtype of matched
random draws and added diagnostic logging. No scientific threshold, seed,
dataset split, control or gate was changed.

## Claim boundary

This confirms selected response-fibre tangent value in the complete final
classifier of a frozen ResNet-18 representation. The sixteen seeds vary the
classifier-training initialization; they do not constitute sixteen independent
datasets, pretrained representations or architectures. No result is claimed
for backbone adaptation, moving response fibres, realizability-cost scaling,
Moving-Fibre F16, LLMs or arbitrary-path variation.

## Authoritative files

Machine-readable derived tables are included as `SI_V40C_ANCHOR_SUMMARY.csv`
and `SI_V40C_SEED_GATES_48.csv`. They are generated from the archived report by
`audits/resnet_v4_0c/build_si_tables.py`; they are not independent data.

- engine SHA-256: `dca49cb187c59a2ac8e3e55fe4970349f258fed36b7ffbfca510770c19af9f63`
- raw-results SHA-256: `a3edf61ed8fcd029f302f922650071dec9a1b60c501ace34a4ec6965ebf71823`
