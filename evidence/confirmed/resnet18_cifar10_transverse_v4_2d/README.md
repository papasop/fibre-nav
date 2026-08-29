# ResNet-18/CIFAR-10 transverse confirmation v4.2d

This is the authoritative evidence package for Section 6. Sixteen new seeds
(76742--76757) prospectively test transverse response amplification in a
trainable ResNet-18 layer4-plus-classifier system using an eight-dimensional,
matrix-free transported response-kernel chart in float64 with TF32 disabled.

All 16 seeds passed the frozen joint gate (required 12/16). Recomputed from the
per-seed records, the medians are: moving/fixed direction cosine 0.9999899860,
active residual ratio 1437.843934, transverse response gain 90.108233,
transverse gain contrast 341763.1519, and finest finite-difference/JVP ratio
0.9999980209. The median per-seed maximum JVP additivity error is 5.05e-14.

`raw/raw_results.zip` is byte-preserved and authoritative. `results/` is its
expanded JSON representation. `code/` contains the frozen executable package.
The result establishes amplification in the declared chart and radii, not a
complete kernel bundle, universal singular-spectrum law, cost advantage, or
global variational theorem.
