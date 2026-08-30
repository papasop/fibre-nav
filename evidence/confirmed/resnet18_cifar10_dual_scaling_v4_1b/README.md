# ResNet-18/CIFAR-10 dual-scaling confirmation v4.1b

This is the authoritative evidence package for Section 5.3. Sixteen new seeds
(68726--68741) prospectively compare current-kernel transport with source-fixed
replay in a trainable terminal residual adapter plus complete classifier over
a frozen ImageNet-pretrained ResNet-18 backbone.

All 16 seeds passed every frozen geometric and held-out gate. Recomputed from
the per-seed records, the medians are: moving-minus-fixed slope separation
0.5751018997, smallest-radius fixed/moving cost ratio 2.996962733, raw
development/confirmation gradient cosine 0.4143130675, and projected cosine
0.9963029752.

`raw/raw_results.zip` is byte-preserved and authoritative. `results/` is its
expanded JSON representation. `code/` contains the frozen executable package.
The claim is finite-radius and construction-specific; it is not a full
backbone result, universal scaling law, F16 ordering, or global variational
theorem.
