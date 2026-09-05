# Source manifest — MFI L3 v1.9.1

- Frozen fresh seed: `82101`
- New balanced holdout split: two concepts per category
- Frozen 24-concept split, gates and controls retained
- 600-step slot-factorized writer retained
- Hybrid mean-pooled and final-token description representation
- Ridge router fit only on 16 training concepts
- Confidence derived from the top-two router score margin
- Cosine category prototypes fitted on training concepts only
- Low-confidence ridge decisions switch to the training-only prototype
- Geometry audited in predeclared response-kernel category coordinates
- Held-out labels excluded from fitting, switching and metric construction
- Category projection rank increased from 3 to 4
- Code rotations increased from 1 to 4
- Semantic pull weight reduced from 2.0 to 1.0
- Smooth move range 0.75–6.0; center 0.50; temperature 0.20
- Four tetrahedral category directions in the declared response kernel
- Prospective backtracking, response retraction and post-move memory audit
- Shuffled-category, pair-only and no-graph controls retained
- Whole-concept holdouts excluded from fitting and checkpoint selection
- L4 not executed
- Scientific status: prospective confirmation pending

Exact hashes are stored in `PACKAGE_SHA256SUMS.txt` when packaged.
