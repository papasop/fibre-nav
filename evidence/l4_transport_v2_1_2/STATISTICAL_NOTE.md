# Statistical note

The reported `28/28` wins compare all unordered distances among eight transported endpoints with matched-random controls. These 28 distances share endpoints and therefore are not independent Bernoulli trials.

The recorded sign-test value (`3.72529e-09`) is retained as a protocol diagnostic, but it should be described as a **naive dependent-pair sign test**, not as an independent confirmatory p-value. The strongest defensible evidence in this run is the predeclared all-gates pass, the distortion ratio (`0.1840`), the complete descriptive win count, preserved read access, and round-trip result. A future confirmation should use independent atlas-level replicates, seeds, or a permutation/bootstrap procedure whose resampling unit respects endpoint dependence.
