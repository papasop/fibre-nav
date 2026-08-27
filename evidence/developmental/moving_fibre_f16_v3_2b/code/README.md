# Moving-Fibre F16 v3.2b: multi-radius scaling audit

This is the frozen four-seed development test following v3.2a. It asks whether natural gradient's larger finite-step response-fibre retraction cost vanishes with step radius while its restricted Moving-Fibre F16 action ordering remains stable.

## Frozen design

- New seeds: 72726--72729.
- Six causal online algorithms.
- Four step radii: 0.08, 0.04, 0.02, and 0.01.
- 24 paths per seed and 96 paths in total.
- Every step transports an eight-dimensional chart and projects it into the current response kernel.
- Every path targets the same 20% capability-loss reduction.
- No fitted scalarization coefficient is introduced.

The primary gate requires at least three comparable seeds in which natural gradient has the minimum Moving-Fibre F16 action at every radius, beats wrong-Fisher natural gradient at every radius, and the six algorithms show positive, well-fit retraction-cost scaling. The two smallest radii must also give converged action estimates. Wrong-metric ordering and moving-fibre rotation remain specificity checks.

A pass supports only the discretization branch: the extra retraction cost is consistent with a finite-step effect while the restricted action ordering survives refinement. It does not establish a continuum theorem, arbitrary-path or global minimality, GPT-2 transfer, or a universal learning law. A pass authorizes a separately frozen 16-seed v3.2c confirmation.

## Colab

Run `COLAB_LAUNCHER_MOVING_FIBRE_F16_V3_2B.py`. An A100 is recommended; this is substantially heavier than v3.2a.
