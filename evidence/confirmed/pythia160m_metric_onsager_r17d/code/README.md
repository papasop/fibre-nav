# R17d frozen five-seed confirmation

Confirms or rejects the R17c-nominated metric-constrained Onsager configuration
on five untouched seeds. Frozen multiplier: `1.15`; global AG-News response
budget: `0.004543482202852718`; learning target: SST-2; model: pretrained
Pythia-160M; 80 steps and a shared 32-dimensional LoRA chart.

Primary gates: median loss beats projected AdamW, at least 4/5 seedwise wins,
median accuracy is noninferior within 0.5 percentage points, current metric
beats source-frozen metric, and every numerical/response gate passes.

Run `COLAB_LAUNCHER_R17D.py`, upload the one ZIP, and download the result ZIP.
Exit status 2 means a scientific gate failed, not an engineering crash.
