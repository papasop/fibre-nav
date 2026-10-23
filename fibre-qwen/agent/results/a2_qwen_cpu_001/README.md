# A2 first real Qwen run — failed development baseline

Real CPU inference using Qwen/Qwen3-0.6B, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, torch 2.8.0+cpu,
transformers 4.56.2, float32, four CPU threads, greedy decoding.
Evaluated the unchanged A2 protocol/code from commit
`539caa6084b7890003201dd46c1a04ee4f8b85d0`.

| Case | Without memory | With memory | Observed failure |
| --- | --- | --- | --- |
| Read result | fail | fail | Markdown-wrapped JSON; copied example path |
| Current memory | fail | fail | list_files repeated until six-step limit |
| Continue task | fail | fail | Unsupported next_action action object |

Both arms passed 0/3. No required tool was successfully invoked in any case.
The full model loop took approximately 59 seconds, excluding installation,
downloads and model loading. These are synthetic public development tasks.
The failure localizes to instruction/action execution for this model/prompt/
backend combination; it does not establish failure of SQLite memory or MFI.
Memory benefit cannot be evaluated when the model never retrieves the memory.
No trained adapter was loaded. No GPU or paid inference service was started.

`report.json` preserves six rows, code/protocol hashes and dependency versions.
`model_calls.jsonl` preserves all 16 original model replies and input messages,
including malformed replies which the evaluator's error rows alone omit.
Only synthetic records are included. The original backend emitted an attention
mask warning (unbatched, unpadded prompts) and a warning about ignored sampling
settings under greedy decoding; both are retained here as execution caveats.

Reproduce in the matching environment, from the repository root:

```bash
OMP_NUM_THREADS=4 HF_HUB_DISABLE_XET=1 python fibre-qwen/agent/results/a2_qwen_cpu_001/reproduce.py /path/to/new-run-directory
```

The wrapper only records model calls and metadata around evaluate(); it does
not modify prompts, parsing, scoring or decoding. Install the recorded torch
CPU build and transformers version first. Model downloads require network access.

Next development step: explicit action schemas and task-specific tool guidance,
with malformed-action diagnostics and a repeated-action stop. Run repairs under
a new protocol identifier and keep this baseline unchanged. Any same-task repair
is development evidence, not fresh confirmation or improved generalization.
