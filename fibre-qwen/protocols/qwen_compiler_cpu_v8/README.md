# Qwen compiler CPU V8 — development archive

Status at import: **partial log, 18/128 model checkpoints; final result not available**.
See the [V6–V8 evidence snapshot](../../evidence/qwen_cpu_interface_v6_v8/README.md).

This directory preserves the delivered launcher, source ZIP and extracted frozen
source byte for byte. `source/VALIDATION.md` is the original pre-delivery engineering
report; its statement that pretrained V8 was not run refers to that validation,
not to the subsequently supplied partial Colab log.

V8 parses four declared input formats into `(op, a, b)`, compiles them to 16 frozen
canonical prompts, and asks frozen Qwen3-1.7B to choose a Boolean output. There are
64 accepted source inputs, a full repeat (128 model checkpoints), and 24 invalid
inputs that must be rejected before calling the model. The parser does not compute
the answer. The output grammar retains both model bit logits and forces EOS after
the selected bit. This tests an externally compiled, bounded interface.

No training, adapter, parameter-memory write, free-generation confirmation,
autonomous instruction execution, or deployment claim is made. The 128 checkpoints
are not 128 independent prompts. V8 changes the interface relative to V7; it cannot
retroactively turn the failed V6/V7 protocols into passes. This import occurs after
partial results were observed and is not prospective public preregistration.

## Run

In a Colab notebook, upload and execute `COLAB_LAUNCHER_QWEN_COMPILER_CPU_V8.py`,
then select the adjacent `qwen_compiler_cpu_v8.zip` in its upload dialog. An optional
V8 results ZIP can be supplied for resumption. The launcher uses CPU only, installs
isolated dependencies without venv, and saves result archives at checkpoints.
See [the frozen instructions](source/README.md) for resource and resume details.

From the repository root, engineering tests without model weights:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s fibre-qwen/protocols/qwen_compiler_cpu_v8/source -p 'test_*.py' -v
python3 fibre-qwen/evidence/qwen_cpu_interface_v6_v8/verify_archive.py
```

Passing these checks validates the archive and implementation contracts; it does
not establish a pretrained-model experimental pass.
