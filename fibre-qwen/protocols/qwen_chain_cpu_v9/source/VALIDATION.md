# V9 engineering validation

Fourteen engineering tests passed before delivery: ten cover program inventory,
scope, no oracle use by execution, actual wrong intermediate propagation,
malformed-output skips, wiring tampering, incomplete/environment/numerical failure,
mid-program resume without repeated calls, budget pause and sequence gaps; four
cover launcher source/result hashes, path safety, old-protocol rejection and
isolated subprocess argument handling.

The controller mock uses preserved V6 model observations to exercise execution
and audits; these are not new V9 inference results. The final source ZIP is checked
by the real launcher and worker --check-only path without loading model weights.
The upload path is checked with a simulated Colab uploader, including a renamed ZIP.

The unchanged bounded backend also receives a random tiny Qwen CPU smoke check.
Local smoke-test dependencies are torch 2.6.0+cpu / transformers 4.57.1, while the
target experiment remains torch 2.8.0+cpu / transformers 4.56.2. A tiny random model
is an implementation check, not pretrained Qwen3-1.7B scientific evidence.

No pretrained V9 model run, training, parameter-memory write, Hugging Face upload,
paid computation or scientific pass has been performed for this delivery.
