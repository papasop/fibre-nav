```
CPU 模式，GPU 不参与计算。Profile: compiled_interface
Qwen compiler V8：声明语法解析＋冻结模板，128个模型检查点，仅受限生成；纯CPU，无训练。
请上传 qwen_compiler_cpu_v8.zip；续跑可同时上传一份 V8 *_results.zip。

```

1. **qwen\_compiler\_cpu\_v8.zip**(application/zip) - 46759 bytes, last modified: 2026/9/23 - 100% done

```
Saving qwen_compiler_cpu_v8.zip to qwen_compiler_cpu_v8.zip
安装隔离的 CPU 依赖（首次需要联网）；无需 venv 或 ensurepip。
Loading pinned Qwen3-1.7B: CPU float32; no Adapter or training.
`torch_dtype` is deprecated! Use `dtype` instead!

Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]
Fetching 2 files:  50%|█████     | 1/2 [00:07<00:07,  7.39s/it]
Fetching 2 files: 100%|██████████| 2/2 [17:23<00:00, 521.63s/it]
Fetching 2 files: 100%|██████████| 2/2 [17:23<00:00, 521.63s/it]

Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:14<00:14, 14.87s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:18<00:00,  8.36s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:18<00:00,  9.33s/it]
The following generation flags are not valid and may be ignored: ['top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
QWEN_COMPILER_V8_POINT {"n": 1, "id": "canonical-00-OR", "accepted": true, "parsed": {"op": "OR", "a": 0, "b": 0}, "bounded": "0", "compiler_matches": 1}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 2, "id": "canonical-00-AND", "accepted": true, "parsed": {"op": "AND", "a": 0, "b": 0}, "bounded": "0", "compiler_matches": 2}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 3, "id": "canonical-00-XOR", "accepted": true, "parsed": {"op": "XOR", "a": 0, "b": 0}, "bounded": "0", "compiler_matches": 3}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 4, "id": "canonical-00-EQUAL", "accepted": true, "parsed": {"op": "EQUAL", "a": 0, "b": 0}, "bounded": "1", "compiler_matches": 4}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 5, "id": "canonical-01-OR", "accepted": true, "parsed": {"op": "OR", "a": 0, "b": 1}, "bounded": "1", "compiler_matches": 5}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 6, "id": "canonical-01-AND", "accepted": true, "parsed": {"op": "AND", "a": 0, "b": 1}, "bounded": "0", "compiler_matches": 6}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 7, "id": "canonical-01-XOR", "accepted": true, "parsed": {"op": "XOR", "a": 0, "b": 1}, "bounded": "1", "compiler_matches": 7}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 8, "id": "canonical-01-EQUAL", "accepted": true, "parsed": {"op": "EQUAL", "a": 0, "b": 1}, "bounded": "0", "compiler_matches": 8}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 9, "id": "canonical-10-OR", "accepted": true, "parsed": {"op": "OR", "a": 1, "b": 0}, "bounded": "1", "compiler_matches": 9}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 10, "id": "canonical-10-AND", "accepted": true, "parsed": {"op": "AND", "a": 1, "b": 0}, "bounded": "0", "compiler_matches": 10}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 11, "id": "canonical-10-XOR", "accepted": true, "parsed": {"op": "XOR", "a": 1, "b": 0}, "bounded": "1", "compiler_matches": 11}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 12, "id": "canonical-10-EQUAL", "accepted": true, "parsed": {"op": "EQUAL", "a": 1, "b": 0}, "bounded": "0", "compiler_matches": 12}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 13, "id": "canonical-11-OR", "accepted": true, "parsed": {"op": "OR", "a": 1, "b": 1}, "bounded": "1", "compiler_matches": 13}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 14, "id": "canonical-11-AND", "accepted": true, "parsed": {"op": "AND", "a": 1, "b": 1}, "bounded": "1", "compiler_matches": 14}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 15, "id": "canonical-11-XOR", "accepted": true, "parsed": {"op": "XOR", "a": 1, "b": 1}, "bounded": "0", "compiler_matches": 15}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 16, "id": "canonical-11-EQUAL", "accepted": true, "parsed": {"op": "EQUAL", "a": 1, "b": 1}, "bounded": "1", "compiler_matches": 16}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
STAGE_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_016_results.zip
已保存：/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_016_results.zip

```

```
QWEN_COMPILER_V8_POINT {"n": 17, "id": "sentence-00-OR", "accepted": true, "parsed": {"op": "OR", "a": 0, "b": 0}, "bounded": "0", "compiler_matches": 17}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip
QWEN_COMPILER_V8_POINT {"n": 18, "id": "sentence-00-AND", "accepted": true, "parsed": {"op": "AND", "a": 0, "b": 0}, "bounded": "0", "compiler_matches": 18}
RESULT_ARCHIVE=/content/qwen_compiler_cpu_v8/run/cpu_compiled_interface_results.zip

```

Downloading "cpu\_compiled\_interface\_016\_results.zip": 