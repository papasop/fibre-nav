# A2 repair 2: real Qwen development failure

Both memory arms passed **0/3** under the unchanged task correctness and
required-tool gates. Model: Qwen3-0.6B at revision
`c1899de289a04d12100db370d81485cdf75e47ca`; CPU float32, four threads.
Wall time excluding model loading: 18.61 seconds.
No weight training, GPU startup or paid model API.

Repair 2 adds an explicit tool-name list, a concrete final-answer JSON example,
and bounded feedback for JSON syntax errors, retaining raw invalid replies in
the trace. The model instead answered immediately without any tool call:
REPORT_PASS for both file tasks, UNKNOWN for memory and continuation tasks.
The scorer rejected all six outputs. Syntax compliance is not grounded task success.

This is adaptive repair on the same public development questions, not independent
confirmation. All attempts are retained; none establishes MFI or memory utility.
The original 0/3 baseline remains in a2_qwen_cpu_001. The same attention-mask and
greedy-generation warnings as the baseline occurred. No decoding settings changed.

To reproduce this exact version, copy snapshot_assistant.py,
snapshot_evaluate.py and snapshot_research_agent.py to a new directory as
assistant.py, evaluate.py and research_agent.py. Run evaluate.py with the model
and revision above and a new --output path; install the versions in report.json.
The snapshots' bytes match report.code_sha256. Current main code may differ.
model_calls.jsonl was captured by a logging wrapper around model.reply with no
changes to prompts, decoding or scoring. It contains only synthetic task data.
