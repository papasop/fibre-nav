# Personal research agent: continuity foundation

Development milestone A1 (includes the A0 continuity foundation). This runnable standard-library component supplies
persistent, project-scoped memory and task continuation for a future personal
agent. The original memory CLI does not call an LLM. The optional `assistant.py`
loop now connects a configured model to read-only tools; it does not execute
experiments or change model weights. It requires no GPU or API credentials.
It supplements the existing Fibre-Qwen experiments without changing their gates.

## Run (Python 3.11+)

From the repository root:

```bash
python fibre-qwen/agent/research_agent.py --project demo remember audit "Awaiting replication" --source "local:notes/experiment.md" --status hypothesis
python fibre-qwen/agent/research_agent.py --project demo task "Review experiment" --next-action "Read the frozen result JSON and check its declared gates"
python fibre-qwen/agent/research_agent.py --project demo resume
python fibre-qwen/agent/research_agent.py --project demo recall audit
python fibre-qwen/agent/research_agent.py --project demo history audit
# Use the actual task ID returned by task:
python fibre-qwen/agent/research_agent.py --project demo done 1 --source "local:results/review.json"
python -m unittest discover -s fibre-qwen/agent/tests -v
```

State defaults to `~/.fibre-agent/state.sqlite3`, outside the repository.
Pass `--db /persistent/path/state.sqlite3` before the subcommand to select storage.
For a hosted deployment use a genuinely persistent private volume and backups;
an ephemeral container filesystem will not preserve memory across rebuilds.
The database is plaintext: protect it with the host's access controls. No data is
uploaded, and no private user records are bundled in this code.

Updating a key appends a revision. Retrieval selects the newest revision before
matching literal, case-insensitive substrings (including Chinese), so superseded
claims do not reappear just because they match the query. History retains all
revisions. Projects are isolated by exact project name. Resume returns the oldest
pending task and current project memory; it never silently marks a task complete.
Completion requires an explicit source reference. Source references and evidence
labels are assertions supplied by the caller, not independently checked facts.
Large-scale semantic retrieval, authentication and multi-user access are not implemented.

## Next integration gate

Connect this context to a model adapter and a small allowlisted read-only tool
loop. Treat retrieved text as untrusted evidence, never as authority to execute
commands. Evaluate held-out research tasks against the same model without
persistent context, measuring task correctness, stale-memory errors, cost and
latency before enabling writes or experiment execution.

MFI remains an experimental backend candidate. This SQLite memory is not neural
memory, L1–L5 confirmation, autonomous learning, or evidence of personalization.
Keep the historical R20–R23 roadmap and review prerequisites intact. Any later
MFI comparison must hold model, tools and task set fixed and report failures.

## A1: model and read-only tools

`assistant.py` sends the question and requested tool results to an explicitly
configured chat-completions-compatible endpoint. It supports local Qwen-serving
endpoints or a compatible remote service; no provider/model is selected implicitly.
Set `FIBRE_AGENT_API_KEY` in the process environment if the service requires it.
Never commit credentials. With a remote endpoint, the question, selected project
memory and read file contents are transmitted to that provider and may incur
inference costs. Expose only a curated evidence directory, not your home directory.

Example with an already-running local server (replace MODEL_ID with its model ID):

```bash
python fibre-qwen/agent/assistant.py --root ./evidence_excerpt --project demo --endpoint http://127.0.0.1:8000/v1/chat/completions --model MODEL_ID "Read the result files, distinguish reported results from hypotheses, and propose the next validation."
```

Create `evidence_excerpt` and place selected small text results there first.
This command does not install/start a model server. Use `--db` to select the same
persistent database used by the A0 CLI. No GPU or endpoint is started by this code.

The model chooses JSON actions: `list_files`, `read_file`, `recall`, `resume`, or
an `answer`. Files carry a relative source path and SHA-256. Hidden paths,
symlinks, paths outside the selected directory, unsupported formats and files
larger than 16 KiB are rejected. Lists and memory output are bounded. There is no
shell/write tool. The loop stops after 8 model turns by default (maximum 20),
returns `step_limit` rather than fabricated success, and never marks tasks done.
Do not run against a directory being concurrently modified by untrusted actors;
this is a local development tool, not an OS sandbox or multi-user security boundary.

The JSON output includes the answer and actual tool trace. An answer remains
model-generated: citations are not automatically validated, and prompt-injection
resistance is not established by these tests. The model can answer without a tool;
inspect its trace before treating claims as grounded. Model output must be a JSON object; one complete JSON Markdown fence is accepted.
JSON syntax errors receive correction feedback within the existing turn budget.
There is no fallback model. Schema errors still fail explicitly.
No conversation transcript is saved automatically.

Validation: 8 tests pass, including a local HTTP server exercising the real client,
file read and answer loop; project persistence; task continuity; path restrictions;
invalid actions; and step exhaustion. The HTTP model is scripted, not a real LLM.
Real-model instruction following, held-out task success, source accuracy, latency
and cost remain unmeasured. Next gate: run the same frozen research questions with
and without memory on a configured model before enabling any writing tools.


## A2: paired development evaluation and direct Qwen backend

Run `evaluate.py` against either the existing HTTP adapter or directly loaded
Qwen weights. Three fixed synthetic tasks test file-grounded reporting, a revised
release decision, and pending-task continuation. Each is run with populated and
empty memory, for six independent contexts. Both arms get identical file tools
and questions; arm order alternates. The memory cases intentionally require
information unavailable to the empty arm: this is a continuity mechanism check,
not a fair broad-intelligence comparison or a held-out generalization study.

An answer must match the declared output and show the required successful tool
call. These limited automatic gates do not replace semantic/human review.
Malformed outputs, backend errors and exhausted loops remain in the results.
Reports include all tool traces, wall times, model identity, protocol and code
hashes. Existing reports cannot be overwritten. Token usage and monetary cost
are not measured. HTTP model revision is operator-supplied, not server-verified.

For direct Qwen execution in a Python environment with PyTorch available:

```bash
pip install transformers==4.56.2 safetensors
python fibre-qwen/agent/evaluate.py --backend qwen --model Qwen/Qwen3-0.6B --revision MODEL_COMMIT_SHA --device cpu --output /persistent/private/a2-run-001.json
```

Replace MODEL_COMMIT_SHA with the model's actual immutable 40-character revision.
Select `--device cuda` only in an already-provisioned GPU environment. Model
weights must be locally cached or downloadable; this command does not provision
hardware. Loading may require substantial RAM and CPU runtime. No LoRA adapter
is loaded: this establishes the base-model baseline before adapter comparisons.

For a configured compatible inference server:

```bash
python fibre-qwen/agent/evaluate.py --backend http --model YOUR_MODEL --revision SERVER_MODEL_VERSION --endpoint http://127.0.0.1:8000/v1/chat/completions --output /persistent/private/a2-run-001.json
```

Initial implementation status was **REAL_MODEL_NOT_RUN**. The first real CPU run is now archived in [A2 results](results/a2_qwen_cpu_001/README.md): both arms passed 0/3, with failures in action formatting and tool selection. Eleven unit/integration tests
passed, including a scripted paired fixture and retention of all backend failures.
Scripted fixture scores are test expectations, not Qwen results. The development
benchmark is public and must not later be relabeled as untouched confirmation.
A successful CLI exit means the evaluation ran without backend errors; it does
not mean every task passed. Read the per-arm counts and original traces.


## A2 adaptive repairs: still blocked on model execution

[Repair 1](results/a2_qwen_cpu_repair_001/README.md) and
[repair 2](results/a2_qwen_cpu_repair_002/README.md) each scored 0/3 in both arms.
The first retrieved memory but emitted malformed final answers; the second
emitted valid final answers while skipping tools. Complete raw calls and exact
source snapshots are archived for both attempts. Thirteen engineering tests pass;
they are not model performance evidence. These are same-task development repairs.

The current free tool-selection loop remains experimental and can produce
ungrounded answers; passing JSON syntax does not imply evidence-backed content.
Next proposed approach: a separately labeled controlled workflow that retrieves
requested evidence before model synthesis, compared with this autonomous baseline.
Do not claim autonomous tool selection for that future controlled workflow.
