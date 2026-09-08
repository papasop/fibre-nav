# Personal research agent: continuity foundation

Development milestone A0. This runnable standard-library component supplies
persistent, project-scoped memory and task continuation for a future personal
agent. It does not yet call an LLM, interpret natural-language commands, execute
experiments, or change model weights. It requires no GPU or API credentials.
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
