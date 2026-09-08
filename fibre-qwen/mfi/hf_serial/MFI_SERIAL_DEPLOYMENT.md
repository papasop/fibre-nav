# Serial overwrite endurance audit 001

Manual-only CPU jobs; no model loading at startup. Run seed 84503, download its records, then run seed 84521. Each job has a 30-minute hard cap including model download. Timeout is incomplete, not a scientific failure.

Each seed initializes exactly one chart/controller. Twenty planned operations alternate the opposite and original bit: WRITE followed by 19 OVERWRITEs. The response and full-vocabulary anchor KL reference never resets. Existing writer code and budgets are unchanged (response .02, KL .01, margin 1, 24 steps per operation).

A rejected operation must restore the pre-operation vector; remaining operations are skipped, with all 160 planned format-choice items per arm retained. Task evaluation restores the committed vector. Four newly declared prompts with both label orders use the existing scorer. External and no-memory baselines run on initial weights. No task feedback enters the writer.

Scope: one-bit hybrid read-to-prompt preference endurance. This does not establish free-text answer correctness, L2-L5 support, general memory capacity or autonomous policies. Two new seeds are a small cohort, not broad reliability certification. Downloaded logs, snapshots, protocol, and per-stage records support audit.

Existing repaired app/config/archive remain in the Space; Dockerfile.repaired.backup restores the previous entry point. No paid hardware changes are required.
