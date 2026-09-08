# Independent task effect evaluation

The execution chain remains deterministic agent controller -> MFIController ->
restricted Qwen LoRA-B -> finite response/KL/margin gates and transactional rollback.
`evaluate_tasks.py` adds a downstream scorer that cannot feed task answers or
scores into the operation optimizer. The existing core and Qwen implementation
are unchanged.

Run in the existing pinned CPU environment:

```sh
python evaluate_tasks.py --out results/task_preference_NEW
```

The output directory must be new. Protocol, source hashes and source snapshots
are saved before computation. Exit 2 denotes failed declared evaluation gates.

## Task and comparison

A binary preference represents brief (0) versus detailed (1) answers. Four English
requests, each with both label assignments, ask Qwen to choose the response format.
The answer is scored by restricted two-token argmax. This tests a format decision,
not the quality of a generated answer. These prompts are absent from the operation
loss and response anchors. They are a predeclared task smoke, not a secret benchmark.

Three fixed chart seeds each have two planned stages: WRITE the opposite initial
bit, then OVERWRITE the initial bit. Each arm has 48 planned task items, grouped
within six state transitions; label flips and repeated prompts are correlated.

- MFI: use the committed restricted bit readout as preference text, and run the
  task on the updated model. Never inject the desired target as MFI memory.
- External memory: supply the desired preference as text on the initial frozen
  chart. This is an ideal reliable external-memory baseline, not retrieval noise.
- No memory: provide no saved preference, also on the initial frozen chart.

All arms use the same task wording and label orders. MFI and external memory have
identical task prompt text when the MFI readout is correct. The evaluation restores
the committed parameter vector after all baseline probes, including on exceptions.
A rejected or skipped operation gets zero MFI workflow successes across all eight
planned items. Baselines are still scored. This prevents selection of only successful
writes from inflating the end-to-end score. A skipped stage is separately visible.

The report separates operation acceptance from task correctness. Declared aggregate
success requires all six operations, MFI accuracy greater than no memory, and MFI
accuracy at least equal to external memory. These are smoke gates, not significance
tests. Raw predictions, references, memory readouts and operation traces are archived.

## Boundaries and next evidence

This is a hybrid parameter-read-to-prompt workflow with a deterministic externally
specified policy. It does not show that Qwen spontaneously uses modified parameters
as semantic memory, autonomously selects instructions, or improves broad generation.
It cannot establish a parameter-memory advantage over external memory. There is
no new SWAP implementation or L2-L5 claim. The existing single-cell backend supports
WRITE and OVERWRITE only.

No task-specific gradient tuning, prompt repair, budget relaxation or seed selection
is performed after viewing results. A future policy that chooses whether/when to
write must be tested against this fixed policy, and on separately frozen tasks.
Before that, failed operation gates remain a reliability problem to resolve.
