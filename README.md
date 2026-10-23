# Moving Fibre Intelligence

We study **response-constrained programmable parametric memory**, moving writing
from per-request numerical optimisation toward **frozen, single-proposal parameter
updates**. Current experiments support local development feasibility. Transfer to
new states, generalisation to excluded requests, and update internalisation are
validated separately.

The long-term framework is **LC1–LC6: six capabilities with separate acceptance criteria**,
not completed stages in a single hierarchy. B4 supplies the causal read–compute–write
foundation, B5 demonstrates model-selected operations, and LC1 has log-supported
local frozen-writer results. **LC2 excluded-request generalisation is the near-term
focus and remains open.** LC3–LC6 are separate open targets. B5 plus a frozen
writer has not yet been accepted as one integrated system.

> **Start here:** [Unified evidence matrix, verification tiers and gaps](docs/PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md)
> · [Parametric-memory research and historical evidence](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md)
> · [C141–C155 development evidence and numerical diagnostics](docs/LINE_C_C141_C155.md)
> · [LC1–LC6 validation framework](docs/LINE_C_VALIDATION_TARGETS.md)
> · [Proposed B5/frozen-writer integration](docs/B5_FROZEN_WRITER_INTEGRATION.md)

The repository also contains response-fibre geometry, optimization and Qwen
experiments. Their acceptance criteria and evidence statuses remain separate.

## Research question and contribution

The aim is to move from solving each write numerically to a frozen, single-proposal
writing mechanism, while retaining selective access and explicit response budgets.
The contributions are the operational interface, acceptance protocols, causal
interventions and traceable experimental evidence. Residual prediction is a
mechanism to investigate within this problem, rather than the definition of the
research programme. Standard null-space projection provides a local constraint
mechanism; it is not presented as a new mathematical discovery.

## Evidence and current status

| Capability | Evidence available here | Boundary / next test |
| --- | --- | --- |
| Selective access and writing (B0–B3) | Results within the [paper's scope](paper/A_Programmable_Address_Space_for_Neural_Memory.pdf), under legacy L1–L5 labels | Separate audits; no unified multi-seed confirmation. |
| Causal read–compute–write (B4) | [Minimal causal XOR development pass](evidence/gpt2_b4_minimal_causal_a100_v1/): parameter reads, external XOR, parameter writeback and interventions | One development seed, fixed input `01`; [full B4 XOR V2](external_tests/gpt2_b4_full_xor_a100_v2/STATUS.json) remains a pending protocol. |
| Learned operation selection (B5) | [V5 four-input XOR development coverage, 4/4](evidence/gpt2_b5_xor_continuation_a100_v5/), retaining [V4's 2/4 failure](evidence/gpt2_b5_xor_truth_table_a100_v4/) | One backbone with partitioned adapters; 00/01 inherited, 10/11 continued with expanded compute caps and unchanged endpoint gates. Runtime supplies routing, XOR and numerical writes; independent B5 confirmation pending. |
| Frozen local writing (Line C / LC1) | [C35–C37 execution logs](docs/LINE_C_LOCAL_WRITING.md): fixed-geometry and learned-calibration composites both pass 16/16, 32/32 and 32/32; raw core passes 3/16, 5/32 and 7/32 | Related same-model/chart development cohorts. Fixed geometry also succeeds, so learned calibration necessity or superiority is not established. Full result ZIP and frozen-weight audit pending. |
| LC2 development and audit | [C52/C54 original compact records](evidence/gpt2_lc2_c52_c54_c55/): both 32/36 on historical seen-class states; C54 remains `C54_DEVELOPMENT_FAILURE`. C55 has a completed read-only audit record retaining this decision | Full local ZIP hashes and compact record consistency checked; full public attachments pending. No excluded-class evaluation or independent model replay; LC2 remains open. |
| Later LC2 development | [C141/C142/C148/C150 compact archive-backed evidence; C154/C155 logs](docs/LINE_C_C141_C155.md): C140 42/48 in each seen-class confirmation round; C148 79/84 in both training arms; C150 diagonal/cyclic 79/84 and 80/84 | Failures retained; later 84 states are exposed training. C154/C155 are numerical diagnostics at log tier, not proof of AD error. No excluded-input acceptance; **LC2 open**. |
| Transfer and internalisation (LC2–LC6) | [Separately declared validation targets](docs/LINE_C_VALIDATION_TARGETS.md) | Excluded-request generalisation, scale, cross-model transfer, continual writing and native updates remain open. |
| Qwen / geometry / optimization | [Separate branches and evidence index](docs/RESPONSE_FIBRE_FOUNDATIONS.md) · [Qwen status](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#qwen-status-and-missing-version-evidence) | GPT-2 results do not change these branches' claims. |

For Line C, the final proposal uses prewrite information; online writing has no
postwrite correction or backtracking. Geometry preparation remains offline,
requested output bits come from the runtime, and parameter commitment is external.
C35–C37 evidence is log-derived, not an independent model or weight audit.
The [matrix](docs/PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md) separates this tier from
archive consistency checks and lists failures, costs and missing evidence.
These results do not establish arbitrary program execution, native self-updates,
lifelong learning or whole-model zero forgetting.

**Reading the labels:** B0–B5 name the memory capability roadmap; LC1–LC6 name
separate Line C validation goals, not a renumbering of L1–L5 or B4/B5; C-number labels identify experiments, not completed
capabilities. V labels identify protocol versions. Legacy L1–L5 and R-series
labels belong to earlier evidence and optimization records; consult the
[evidence index](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#evidence-index-and-version-history)
for their specific scope.

## Verify existing evidence without a GPU

From the repository root, with Python 3 and no model download. The C141–C155
checker requires Python 3.10+ and NumPy and performs only archive arithmetic:

```bash
python3 evidence/gpt2_line_c_c35_c37/verify_reported_evidence.py
python3 evidence/gpt2_lc2_c52_c54_c55/verify_records.py
python3 evidence/gpt2_lc2_c141_c155/verify_records.py
python3 evidence/gpt2_b5_xor_continuation_a100_v5/verify_archive.py
python3 evidence/gpt2_b5_xor_truth_table_a100_v4/verify_archive.py
python3 evidence/gpt2_b5_single_backbone_a100_v3/verify_archive.py
python3 evidence/gpt2_b5_trained_dispatch_a100_v2/verify_archive.py
python3 evidence/gpt2_b4_minimal_causal_a100_v1/verify_archive.py
```

C141–C155 checks the delivered compact members, included tensors, reconstructed
proposals and gates plus two log summaries. Four full-result audits are recorded
as performed during packaging; this checkout does not contain or rehash those
full ZIPs. Verifier completion does not mean LC2 completion.

The C35–C37 check verifies source identities and log-derived records; the
C52/C54/C55 check verifies compact hashes, parent identities and recorded gates; the B4/B5
checks verify archive hashes and recorded consistency. These read-only checks do
not re-execute GPT-2, inspect absent Line C weights or independently reproduce model readouts. See the
[verification guide and known legacy blocker](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#verify-archives-without-a-gpu).

## Run an actual A100 experiment

For the currently archived Line C protocols, start with the
[C35–C37 launchers and input guide](external_tests/gpt2_line_c_c35_c37/).
Upload the chosen PY to an A100 Colab and run `%run /content/<filename>.py`.

| Protocol | Required parent for a new run | Resume input |
| --- | --- | --- |
| C35 | Completed compatible C34 result ZIP | Matching C35 result ZIP |
| C36 FIX1 | Completed passing C35 result ZIP | Matching C36 result ZIP |
| C37 | Completed passing C36 FIX1 result ZIP with pinned freeze identity | Matching C37 result ZIP |

**The full Line C parent results and checkpoints are not included here.**
The source/log package alone cannot reproduce these experiments. Use actual
result ZIPs, not source, launcher or GitHub update ZIPs. A completed archive
resumed or recounted is not a fresh evaluation. C38–C41 local result summaries are discussed on the
[status page](docs/LINE_C_LOCAL_WRITING.md#research-frontier-and-subsequent-local-records);
their full archives and reproduction entry points are not published here.
C52/C54/C55 compact records likewise do not replace their complete result ZIPs
for replay. See the [missing-materials list](docs/PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md#reproduction-and-missing-materials).

For the operational foundation, the
[B5 V5 launcher](evidence/gpt2_b5_xor_continuation_a100_v5/GPT2_B5_XOR_CONTINUATION_A100_V5.py)
accepts the [failed V4 result](evidence/gpt2_b5_xor_truth_table_a100_v4/gpt2_b5_xor_truth_table_a100_results.zip)
for continuation, or a matching V5 result for resume. The
[topic's A100 guide](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#a100-experiments-and-input-files)
retains B4, B5 V2 and V3 entry points and detailed input protocols. Actual model
runs require the GPU environment and are distinct from the read-only checks above.

## Mechanism, roadmap and evidence

For the declared protected response map, its Jacobian is \(J\); \(Jv=0\)
expresses first-order preservation. If \(N\) spans the permitted response kernel
and \(H\) is the readout map, \(C=DH\,N\) describes local readout controllability
within those directions. Finite updates remain limited by curvature, numerical
approximation, response/KL budgets and parameter constraints.

- [Mathematical scope and B0–B5 / Line C roadmap](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#mathematical-mechanism).
- [Parametric-memory evidence index](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#evidence-index-and-version-history), with original summaries, protocols and historical limitations.
- [Response-fibre geometry, Pareto and optimization foundations](docs/RESPONSE_FIBRE_FOUNDATIONS.md) and [claim boundaries](docs/CLAIM_BOUNDARIES.md).
- [Qwen branch](fibre-qwen/README.md), separate from GPT-2 memory claims.
- [Previous README and full historical links](docs/README_HISTORY_2026-10-11.md).

## Papers and attribution

[A Programmable Address Space for Neural Memory](paper/A_Programmable_Address_Space_for_Neural_Memory.pdf)
reports the legacy L1–L5 stage; the B4–B5 archives are later repository evidence.
[Moving Response Fibres](paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf)
covers the geometry line. Cite the specific protocol and repository commit.
Author: **Y. Y. N. Li**. Repository and upstream model/data licenses apply independently.
