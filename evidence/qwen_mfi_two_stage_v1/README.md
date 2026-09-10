# Computing from Model-Generated Readouts of Parameter Memory

**Status: bounded two-stage development result supported. Strong/single-pass L6 remains unpassed.**

A frozen Qwen3-0.6B configuration reads the known parameter-memory states `10` and `11` using a calibrated fixed prompt. A second generation call to the same model receives the first call's decoded output unchanged, followed by a fixed logical task instruction. The model correctly reports the two values and computes XOR/equality in all four known-state cases. The controller transfers text and evaluates results; it does not calculate the model's answers.

## Recorded results

| Condition | Result | Meaning |
|---|---:|---|
| Original model-generated readout, known10/11 | 4/4 | Correct first-stage read and correct second-stage values/answer |
| Flip one intermediate-text bit | 8/8 | Model follows the altered text and changes answer as prescribed |
| Flip both intermediate-text bits | 4/4 | Model reports the altered values; XOR/equality result is preserved |
| Omit intermediate text, known10/11 | 0/4 strict | Direct implicit route remains unsuccessful |
| Full planned computation rows | 30/30, none skipped | Includes zero-memory and all text-control conditions |
| Reading before/after | 18/18 records, identical paired outputs | No observed change in the audited reading behavior |
| Logged CPU response/KL, rule and single-read gates | Pass | Fixed CPU reference and original finite thresholds |
| Training updates / new memory writes | 0 / 0 | Existing serialized states loaded only |

The actual run used CPU float32, four threads: 139.72 seconds for the audit, 204.05 seconds total including setup. These are this run's recorded times, not general runtime guarantees. The original 16 package tests passed in the user's CPU runtime.

## What is closed, and what is not

**中文：本轮闭合的是“已知状态、固定接口、经模型自生成文本中介的分步读取与计算”。** 这是可保留的正面系统结果；不能改称原定单次隐式L6已经通过，也不宣称整个研究历史的首次发现。

The second prompt contains values. They originate in the model's first-stage output, but they are still externally carried text. This is a text-mediated two-call pipeline, not unmediated single-pass computation with parameter memory. There is no claim of general reasoning improvement, a new learning capability, semantic-mechanism identification, or Turing completeness.

Only two parameter snapshots (`10`, `11`) were tested; the first cell is fixed at1. Flipping intermediate text is not writing or flipping a parameter address. No independent causal role for both parameter cells, fresh-write transfer, unseen-state generalization, blind evaluation, or independent model replication is established.

Zero memory also generates a default readout (`11`) and can compute from it. Zero-memory rows have no assigned stored state; successful computation from that default is not scored as successful memory storage and must not be described as a failing arithmetic control.

The rule/reader have prior supervised calibration and the fixed interface was selected through development diagnostics. The counts above are deterministic cases in a tiny declared cohort, not independent statistical trials.

## Evidence and verification

- [Original archives and CPU launcher](artifacts/)
- [Original experiment source](source/)
- [Raw rows, reference, protocol and logs](results/)
- [Offline reviewer](verify_evidence.py)
- [Offline review report](OFFLINE_AUDIT.json)
- [Claim boundaries](CLAIMS.json)
- [Provenance](PROVENANCE.json)
- [File hashes](SHA256SUMS.json)

From the repository root:

```bash
python evidence/qwen_mfi_two_stage_v1/verify_evidence.py
python -m unittest discover -s evidence/qwen_mfi_two_stage_v1 -p test_verify_evidence.py
```

The reviewer uses Python's standard library only. It checks archive/file hashes, original source equality, stage1-to-stage2 text provenance, exact condition coverage, strict scoring, text-intervention responses, recorded budget thresholds and before/after reading/audit consistency. It does not execute the bundled experiment modules.

**An offline record audit is not an independent model execution.** It cannot independently observe the original run or recompute forward logits/KL from a model. Freeze flags and numerical audit values originate in the supplied runtime records; byte hashes establish consistency, not independent authenticity or historical timestamp certification.

The logged CPU reference is reconstructed at the original rule-on/zero-memory/no-reader parameter state before reader attachment. It is never centered on the successful reader. CPU KL is not a re-certification of the original GPU KL. This package does not re-audit all L1–L5 geometry or operations.

## Optional model rerun

Use Python3.11–3.13 on Linux x86_64/Colab, with about8GB available RAM and network access for pinned model dependencies. Run the exact original launcher in `artifacts/` with the exact original source ZIP. See [source README](source/README.md).

In Colab:

```python
%run run_qwen_self_read_compute_cpu_v1.py --threads 4 --archive "qwen_self_read_compute_cpu_v1.zip"
```

The frozen launcher expects an exact filename. If a browser added `(1)`, pass that filename explicitly via `--archive`, for example:

```python
%run run_qwen_self_read_compute_cpu_v1.py --threads 4 --archive "qwen_self_read_compute_cpu_v1 (1).zip"
```

Original source and launcher are preserved without patching to avoid changing what was tested. Full reruns are optional and may require downloads; the default evidence review does not rerun the model.

## Suggested citation sentence / reusable claim

> Within a frozen Qwen3-0.6B configuration and two known parameter-memory states, model-generated readouts supported a text-mediated two-stage pipeline for XOR and equality, with successful intermediate-text interventions and finite CPU response/KL audits. Direct implicit computation and generalization to new parameter-memory states remain unestablished.

No GitHub commit, tag, DOI, publication or new-model capability claim is assigned by this archive. Existing results, including direct implicit failures, are preserved.
