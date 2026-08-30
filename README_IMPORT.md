# GitHub ingestion bundle: low-response Pareto evidence

Target repository: `papasop/neural-fibre-geometry`

This archive is laid out relative to the repository root. Copy its contents into
a clean checkout, review the diff, run the verification commands below, and
commit the added files. The bundle does not modify or delete existing evidence.

## Included evidence

- `evidence/low_response_pareto_v1/resnet_v4_6/`
  - reduced ResNet-18/CIFAR-10 CPU protocol and launcher;
  - eight prospective seed records and node-level metrics;
  - decision: `LOW_RESPONSE_PARETO_ADVANTAGE_SUPPORTED` (7/8 seeds).
- `evidence/low_response_pareto_v1/gpt2_lora_b_v1/`
  - GPT-2 rank-2 native-LoRA-B CPU protocol and launcher;
  - eight prospective seed records and node-level metrics;
  - decision: `GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED` (8/8 seeds,
    24/24 noninitial nodes).
- `paper/Moving_Response_Fibres_v5_unified_hierarchy.pdf`
  - manuscript PDF that distinguishes the 24-node Pareto audit from the
    separately frozen 32-node GPTW-v3 correction-cost audit.

## Verification

From the repository root:

```bash
sha256sum -c MANIFEST_LOW_RESPONSE_PARETO_V1.sha256
python verify_low_response_snapshot.py
python -m py_compile \
  evidence/low_response_pareto_v1/resnet_v4_6/cner_v4_6_cpu_pareto_frontier.py \
  evidence/low_response_pareto_v1/resnet_v4_6/COLAB_LAUNCHER_V4_6_CPU.py \
  evidence/low_response_pareto_v1/gpt2_lora_b_v1/gptw_lora_pareto_cpu_8seed.py \
  evidence/low_response_pareto_v1/gpt2_lora_b_v1/COLAB_LAUNCHER_GPTW_LORA_8SEED_CPU.py
```

The full reruns download public model/data dependencies and are intentionally
not executed during ingestion. Their commands are documented in the two
evidence READMEs.

## Required repository text update

After this bundle is committed, update the manuscript/Data and Code statement
to cite the new commit hash. Only then may the two Pareto packages be described
as publicly archived. Do not retroactively claim that they were present in
commit `236f646c472018a7e38be11fd658519763bc2346`.

Suggested commit title:

```text
Archive prospective low-response Pareto audits for ResNet and GPT-2
```
