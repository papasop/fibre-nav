# GPT-2 B4 minimal causal A100 V1

Single development seed82931, fixed input01, fixed four-cell access interface.
Derived from the V1 writer and the executed-control implementation in V2.
Fresh protocol; old V1/V2/quick result ZIPs cannot resume this experiment.

## Frozen sequence
1. Zero LoRA-B -> prepare [0,1,1,0], deliberately wrong XOR output (600 steps).
2. Read actual A/B, external XOR -> [0,1,1,1], output must flip (480).
3. Branch from normal endpoint: block original reader, substitute A=1,B=1,
   external XOR -> [0,1,1,0], actual output must flip (480).
4. Separate branch from normal endpoint: change stored A only -> [1,1,1,1] (480).
5. Read actual substituted A/B -> [1,1,1,0], output must flip (480).
6. Reload all endpoints, zero actual B and compute from reset reads; restore
   normal endpoint and require original contents and XOR result to recover.

No-write controls use the actually measured pre-write output in steps2/5.
All five endpoints require exact and original development paraphrase accuracy1,
minimum target-relative margins >=0.05, averaged protected response drift <=0.002,
mean anchor KL <=0.005. Checkpoint selection and budgets use the unchanged parent
writer. Every stage runs its full fixed step allocation, maximum2520 steps.
Training every5 steps: atomic result ZIP, CRC and member hashes, verified Drive
copy; only latest3 verified backups of THIS protocol retained. No wall-clock cap.

The reused V2 expression family is evaluated after training as auxiliary
DEVELOPMENT diagnostics, never as new held-out confirmation or acceptance gates.
Passing supports only one-input, single-seed externally controlled causal
read-compute-write development evidence. It establishes neither full B4 nor B5,
geometry superiority, independent replication or uniform response preservation.

Estimated A100 time25–35min; not an actual measured runtime for this variant.
Local verification covers synthetic workflow, frozen protocol, resume, archive
integrity and CPU writer continuation; real GPT-2/A100 execution is pending.
