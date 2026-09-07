# L5 programmable address operations — v2.2.4

This directory archives a prospectively frozen fresh-seed and fresh-expression single-seed confirmation candidate for programmable address operations in one pretrained GPT-2 restricted LoRA chart.

## Frozen program

One shared parameter state executes:

```text
WRITE [0,0,1,1]
OVERWRITE address 01 -> [0,1,1,1]
SWAP addresses 00/11 -> [1,1,1,0]
MOVE to a distinct finite-response level set
RETURN to the source response set
```

All 16 declared gates passed in fresh seed `82931`. Exact and fresh held-out-expression access were `1.0` after every operation. Minimum held-out signed margins were `0.615448`, `3.759979` and `1.122581` after WRITE, OVERWRITE and SWAP. They remained `1.112556` after MOVE and `1.122551` after RETURN.

The requested OVERWRITE scored `1.0` exact/held-out versus `0.5` for the wrong operation and `0.75` for the matched-norm random-kernel control. The requested SWAP scored `1.0` versus `0.5` for both corresponding controls. Maximum recorded response drift was approximately `1.526e-5`; maximum recorded anchor KL was `0.003993`, below the frozen `0.005` gate.

## Interpretation

The result prospectively supports, in one seed and one GPT-2 LoRA chart, a shared four-cell state with externally specified write, selective overwrite, content swap and local transport operations. The base GPT-2 weights remain frozen while restricted LoRA coordinates are updated through response-constrained projected-gradient and retraction procedures.

It does **not** establish autonomous interpretation of an instruction language, gradient-free operations, concurrent concept-level storage beyond the declared binary cells, multi-seed confirmation, cross-model replication or global transport.
