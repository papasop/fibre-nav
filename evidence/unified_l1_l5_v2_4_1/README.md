# Unified prospective L1–L5 audit — v2.4.1-r1

This archive records the first prospectively frozen three-seed unified audit.
Each seed executes one continuous shared-state sequence spanning WRITE, L3
REASSIGN, selective OVERWRITE, SWAP, MOVE and RETURN.

All three seeds completed. Exact operations, selectivity, response/KL budgets,
operation controls and the L5 program criteria passed in all three seeds, but
strict held-out-expression access did not: zero of three seeds passed every
gate. Layer pass counts were L1 3/3, L2 0/3, L3 1/3, L4 1/3 and L5 3/3.

This is a scientifically useful negative result. It localizes the principal
failure to held-out language access rather than to exact address operations or
the declared response and KL constraints. The L4 component checks access and
response eligibility during MOVE/RETURN; it does not replace the separate
eight-endpoint, 28-relation geometric audit.

Run `python verify_snapshot.py` to verify the archive and its frozen summary.
