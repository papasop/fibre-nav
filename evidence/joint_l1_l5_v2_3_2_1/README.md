# Joint L1--L5 development smoke v2.3.2.1

Status: **PASS -- post-failure development evidence**.

- Protocol: `MOVING_FIBRE_INTELLIGENCE_L3_REASSIGN_MARGIN_POLISH_V2_3_2_1`
- Seed: `83001`
- `all_layer_smoke_gates_pass`: `true`
- `all_gates_pass`: `true`
- Runtime: `1266.5119502544403` seconds
- Config SHA-256: `eb6a76903e8ed93188af87ce464bc6e82c405694342824d010e24ed629f73ef2`

The minimum held-out margins were 0.5478 after WRITE, 0.6141 after REASSIGN,
1.4553 after OVERWRITE and 6.1988 after SWAP. MOVE and RETURN preserved exact
and held-out access to all four addresses.

See `CLAIM_BOUNDARY.md` before citing this result.
