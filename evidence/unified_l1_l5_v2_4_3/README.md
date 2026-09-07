# Unified L1–L5 expression-coverage audit — v2.4.3

This post-v2.4.1 repair expands the training-view coverage, freezes three new
seeds and evaluates a new held-out expression family. All three seeds completed
and two passed every gate.

Layer pass counts were L1 3/3, L2 2/3, L3 2/3, L4 3/3 and L5 3/3. Exact
address operations, preservation of unselected cells, response/KL budgets,
wrong-operation and matched-random controls, MOVE/RETURN and the declared L5
program criteria passed in all three seeds. Seed 83221 failed the strict
held-out gates during the initial WRITE and REASSIGN stages; later OVERWRITE,
SWAP and transport stages passed.

Because training-view coverage and the held-out family changed after inspecting
v2.4.1, this is not an untouched replication and cannot by itself establish
improved expression generalization relative to v2.4.1. The defensible result is
that, under the denser frozen coverage, two of three new seeds passed the full
unified protocol while exact operations and L4/L5 passed in all three.

Run `python verify_snapshot.py` to verify the archive and its frozen summary.
