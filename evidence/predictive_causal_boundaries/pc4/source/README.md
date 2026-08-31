# GPTW-PC4 paired causal capacity audit

PC1-PC3 remain negative. PC4 asks a different, causal question: among locally
response-matched GPT-2/LoRA-B states, does a state selected for higher frozen
capacity learn more under the same subsequent AdamW budget than a state selected
for lower capacity?

The selection prompts, branch-training prompts and final-evaluation prompts are
disjoint. Results are blocked by seed. Random candidate pairing and response-cost
matching are frozen controls. This is a CPU protocol with a two-hour hard limit.

In Colab, upload and run `COLAB_LAUNCHER_GPTW_PC4_CPU.py`, then upload this ZIP.

The launcher downloads the result ZIP automatically. A positive result does not
erase or reinterpret the negative PC1-PC3 decisions.
