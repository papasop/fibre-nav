# GPT-2 B4 minimal causal XOR development record

Result: **B4_MINIMAL_DEVELOPMENT_PASS**, one development seed (82931), two XOR input states. This archive supports a minimal externally controlled read–compute–write demonstration; full B4 independent confirmation remains open.

Four fixed-access shared LoRA-B cells store input A, input B, an unselected sentinel and output. Base GPT-2 and LoRA-A remain frozen. Actual model forwards read the inputs; an external deterministic controller computes XOR; constrained optimisation writes the result into the output cell.

| Phase | Target / observed bits | Min exact margin | Min development-expression margin | Averaged response drift | Anchor KL |
|---|---|---:|---:|---:|---:|
| prepare | `[0, 0, 1, 1]` | 0.576408 | 0.608414 | 7.6293945e-06 | 0.003734991 |
| normal_writeback | `[0, 0, 1, 0]` | 3.654167 | 3.594391 | 1.5258789e-05 | 0.0034965794 |
| substitute_input | `[1, 0, 1, 1]` | 2.520126 | 2.292549 | 1.5258789e-05 | 0.0041040322 |
| substituted_writeback | `[1, 0, 1, 1]` | 10.140190 | 10.771202 | 0 | 0.0025096438 |

## Scope and control limitations

- Inputs tested: 00 and 10. The full four-row XOR truth table was not tested.
- The normal branch changes output 1→0. In the substituted branch output is already 1 before writeback; it does not demonstrate a required 0→1 output transition.
- Input substitution changes the controller result 0→1. Read blocking substitutes fixed inputs (0,0) and checks the controller result; it does not execute a separate blocked-read writeback.
- The no-write control is a before/after output comparison.
- `reset_task_sensitivity` is a weak check: comparison with either of two opposite expected outputs makes its Boolean pass automatic. Baseline reset and saved-endpoint reload are separate recorded checks. Stronger reset causality remains to be tested.
- Previously inspected B3 expressions are development data, despite legacy `heldout` field names. There are no fresh evaluation expressions or independent seeds in this run.
- Response drift protects six selected logits averaged across 12 anchors; KL is also averaged. Neither is a universal or per-anchor preservation guarantee.
- This run includes no matched-gradient/random-update B4 comparison and makes no B5 or internally executed XOR claim.

## Reproduction and verification

Run `python verify_archive.py` here to verify the original ZIP manifest and CRC, and recompute recorded endpoint acceptance without loading a model. This checks archived consistency, not a new model evaluation.

The complete uploaded result (including four real LoRA-B endpoint tensors) is `release_assets/gpt2_b4_a100_results.zip`. Extract it to inspect source and checkpoints. `results/` contains readable machine records; `source/` is the exact archived run source.

The visible-output v1.1 launcher is in `release_assets/GPT2_B4_CAUSAL_XOR_A100_V1_1.py` and its ZIP. It preserves the experiment engine and fixes installer/child-process logging. The archived run source is preserved verbatim, including its older launcher snapshot. Use Colab A100 with CUDA PyTorch already installed. Run the launcher to rebuild from zero or restore a compatible result ZIP. This can consume paid GPU time.

Parent: legacy L5 v2.2.4 (roadmap B3), checked against repository commit `73c7ec2b803ee623eef4a7f62d275a4fdcfc2c4d`. This B4 run is not a reproduction of the full unified v2.4.3 protocol.

Next validation: all four XOR inputs, required output flips in every branch, executed read-blocking/reset controls, fresh fixed evaluation expressions and independent seeds.
