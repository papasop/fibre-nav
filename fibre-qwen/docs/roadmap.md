# Roadmap

1. Freeze Qwen3-8B free generation, deterministic router, rule cards, v2 evaluator, thresholds, and at least 20 untouched R21d prompts in Git before execution.
2. Record the R21d prompt-set SHA-256 and repository commit before opening any model output.
3. Run R21d against bare Qwen3-8B without protocol changes and preserve pass or failure unchanged.
4. Add blinded human adjudication because v2 was calibrated post hoc on R21c.
5. Collect 200+ explicit human preferences before another weight update.
6. Only after robust confirmation, package an inference service and versioned profile updater.
