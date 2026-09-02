# Roadmap

1. Seal the 40 completed R21d-r1 pairwise judgments before decoding arm identity.
2. Run `finalize_human_review.py`; preserve pass or failure unchanged and report single-reviewer evidence separately from automatic scores.
3. If the single-reviewer candidate passes, obtain a second independent blinded review and report agreement before nominating an R21e protocol.
4. Repair router coverage prospectively, then freeze entirely new prompts; do not regenerate the R21d items under repaired routing.
5. Collect 200+ explicit human preferences before another weight update.
6. Only after robust confirmation, package an inference service and versioned profile updater.
