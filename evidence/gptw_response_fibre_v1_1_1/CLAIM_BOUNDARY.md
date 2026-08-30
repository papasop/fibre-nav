# GPTW v1.1.1 Claim Boundary

This documentation revision preserves the v1.1.0 evidence boundary and clarifies
the interpretation of the v2 current-versus-source-fixed scaling slopes.

## Supported

Within the frozen GPT-2 rank-2 LoRA-B domain over layers `[10, 11]`, the v2
codeword audit supports a finite local current-fibre realizability advantage
over source-fixed replay along constructed response-retracted paths.

The empirical v2 support is:

- 6/6 seeds and 18/18 interior nodes passed the frozen gates;
- finest-radius fixed/current correction-cost ratios were 8.60-25.70;
- principal angles between current and source-fixed directions were
  0.0161-0.0713 rad;
- maximum path-response error was approximately 6.7e-16, below the frozen
  `2e-4` gate.

The observed v2 slope ranges, `alpha_current = 1.9986-2.0084` and
`alpha_fixed = 1.0849-1.2151`, and the slope gap are retained as
numerical-correctness checks. They follow from the Taylor expansion and arm
construction:

```text
R(theta + h v) - R(theta) = h J v + (h^2 / 2) H[v, v] + O(h^3)
J v_current approx 0  =>  e(h) = O(h^2)
J v_fixed   != 0      =>  e(h) = O(h)
```

They should not be presented as an independent discovery.

## Natural-Text Cohort

The v3 natural-text audit is a separate frozen cohort from v2. It is already
part of the paper-facing claim set and reports 8/8 seeds and 32/32 nodes. It
supports that the separation is not confined to the original codeword prompts,
with finest-radius cost ratio 8.87-35.56, active-J residual amplification
1.41e7-1.12e8 and maximum path-response error about 1.08e-15. Its alpha and
cost-ratio ranges must be reported separately from the v2 codeword cohort.

## Not Claimed

This evidence does not establish:

- full-parameter GPT-2 or unrestricted LLM transfer;
- semantic invariance beyond the declared frozen prompt cohorts;
- arbitrary LoRA ranks, layers or response definitions;
- ordinary optimizer behavior;
- arbitrary-path or global variational minimality;
- a complete response-kernel bundle;
- PASQAL Cloud, historical robustness-output, waveform modulation,
  phase-jump delay, dissipation, detuning, interaction, EMU or QPU evidence.
