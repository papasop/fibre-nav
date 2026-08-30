# GPT-2 Boundary Status

GPTW v1.1.0 is archived under `evidence/gptw_response_fibre_v1_1_0/`.
The v1.1.1 documentation addendum calibrates the v2/v3 scaling interpretation
without changing frozen results or gates.

Current status:

- prospective adaptive value is confirmed in the declared rank-2 LoRA-B
  subspace of GPT-2 attention layers 10 and 11 (8/8 new seeds);
- current-fibre versus source-fixed finite-radius separation is SUPPORTED in
  6/6 codeword instances and 8/8 disjoint natural-English instances (18/18 and
  32/32 interior nodes, respectively);
- the 2-versus-1 exponent split is analytically forced by the current-kernel
  and source-fixed arm definitions and retained as a numerical-correctness
  check, not an independent discovery;
- the substantive v2 evidence is the finest-radius fixed/current
  correction-cost ratio 8.60-25.70, principal angles 0.0161-0.0713 rad,
  active-J residual amplification about 1.4e4-1.4e5 and numerical-precision
  path-response error;
- the substantive v3 natural-text evidence is the finest-radius fixed/current
  correction-cost ratio 8.87-35.56, active-J residual amplification
  1.41e7-1.12e8, maximum path-response error about 1.08e-15, and the
  restriction that the result is not confined to the original codeword prompts;
- the formally unsuccessful initial audit and its precision repair are
  preserved rather than overwritten;
- GPT-2 does not support capacity-weighted CNER, Moving-Fibre F16 ordering,
  local/global variational minimality, full-model transfer, semantic
  invariance, or downstream-task improvement.
