# R17 development chain

- R17a: Euclidean constrained Onsager satisfied the exact KKT condition but
  lost to projected AdamW.
- R17b: the Adam diagonal metric closed roughly 94% of the R17a loss gap and
  beat Euclidean and source-frozen controls, but still narrowly lost to
  projected AdamW.
- R17c: same-seed budget calibration selected multiplier 1.15.  This stage is
  development only and cannot confirm its own selected candidate.

R17a and R17b retain original result ZIPs.  For R17c, the available Colab log
is preserved verbatim and `run_summary.recovered_from_log.json` is a mechanical
JSON extraction from that log; no raw R17c result ZIP was supplied.
