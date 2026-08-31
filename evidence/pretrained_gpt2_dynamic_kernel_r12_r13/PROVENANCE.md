# Provenance

Original returned result ZIPs are preserved for R12a, R12a-r1, R12a-r2 and
R13. Their extracted JSON files are included without modification.

The original `picard_r12b_results.zip` was not supplied to this archive build.
Instead, `results/r12b_recovered/original_complete_colab_log.md` is the complete
returned Colab log supplied by the experimenter. It contains the complete JSON
`run_summary`, including all 21 full arm records. `run_summary.json` and the 21
individual JSON files in that directory were mechanically and losslessly
recovered from that embedded JSON object. They are therefore derived archive
files, not byte-identical copies of the absent original ZIP members.

This distinction does not affect the reported R12b aggregate values, but it
must remain disclosed. If the original R12b ZIP becomes available, add it as
`results/r12b_recovered/original_results.zip`; do not silently replace or
delete this provenance statement.
