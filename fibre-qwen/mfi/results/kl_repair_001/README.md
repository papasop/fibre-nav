# KL-only candidate 001: complete, aggregate failed

Frozen commit: a5fcb8b2aae27f5ae44166a9b8ca9857fedaf072.
Both algorithms ran all three development seeds and three frozen new seeds.
Original development programs: 1/3; KL-only: 2/3. MFI task score: 11/48 -> 30/48.
Original new-seed programs: 1/3; KL-only: 2/3. MFI task score: 19/48 -> 31/48.
Paired no-regression and failed-operation rollback gates passed; all-programs gate failed.
The known failed WRITEs improved, but response drift still stopped some OVERWRITEs.

All raw operations, task predictions, source hashes and protocol are retained.
Second-candidate design used development failure observations while this trial
was running. Candidate 2 uses a separate new-seed cohort. No candidate-1 settings
were changed during its execution. Total runtime 905.68 seconds includes task
scoring and both algorithms; trials partly overlapped on the same CPU host, so
this is not a comparative speed benchmark. Exit code 2 is a scientific gate failure.
