# C141 R1: test-phase timeout repair

The supplied Colab log completed 23 synthetic tests and stopped at the fixed 900-second test-phase budget before CPU dependency checks or the real GPT-2 worker. Its failure result remains valid as an incomplete launch: SHA256 adacec60f382363405649c022a82a4a4cbef5087472f565cf3472a36a34fedcf. No candidate acceptance outcome can be inferred.

R1 changes only the launcher test-phase budget to 2400 seconds, preserving every test and the real-worker budget of 3000 seconds. Scientific worker, runtime, core, audit, fixture and test sources, vendor sources, input pins and protocol are byte-identical to the original delivery. Only launch timing and delivery metadata/documentation differ. New bundle and launcher hashes are pinned together.

Run the R1 launcher with its R1 bundle. Existing C132-C140 scientific archives remain the inputs. This is the same C141 confirmation design, not a new experiment, checkpoint choice or acceptance threshold. Test and worker budgets are separate; dependencies add time. Full execution may exceed one hour.

The full 29-test synthetic suite was rerun locally. Real GPT-2 and the nine original scientific archives were unavailable here and were not replayed before delivery.
