# Fresh-process reload verification

QWEN_MFI_L1_RELOAD_001 passed all 7 gates in 6.60 seconds on CPU.
Reloaded the original l1_development_001 final checkpoint into a freshly
constructed Qwen model. No training or gradient updates were performed.
All B tensors exactly matched the checkpoint. Readout margin, response infinity
drift and original-reference KL reproduced with absolute differences of zero.
An intentionally wrong expected value was rejected without changing the state.

Both initial and final bits happen to be 0: equality of the bit alone would not
prove restoration. The parameter equality and changed/reproduced margin distinguish
this checkpoint from the unmodified model. The saved WRITE-1 intermediate was
not available and is not tested here. This is same-seed persistence verification,
not independent confirmation, a control experiment or L2 expression access.

Sources and tolerances are recorded in protocol.json; report.json includes gates
and source hashes. To reproduce, use verify_reload.py with the archived source
directory and a new output path, as described in the parent README.
