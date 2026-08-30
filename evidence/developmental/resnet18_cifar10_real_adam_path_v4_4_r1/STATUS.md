# ResNet-18/CIFAR-10 Real AdamW Path v4.4-r1 Status

Status: DEVELOPMENTAL STAGING ONLY.

This directory archives the frozen protocol and source materials available for
the v4.4-r1 ordinary-optimizer audit. It does not import the authoritative
recovered results ZIP and does not create a repository-backed positive or
negative scientific decision.

## Scientific Question

v4.4-r1 tests whether ordinary AdamW updates, with no response projection,
response penalty or retraction, preferentially align with the current response
geometry rather than the source-frozen geometry in one frozen-backbone
ResNet-18/CIFAR-10 construction.

The primary comparison evaluates the same raw AdamW update under the current
and source Jacobians. Time-shuffled realized updates and equal-norm ambient
random directions are frozen controls.

## Known Run State

According to the run record supplied for staging:

- the A100 run completed 8 seeds;
- the run audited 128 nodes;
- the failure occurred only during result packaging;
- the reported packaging error was `NameError: name '__file__' is not defined`;
- the error occurred after per-seed computation and after `report.json` was
  written;
- the authoritative recovered results ZIP has not yet been imported into this
  repository.

The frozen engine line involved in packaging is:

```text
Path(__file__).with_name("protocol.json")
```

The engine is preserved byte-for-byte from the available source ZIP. This PR
does not patch the frozen engine to repair notebook packaging.

## Claim Boundary

Before the authoritative recovered results ZIP is imported and verified, this
directory may only be cited as a frozen-code/protocol staging record. It must
not be cited as a repository-backed positive or negative v4.4-r1 result.

In particular:

- do not reconstruct or fabricate `report.json`, per-seed JSON files or
  `node_metrics.csv` from console text;
- do not promote console-recorded positive or negative decision text into the
  formal claim boundary;
- do not move this stage to `evidence/confirmed/`;
- do not treat any recovery utility as a new protocol or new scientific run;
- preserve the original frozen engine hash if a later notebook-compatible
  packaging-only tool is added.

Console notes indicate that ordinary AdamW current-to-source and time-shuffled
separation was not supported, but because the authoritative recovered ZIP is
not present here, this repository records the state only as RESULT IMPORT
PENDING.

## Blockers

- `code/CNER_V4_4_R1_RECOVER_RESULTS.py` was not present in the current
  repository or the located source ZIP and is therefore not included.
- The authoritative recovered results ZIP is not present.
- No formal v4.4-r1 decision can be repository-backed until the recovered ZIP is
  imported and checked for exactly 8 seeds, 128 audited nodes, protocol hash
  consistency and ZIP integrity.
