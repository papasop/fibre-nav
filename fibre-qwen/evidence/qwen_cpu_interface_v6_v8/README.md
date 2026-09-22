# Qwen CPU interface V6–V8 evidence snapshot

> **Latest V8 evidence: 103/128, PAUSED — incomplete.** The [raw checkpoint audit](v8/checkpoint_103/README.md) verifies 103/103 correct outputs, 24/24 invalid-input rejections and 39/39 exact available repeats. Twenty-five points remain. The earlier 18-point statements below describe the initial import snapshot; they are retained as history. No final pass is claimed.

This is a development archive of frozen Qwen3-1.7B inference tests. It is separate
from the repository's GPT-2 parameter-address experiments and Qwen3-0.6B R23
restricted-chart optimization results.

| Protocol | Evidence held here | Observed status |
|---|---|---|
| V6 operation rule | Original complete results ZIP and session output | 64/64, original overall gate **false** |
| V7 expression confirmation | Complete session log only | 96/96 reported, original overall gate **false**; raw results not audited |
| V8 compiled interface | Frozen executable package and partial session log | 18/128 observed, final pass **unknown** |

V6's recorded selected-rule primary and bounded counts are 16/16, while strict
free output is 11/16. Its original cross-path numerical gate also fails. These
counts do not override its failed formal decision. The complete result ZIP is
authoritative and includes its original frozen source, records and summary.

V7's visible first-pass bounded counts are historical 16/16, fresh_a 15/16 and
fresh_b 16/16. The fresh_a `00-OR` output is 1 instead of 0, also in the repeated
log. The session reports `FRESH_EXPRESSION_OR_REGRESSION_FAILURE`. A text log does
not provide the full scores or numerical records needed for a raw-result audit.

V8's visible canonical 16/16 and first two sentence points have correct bounded
outputs and matching parsed fields. The log stops at point 18; it supplies no
final summary, full repeated run, or raw numerical results. This is the latest
supplied snapshot, not a claim about the runtime's current live state.

`STATUS.json` is derived by `verify_archive.py`; it is not an original model output.
The verifier checks supplied hashes and recomputes visible-log Boolean counts.
It preserves V6's original summary values and does not rerun the pretrained model.
All imported payload files are indexed by repository-relative paths in
`MANIFEST.sha256`; the manifest itself and `IMPORT_VALIDATION.md` are covered by
the repository provenance checksum index. Renamed V7/V8 logs retain their bytes.

The source was frozen for execution before these supplied results; this GitHub
import is retrospective and is not a public preregistration timestamp. No trained
model, native Qwen memory capability, general expression understanding, or release
readiness follows from this archive. To complete the record, add original V7 and
final V8 results ZIPs, verify their payload hashes and frozen gates, then append
the findings without replacing these earlier observations.
