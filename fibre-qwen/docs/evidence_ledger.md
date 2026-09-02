# Evidence ledger

| Stage | Question | Result | Status |
|---|---|---|---|
| R20c | Can 14 conversation-derived preferences train a LoRA? | Loss 6.2768 → 3.3214 | Training completed; nonconfirmatory |
| R20d | Does R20c improve three frozen preference margins? | 2/3 improved, but severe repetition | Fail-closed after generation audit |
| R20c-r1 | Can weaker LoRA plus replay stabilize training? | Loss 5.3637 → 4.1869 | Training completed |
| R20d-r1 | Is stable LoRA healthy and preferred? | All three margins declined; health 1/3 | Fail-closed |
| R21a | Does a full research constitution work? | margin 6/10 improved; semantic 2/10 | Fail-closed |
| R21b | Does targeted retrieval reduce interference? | margin 7/10 improved; declared 8/10; semantic 5/10; health 7/10 | Mechanism signal; fail-closed |
| R21b-r1 | Can one-card structured output restore health? | margin 8/10 improved; declared 8/10; health 9/10; semantic 5/10 | Stability repaired; semantic gate failed |
| R21b-r2 | Is the remaining error caused by retrieval or generation? | Protocol frozen; no result in this archive | Same-item mechanism diagnostic |
