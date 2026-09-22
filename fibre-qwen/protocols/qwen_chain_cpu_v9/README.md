# Qwen V9 external two-step CPU protocol

**Audited result: 112/112 slots and 32/32 program endpoints passed the frozen V9
gates.** See [raw evidence, reproducible audit and limitations](../../evidence/qwen_cpu_chain_v9/README.md).

This directory preserves the originally delivered `QWEN_V9_TWO_STEP.py`,
`qwen_chain_cpu_v9.zip`, and all manifest-listed source bytes. The frozen
`source/VALIDATION.md` describes checks before delivery; its statement that no
pretrained V9 run had yet occurred refers to that earlier engineering stage.
The subsequently supplied model run is documented separately in the evidence link.

Run the launcher in Colab CPU and upload the adjacent source ZIP. To resume,
also upload one latest V9 results ZIP. See [the original instructions](source/README.md).

Source ZIP SHA-256:
`d983ecf996f9ef537c47418354758924e891c205f39f742644af9d496e0af8fc`.
Canonical frozen-source manifest SHA-256:
`9fdfc82381f988451d3b84695a4f3703191deb44f95d9faadd7834ac5c37a038`.

Engineering tests from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s fibre-qwen/protocols/qwen_chain_cpu_v9/source -p 'test_*.py' -v
```

These tests use mock observations. They do not replace the archived pretrained
model evaluation and do not establish parameter-memory or native planning.
