#!/usr/bin/env python3
"""One-click verifier/launcher for the GPT-2 LoRA Picard r5 audit."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--outdir", default="picard_gpt2_lora_r5_results")
    parser.add_argument("--device", default="cuda")
    args, passthrough = parser.parse_known_args()

    root = Path(args.source_root).resolve()
    run([sys.executable, "verify_picard_gpt2_lora_v1_6.py"], root)

    if args.verify_only and not args.run:
        return
    if not args.run:
        print("Verification complete. Pass --run to execute the r5 audit.", flush=True)
        return

    script = root / "evidence" / "audits" / "picard_gpt2_lora_r5_ten_step" / "code" / "gpt2_lora_picard_confirm_v0_2_6_r5.py"
    run(
        [
            sys.executable,
            str(script),
            "--outdir",
            args.outdir,
            "--device",
            args.device,
            *passthrough,
        ],
        root,
    )


if __name__ == "__main__":
    main()
