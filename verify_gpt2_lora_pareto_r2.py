#!/usr/bin/env python3
"""Verify the imported R2 snapshot without model execution."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"evidence/low_response_pareto_v1/gpt2_lora_b_r2_strict"

def fail(message):raise SystemExit("VERIFY FAILED: "+message)

manifest=ROOT/"MANIFEST_GPT2_LORA_PARETO_R2.sha256"
for raw in manifest.read_text().splitlines():
    if not raw.strip():continue
    expected,relative=raw.split(None,1);path=ROOT/relative.strip()
    if not path.is_file():fail(f"missing {relative}")
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:fail(f"hash mismatch {relative}")

report=json.loads((BASE/"results/report.json").read_text())
protocol=json.loads((BASE/"source/protocol.json").read_text())
result_protocol=json.loads((BASE/"results/protocol.json").read_text())
if protocol!=result_protocol:fail("source and result protocols differ")
if report["protocol"]!=protocol["protocol"]:fail("report protocol mismatch")
if report["decision"]!="GPT2_LORA_LOW_RESPONSE_PARETO_STRICT_CONFIRMED":fail("unexpected decision")
if report["completed_seeds"]!=8 or report["supporting_seeds"]!=6:fail("cohort counts")
primary=report["primary_random_current_control"]
if primary["positive_seeds"]!=7:fail("positive seed count")
lo,hi=primary["bootstrap_mean_95ci"]
if not (lo>0 and hi>lo):fail("bootstrap interval")
if primary["exact_two_sided_sign_flip_p"]!=0.03125:fail("sign-flip p")
if not all(report["cohort_gates"].values()):fail("a frozen cohort gate failed")

rows=list(csv.DictReader((BASE/"results/node_metrics.csv").open()))
if len(rows)!=32:fail("expected 32 total nodes")
primary_rows=[r for r in rows if int(r["step"])>0]
if len(primary_rows)!=24:fail("expected 24 noninitial nodes")
if sum(float(r["current_minus_best_of_16_random_current_kernel_auc"])>0 for r in primary_rows)!=20:
    fail("same-kernel node win count")
if any(int(r["far_step"])==0 for r in primary_rows):fail("far-time duplicates source")

seed_files=list((BASE/"results").glob("seed_*.json"))
if len(seed_files)!=8:fail("expected eight seed records")
print("VERIFIED: GPT-2/native-LoRA-B Pareto R2 strict snapshot")
