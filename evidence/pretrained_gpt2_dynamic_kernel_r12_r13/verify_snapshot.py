#!/usr/bin/env python3
"""Fail-closed verifier for the R12-R13 pretrained GPT-2 evidence snapshot."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(path): return json.loads((ROOT/path).read_text())

def verify_manifest():
    lines=(ROOT/"MANIFEST.sha256").read_text().splitlines()
    for line in lines:
        expected,rel=line.split("  ",1); p=ROOT/rel
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=expected: raise RuntimeError(f"checksum mismatch: {rel}")

def main():
    verify_manifest()
    r12b=load("results/r12b_recovered/run_summary.json")
    r13=load("results/r13/extracted/run_summary.json")
    assert r12b["scientific_status"]=="R12B_MULTISEED_DEVELOPMENT_CANDIDATE_SUPPORTED"
    assert r12b["candidate_for_untouched_seed_confirmation"] is True
    assert len(r12b["records"])==21 and all(r12b["gates"].values())
    assert r13["scientific_status"]=="R13_PRETRAINED_GPT2_CURRENT_KERNEL_BUDGETED_DUAL_ADVANTAGE_CONFIRMED"
    assert r13["seeds"]==[33211,33217,33229,33241,33253]
    assert r13["budgets"]==[2e-5,5e-5]
    assert r13["supporting_budget_count"]==2 and all(r13["gates"].values())
    assert len(r13["records"])==35
    assert all(p["current_beats_both"] for p in r13["seed_pairs"])
    assert len(r13["seed_pairs"])==10
    assert all(q["dual_win_seeds"]==5 and q["supports_frozen_confirmation_gate"] for q in r13["budget_results"])
    print("PRETRAINED_GPT2_DYNAMIC_KERNEL_R12_R13_SNAPSHOT_VERIFIED")
    return 0

if __name__=="__main__": raise SystemExit(main())
