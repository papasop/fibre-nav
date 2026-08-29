#!/usr/bin/env python3
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
expected={
 "experiments/v1_r1/results/report.json":("GPT2_LORA_SELECTED_RESPONSE_FIBRE_VALUE_NOT_SUPPORTED",0,6),
 "experiments/v1_r2a_precision/results/report.json":("GPT2_LORA_DERIVATIVE_PRECISION_REPAIR_SUPPORTED",7,6),
 "experiments/v1_r3_newseed/results/report.json":("GPT2_LORA_SELECTED_RESPONSE_FIBRE_VALUE_PROSPECTIVELY_CONFIRMED",8,6),
 "experiments/v2_current_vs_fixed/results/report.json":("GPT2_LORA_CURRENT_FIBRE_REALIZABILITY_ADVANTAGE_SUPPORTED",6,4),
 "experiments/v3_natural_text/results/report.json":("NATURAL_TEXT_CURRENT_FIBRE_ADVANTAGE_SUPPORTED",8,6),
}
for rel,(decision,supporting,required) in expected.items():
    row=json.loads((ROOT/rel).read_text())
    assert row["decision"]==decision,(rel,row["decision"])
    assert row["supporting_instances"]==supporting,(rel,row["supporting_instances"])
    assert row["required_instances"]==required,(rel,row["required_instances"])
v2=json.loads((ROOT/"experiments/v2_current_vs_fixed/results/report.json").read_text())
assert all(x["passing_nodes"]==3 and x["instance_supported"] for x in v2["instance_summaries"])
v3=json.loads((ROOT/"experiments/v3_natural_text/results/report.json").read_text())
assert v3["total_interior_nodes"]==32
assert all(x["passing_nodes"]==4 and x["instance_supported"] for x in v3["instance_summaries"])
assert max(x["maximum_path_response_error"] for x in v3["instance_summaries"])<=1e-10
print("GPTW v1.1.0 evidence decisions, cohort counts and V3 path gates verified")
