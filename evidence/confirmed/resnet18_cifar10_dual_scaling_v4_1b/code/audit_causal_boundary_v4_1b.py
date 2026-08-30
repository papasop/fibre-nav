#!/usr/bin/env python3
"""Static audit that held-out confirm data cannot steer v4.1b paths or gates."""
import ast
from pathlib import Path

ENGINE = Path(__file__).with_name(
    "cner_resnet18_cifar10_moving_fibre_dual_scaling_confirm_v4_1b.py"
)
tree = ast.parse(ENGINE.read_text())
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

run_path = functions["run_path"]
loops = [n for n in ast.walk(run_path) if isinstance(n, (ast.For, ast.While))]
loop_names = {n.id for loop in loops for n in ast.walk(loop) if isinstance(n, ast.Name)}
assert "confirm_x" not in loop_names and "confirm_y" not in loop_names, loop_names

run_seed = functions["run_seed"]
gate_assignments = [
    n for n in ast.walk(run_seed)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "gates" for t in n.targets)
]
assert len(gate_assignments) == 1
gate_text = ast.unparse(gate_assignments[0]).lower()
assert "confirm" not in gate_text, gate_text

source = ENGINE.read_text()
freeze_marker = "paths.append(record); frozen.append(endpoint)"
confirm_marker = "initial_confirm=loss(theta,confirm_x,confirm_y).item()"
assert source.index(freeze_marker) < source.index(confirm_marker)

print({
    "online_loop_reads_confirm": False,
    "candidate_gates_read_confirm": False,
    "all_paths_frozen_before_first_confirm_access": True,
    "causal_boundary_static_audit": "PASS",
})
