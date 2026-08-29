#!/usr/bin/env python3
"""Static audit for the frozen v4.2d prospective confirmation."""
import ast
from pathlib import Path

p=Path(__file__).with_name("cner_resnet18_cifar10_full_layer4_transverse_confirm_v4_2d.py")
s=p.read_text(); tree=ast.parse(s)
assert "autograd.functional.jacobian" not in s
assert "torch.func import functional_call, jvp, vjp, vmap" in s
assert "MICRO_SCALES=[1.0,0.25,0.0625,0.015625]" in s
assert "model=model.double()" in s and "anchor_x=anchor_x.double()" in s
assert "torch.backends.cuda.matmul.allow_tf32=False" in s
assert "torch.backends.cudnn.allow_tf32=False" in s
assert "SEEDS=[76742" in s and "76757]" in s
func={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
loop_names={n.id for loop in ast.walk(func["run_path"]) if isinstance(loop,(ast.For,ast.While)) for n in ast.walk(loop) if isinstance(n,ast.Name)}
assert "confirm_x" not in loop_names and "confirm_y" not in loop_names
text=ast.unparse(func["run_seed"])
assert text.index("paths =") < text.index("source_confirm =")
for token in ("finite_radius_ladder","median_finest_finite_over_linear_prediction",
              "fraction_states_improving_toward_linearity","jvp_linearity_relative_error",
              "chart_level_separation","active_response_separation",
              "transverse_gain_contrast","finite_radius_validation",
              "radius_convergence","jvp_identity"):
    assert token in s
assert "<= 1e-08" in text and "0.9 <=" in text and "<= 1.1" in text
assert ">= 0.75" in text and ">= 1000" in text
main=ast.unparse(func["main"])
assert "count >= 12" in main and "'seeds': 16" in main
print({"explicit_jacobian":False,"matrix_free":True,"online_confirm_access":False,
       "audit_float64":True,"tf32_disabled":True,"four_radius_levels":True,
       "sixteen_new_seeds":True,"required_seeds":12,"audit":"PASS"})
