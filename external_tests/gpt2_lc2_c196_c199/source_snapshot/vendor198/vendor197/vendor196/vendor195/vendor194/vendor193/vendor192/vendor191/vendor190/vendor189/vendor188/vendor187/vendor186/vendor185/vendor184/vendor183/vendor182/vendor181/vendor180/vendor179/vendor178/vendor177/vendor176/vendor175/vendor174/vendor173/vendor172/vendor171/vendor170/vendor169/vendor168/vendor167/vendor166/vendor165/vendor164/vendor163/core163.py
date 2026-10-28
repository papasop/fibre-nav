"""Fixed smooth-risk FP64 restart from the archived C161 step8 head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor162'))
import core162 as previous
from core162 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,array_digest,point_hash,parameter_hash,registry,archive155
parent=previous.parent  # trusted core161; runtime/model implementation is unchanged
N=84;STEPS=8;RATE=1e-5;TAU=.01;TIMEOUT=10800
PROTOCOL='GPT2_LC2_C163_SMOOTH_RISK_FP64_CPU_V1'
INPUT_KEYS=previous.INPUT_KEYS+('C162',)
REQUESTED_FORWARDS=2187;REQUESTED_BACKWARDS=1344;FUNCTIONAL_FORWARDS=1514
IDS=previous.IDS
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=previous.contract(),source162=clean_json((ROOT/'trusted/C162_SOURCE_PINS.json').read_bytes()))
def design():
 return dict(protocol=PROTOCOL,training_registry=registry(),all84_exposed_training=True,initial_head='Exact C161 fixed final step8 tensor, verified against actual original ZIP and C162 arithmetic; no intermediate recovery choice.',fixed_change='Replace0.9*hardmax(R) by0.9*tau*log(mean(exp(R/tau))), tau=.01. Keep0.1*mean(R), every guard and all native gates unchanged. Begin a new eight-step fit with explicit zero Adam moments.',hypothesis='Exposed C162 records show nearly opposing gradients at a maximizer switch and two uphill recorded-gradient update dot products. Smoothing is a registered research choice, not a proven cause or guaranteed cure.',objective='rho=tau*log(mean(exp(R/tau))); J_s=.9*rho+.1*mean(R)+mean(G). Stable shifted log-mean-exp over all84, weights=.9*softmax(R/tau)+.1/84. Also report original hardmax objective and all84 losses at every step.',optimizer=dict(name='manual projected FP64 Adam',new_updates=8,rate=RATE,betas=[.9,.999],epsilon=1e-8,global_gradient_L2_clip=1.,box=[-.125,.125],moment_initialization='Fresh zeros, not resumed C161 moments',local_bias_correction_steps=list(range(1,9)),combined_C161_C163_new_update_positions=list(range(9,17))),fixed_final_checkpoint='Local step8 only (eight new updates after C161 step8); never select an intermediate head, rate or temperature.',frozen='Original V,84 prewrite routing weights, prepared/raw states, GPT-2/LoRA-A and original cached response/logits references. Only216 head coefficients trained in production.',loss_parts=parent.design()['loss'],response_risk_override='Only the hardmax aggregation is replaced as stated above; loss_parts.aggregate describes the original parallel diagnostic, not the C163 training objective.',acceptance_budgets=parent.design()['acceptance_budgets'],replay=dict(native_initial_vs_C161_final=dict(observed_Linf=.0002,KL_absolute=5e-6,same_gates_and_point_required=True),FP64_initial_vs_C161_final=dict(observed_Linf=2e-9,KL_absolute=2e-9,checked_before_first_head_update=True)),point_archiving='Before every gradient batch and final no-grad evaluation, save all84 actual Torch FP64 points and hashes with the current head. Verify actual graph points against these seals before model forwards. Total756 actual FP64 point tensors. NumPy reconstruction remains a separate map check.',measurement_order='Native initial round0; eight complete84 gradient/update batches; final84 FP64 objective; native final round0; native initial round1; native final round1.',counts=dict(total_records=1092,training_records=672,final_FP64_records=84,native_records=336,actual_FP64_point_tensors=756,model_forwards=2187,functional_FP64_forwards=1514,scalar_backwards=1344,coefficient_updates=8),worker_budget_seconds=TIMEOUT,LC2_closed=False,scope='New training-only smooth-risk fit from C161 step8. Objective aggregation, additional optimization and explicit moment restart are not a matched causal comparison; no optimizer superiority claim. All84 remain exposed training. Original hardmax diagnostics and original native acceptance retained. No new preparation, fresh confirmation, excluded input, candidate selection, deployment, convergence, feasibility or LC2 closure claim.')
def objective(rows):
 old=previous.objective(rows);R=np.asarray([r['objective_parts']['response_loss'] for r in rows],float);mx=float(max(R));scaled=(R-mx)/TAU
 with np.errstate(under='ignore'):ex=np.exp(scaled)
 weights=ex/ex.sum();rho=mx+TAU*float(np.log(ex.mean()));need(np.isfinite(rho) and rho>=-1e-14,'Finite nonnegative log-mean-exp risk');rho=max(0.,rho);J=.9*rho+.1*old['mean_response_loss']+old['mean_guard_loss'];need(np.isfinite(J),'Finite smooth full84 objective')
 return dict(objective=float(J),smooth_response_risk=float(rho),temperature=TAU,original_hardmax_objective=old['objective'],maximum_response_loss=old['maximum_response_loss'],mean_response_loss=old['mean_response_loss'],mean_guard_loss=old['mean_guard_loss'],response_weights=(.9*weights+.1/N).tolist(),guard_weights=[1/N]*N,hardmax_maximizing_state_ids=old['maximizing_state_ids'],hardmax_exact_tie_count=old['exact_max_tie_count'])
def aggregate(rows,shape):
 a=objective(rows);rg=np.asarray([farray(np.asarray(r['response_gradient'],float),shape) for r in rows]);gg=np.asarray([farray(np.asarray(r['guard_gradient'],float),shape) for r in rows]);g=np.sum(np.asarray(a['response_weights'])[:,None,None,None]*rg+gg/N,axis=0);need(np.isfinite(g).all(),'Finite complete coefficient gradient');return g,a
def independent_gradient(rows,shape):
 # Separate flatten/loop implementation and normalized exponent arithmetic.
 R=np.array([r['objective_parts']['response_loss'] for r in rows]);ex=np.exp((R-R.max())/.01);total=ex.sum();result=np.zeros(int(np.prod(shape)),float)
 for i,r in enumerate(rows):result+=(.1/84+.9*ex[i]/total)*np.asarray(r['response_gradient'],float).ravel()+np.asarray(r['guard_gradient'],float).ravel()/84
 return result.reshape(shape)
def update(p,g,m,v,step):
 # Independently implemented scalar C162 Adam; same hyperparameters as C161.
 return previous.adam(p,g,m,v,step)
mapping=previous.mapping
seal_head=parent.seal_head
requests=parent.requests
native_summary=parent.native_summary
paired=parent.paired
