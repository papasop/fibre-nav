"""Bounded CPU archive audit; no model, derivative, fitter or solver calls."""
import argparse,traceback
from core147 import *
from audit147 import audit

def run(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);errors=[];s=dict(protocol=PROTOCOL,complete=False,decision='C147_INCOMPLETE_READONLY_AUDIT_NOT_ACCEPTED',new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,gradients_independently_recomputed=False,parent_decision_reclassified=False,thresholds_changed=False,candidate_selected=False,LC2_closed=False,LC2_single_fold_passed=False)
    dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());dump(out/'SUMMARY.json',s)
    try:
        with no_models(),safe.no_solvers():s=audit(paths,out,c)
    except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    s['errors']=errors;dump(out/'SUMMARY.json',s);return s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
