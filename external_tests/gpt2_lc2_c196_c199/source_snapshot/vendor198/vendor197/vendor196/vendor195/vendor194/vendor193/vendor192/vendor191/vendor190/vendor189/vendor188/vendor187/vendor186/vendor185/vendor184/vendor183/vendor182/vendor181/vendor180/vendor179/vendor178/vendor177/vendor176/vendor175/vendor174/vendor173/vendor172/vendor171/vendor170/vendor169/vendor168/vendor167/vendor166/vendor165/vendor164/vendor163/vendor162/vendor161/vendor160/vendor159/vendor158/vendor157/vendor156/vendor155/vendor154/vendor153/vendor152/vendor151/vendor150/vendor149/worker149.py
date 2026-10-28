import argparse,traceback
from core149 import *
from audit149 import audit

def run(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);s=dict(protocol=PROTOCOL,complete=False,decision='C149_INCOMPLETE_READONLY_AUDIT_NOT_ACCEPTED',new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_router_candidates=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,gradients_independently_recomputed=False,thresholds_changed=False,candidate_selected=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False);errors=[];dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());dump(out/'SUMMARY.json',s)
    try:
        with no_models(),safe.no_solvers():s=audit(paths,out,c)
    except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
    checks={}
    for k,p in paths.items():
        try:checks[k]=sha(p)==c['pins'][k]['sha256']
        except Exception:checks[k]=False
    s['original_inputs_unchanged']=checks
    if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):s.update(complete=False,decision='C149_INCOMPLETE_READONLY_AUDIT_NOT_ACCEPTED');errors.append('Original input missing or changed')
    s['errors']=errors;dump(out/'SUMMARY.json',s);return s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);console={k:v for k,v in s.items() if k not in ('paired_all_arms','final_training_summaries')};console['final_training_summaries']={'all84':s.get('final_training_summaries',{}).get('all84',{})};print(json.dumps(console,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
