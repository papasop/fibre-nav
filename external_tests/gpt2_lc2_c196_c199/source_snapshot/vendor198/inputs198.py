"""Reconstruct C196/C197 with immutable delivered checkers, never execute archive source."""
from core198 import *
from inputs196 import load as load196
from audit197 import audit
from diagnostics197 import diagnostics
from focus197 import focus_report
from replay197 import replay_report
from audit147 import source_check
def bind197(path,a,d,focus,replay,recon,out,c):
 expected=dict(parent_protocol=parent.prior.PROTOCOL,parent_decision=a['summary']['decision'],native_summaries=a['native_summaries'],paired_by_round=a['paired'],initial_head_digest=parameter_hash(a['heads'][0]),fixed_final_step8_head_digest=parameter_hash(a['heads'][-1]),diagnostics=d,endpoint_precision_reconstruction=a['precision'],rescue_and_remaining_failure_focus=focus,cross_run_anchor_replay=replay)
 with contextlib.closing(Archive(path)) as z:
  count=z.verify_manifest();source_check(z,c['source197'],'C197');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['decision'],parent.DECISION);compare(s['synthetic_fixture'],not c['production']);compare(s['errors'],[]);compare(s['stage'],'complete');compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),parent.design());compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS})
  for k,v in expected.items():compare(s[k],v,'Reconstructed original C197/'+k)
  for k,v in parent.no_compute().items():compare(s[k],v,'Original C197 no compute/'+k)
  for k,v in dict(training_states_audited=84,training_updates_audited=8,gradient_records_audited=672,ordered_records_audited=1176,actual_FP64_point_arrays_rehashed=840,native_endpoint_records_audited=336,all_forty_two_originals_independently_bound=c['production'],C196_zip_sha256=c['pins']['C196']['sha256']).items():compare(s[k],v,'Original C197 coverage/'+k)
  links=z.json('ORIGINAL_INPUTS.json');compare(set(links),set(parent.INPUT_KEYS))
  for k in parent.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'Original C197 ancestry/'+k)
  checked=[]
  for directory in ('C196_arithmetic','diagnostics','focus','replay'):
   for p in sorted((Path(recon)/directory).rglob('*')):
    if not p.is_file():continue
    n=p.relative_to(recon).as_posix()
    if p.suffix=='.json':compare(z.json(n),clean_json(p.read_bytes()),'Original C197 reconstructed/'+n)
    elif p.suffix=='.csv':
     import csv,io
     old=list(csv.reader(io.StringIO(z.raw(n).decode())));new=list(csv.reader(io.StringIO(p.read_text())));compare(len(old),len(new),'CSV rows/'+n)
     for u,v in zip(old,new):
      compare(len(u),len(v),'CSV columns/'+n)
      for x,y in zip(u,v):
       try:xf=float(x);yf=float(y)
       except ValueError:compare(x,y,'CSV literal/'+n)
       else:need(math.isfinite(xf) and math.isfinite(yf),'Finite CSV');compare(xf,yf,'CSV numeric/'+n)
    else:compare(z.raw(n),p.read_bytes(),'Original C197 artifact/'+n)
    checked.append(n)
  need(len(checked)>=42,'Full C197 diagnostic artifact inventory')
 receipt=dict(protocol=parent.PROTOCOL,decision=s['decision'],members_hashed=count,artifacts_bound=checked,model_outputs_independently_recomputed=False,new_model_forwards=0)
 dump(Path(out)/'C197_ORIGINAL_REBOUND.json',receipt);return receipt
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original before payload/'+k)
 for n,h in c['source197'].items():compare(sha(ROOT/'vendor197'/n),h,'Immutable delivered C197 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load196({k:paths[k] for k in parent.prior.INPUT_KEYS},out/'forty_one_originals',c['parent']['parent'],budget)
  recon=out/'C197_reconstruction';a=audit(paths['C196'],ctx,recon/'C196_arithmetic',c['parent'],budget);d=diagnostics(a,recon/'diagnostics');focus=focus_report(a,recon/'focus');replay=replay_report(a,ctx,recon/'replay');budget();receipt=bind197(paths['C197'],a,d,focus,replay,recon,out,c)
 p=a['heads'][-1].copy();check_head(p,c['production']);need(a['summary']['final_training_all84_passed_both_rounds'],'C196 fixed final qualification prerequisite')
 ctx.update(C197_receipt=receipt,C196_final_head=p,C196_final_native={str(j):a['native']['native_final'+str(j)] for j in (0,1)},C196_final_points={sid:pt.copy() for sid,pt in a['native_points']['final'].items()})
 dump(out/'ALL43_ORIGINALS_BOUND.json',dict(all_originals_bound=c['production'],C196_head_digest=parameter_hash(p),C197_decision=receipt['decision'],new_model_forwards=0));return ctx
