"""Fixed symmetric scalar directional checks; no head or step selection."""
from core165 import *
def analyse(anchor,repeat,probes,d,ds,native):
 rr=[]
 for label,h in zip(('1e-6','5e-6','1e-5'),H):
  minus=probes['minus_'+label];plus=probes['plus_'+label]
  for i,sid in enumerate(IDS):
   base=anchor[i];a=minus[i];b=plus[i];parts={}
   for name,terms,grad in [('response',('response_loss',),'response_gradient'),('guard',PARTS[1:],'guard_gradient')]:
    lo=sum(a['objective_parts'][k] for k in terms);hi=sum(b['objective_parts'][k] for k in terms);ad=float(np.sum(np.asarray(base[grad])*d));fd=(hi-lo)/(2*h);parts[name]=dict(AD_directional=ad,central_difference=fd,difference=fd-ad,absolute_difference=abs(fd-ad),no_derivative_error_or_curvature_cause_claim=True)
   rr.append(dict(state_id=sid,h=h,checks=parts))
 repeats=[]
 for a,b in zip(anchor,repeat):
  compare(a['state_id'],b['state_id']);err=float(np.max(abs(np.asarray(a['observed'])-b['observed'])));kl=abs(a['KL']-b['KL']);compare(a['FP64_point_sha256'],b['FP64_point_sha256'],'Same anchor actual FP64 identity on repeat');need(err<=2e-9 and kl<=2e-9,'Same-run full84 anchor repeat');repeats.append(dict(state_id=a['state_id'],observed_Linf=err,KL_absolute=kl,within_original_repeat_tolerance=True))
 summaries={};pairs={};native_repeat=[]
 for arm in ['anchor']+[a['name'] for a in ARMS]:
  a=native[(arm,0)];b=native[(arm,1)];summaries[arm]={str(t):parent.native_summary(native[(arm,t)]) for t in (0,1)}
  for x,y in zip(a,b):
   r=archive155.repeat_check(x['metrics'],y['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Native same-point/gates repeat');native_repeat.append(dict(arm=arm,state_id=x['state_id'],**r))
  if arm!='anchor':pairs[arm]={str(t):parent.paired(native[('anchor',t)],native[(arm,t)]) for t in (0,1)}
 return dict(native_summaries=summaries,paired_vs_anchor=pairs,all84_all6_arms_retained=True,computed_strict_common_response_descent=ds['computed_strict_common_response_descent'],direction_zero=ds['direction_zero'],all6_FP64_maximum_response={k:max(float(max(abs(np.asarray(r['observed'])[:6]))) for r in v) for k,v in probes.items()},any_probe_pass_is_training_diagnostic_only=True,step_or_head_selected=False,LC2_closed=False),rr,repeats,native_repeat
