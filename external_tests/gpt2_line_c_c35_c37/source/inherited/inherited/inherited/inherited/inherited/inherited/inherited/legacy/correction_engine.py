"""The C9 response-only correction rule, with resumable, instrumented callbacks.

No target bits or writer/teacher are available to this engine.
"""
import time
from geometry import ridge_correction
from confirmation_protocol import correction_accept

def run_correction(initial,measure,jacobian,propose,clone,persist,settings,resume=None,synchronise=lambda:None):
 current=clone(initial) if resume is None else clone(resume['theta'])
 begin=0 if resume is None else int(resume['iteration'])
 history=[] if resume is None else list(resume['history'])
 elapsed=0. if resume is None else float(resume['elapsed_seconds'])
 if not 0<=begin<=settings['correction_max_steps']:raise ValueError('Invalid correction iteration')
 if resume is not None and resume.get('done'):return current,history,elapsed
 def saved(iteration,done):
  persist({'theta':clone(current),'iteration':iteration,'history':history,'elapsed_seconds':elapsed,'done':done})
 for step in range(begin,settings['correction_max_steps']):
  synchronise();start=time.perf_counter();e,drift,kl=measure(current)
  if drift<=settings['response_budget']:
   synchronise();elapsed+=time.perf_counter()-start
   history.append({'iteration':step,'stop':'RESPONSE_WITHIN_BUDGET','drift':drift,'kl':kl});saved(step,True);return current,history,elapsed
  synchronise();measurement_seconds=time.perf_counter()-start;ts=time.perf_counter();J=jacobian(current);synchronise();jacobian_seconds=time.perf_counter()-ts
  ts=time.perf_counter();delta,info=ridge_correction(J,e,settings['correction_ridge']);svd_seconds=time.perf_counter()-ts
  ts=time.perf_counter()
  accepted=False;trials=[]
  for bt in range(settings['correction_backtracking']):
   alpha=2.**(-bt);trial=propose(current,alpha*delta);_,new_drift,new_kl=measure(trial)
   ok=correction_accept(drift,new_drift,alpha)
   trials.append({'alpha':alpha,'drift':new_drift,'kl':new_kl,'accepted_for_response_only':ok})
   if ok:current=trial;accepted=True;break
  synchronise();backtracking_seconds=time.perf_counter()-ts;elapsed+=time.perf_counter()-start
  history.append({'iteration':step+1,'measured_pre_step_residual':e.tolist(),'pre_step_drift':drift,
   'correction':info,'pre_step_measurement_seconds':measurement_seconds,'jacobian_seconds':jacobian_seconds,
   'svd_seconds':svd_seconds,'backtracking_seconds':backtracking_seconds,'trials':trials,'accepted':accepted})
  done=not accepted or step+1==settings['correction_max_steps'];saved(step+1,done)
  if done:return current,history,elapsed
 saved(begin,True);return current,history,elapsed
