"""Transactional Adam proposal and line search; all gates supplied by real evaluator."""
import copy
import torch

def cpu_state(head):return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}

def attempt_update(head,optimiser,loss_fn,evaluate,guard_fn,backtracks=8,clip=1.,progress=lambda *a:None):
 old=cpu_state(head);optold=copy.deepcopy(optimiser.state_dict());optimiser.zero_grad(set_to_none=True)
 try:
  loss,meta=loss_fn()
  if not torch.isfinite(loss):raise ValueError('Nonfinite offline training loss')
  if not meta.get('gradients_accumulated',False):loss.backward()
  grads=[p.grad for p in head.parameters() if p.grad is not None]
  if not grads or any(not torch.isfinite(g).all() for g in grads):raise ValueError('Invalid offline head gradient')
  gn=float(torch.nn.utils.clip_grad_norm_(head.parameters(),clip));optimiser.step()
  proposed=cpu_state(head);optnew=copy.deepcopy(optimiser.state_dict());trials=[]
  if any(not torch.isfinite(v).all() for v in proposed.values()):raise ValueError('Nonfinite proposed head')
  for bt in range(backtracks):
   alpha=2.**(-bt)
   # Always interpolate from the same accepted point; do not compound rejected updates.
   trial={k:old[k]+alpha*(proposed[k]-old[k]) for k in old};head.load_state_dict(trial)
   progress(bt,alpha);rows=evaluate();g=guard_fn(rows);trials.append({'alpha':alpha,'guard':g})
   if g['accepted']:
    optimiser.load_state_dict(optnew);optimiser.zero_grad(set_to_none=True)
    return rows,{'accepted':True,'alpha':alpha,'gradient_norm_before_clip':gn,'loss':float(loss.detach()),'terms':meta,'trials':trials}
  head.load_state_dict(old);optimiser.load_state_dict(optold);optimiser.zero_grad(set_to_none=True)
  return None,{'accepted':False,'alpha':None,'gradient_norm_before_clip':gn,'loss':float(loss.detach()),'terms':meta,'trials':trials}
 except BaseException:
  head.load_state_dict(old);optimiser.load_state_dict(optold);optimiser.zero_grad(set_to_none=True);raise
