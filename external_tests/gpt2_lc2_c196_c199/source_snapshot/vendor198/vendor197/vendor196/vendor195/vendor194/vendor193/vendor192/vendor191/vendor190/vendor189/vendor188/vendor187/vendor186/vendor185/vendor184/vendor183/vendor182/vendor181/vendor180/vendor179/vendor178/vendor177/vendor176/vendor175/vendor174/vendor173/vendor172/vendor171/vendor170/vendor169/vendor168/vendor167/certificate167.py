"""Independent exact verifier: no SciPy import, no reliance on solver statuses."""
from fractions import Fraction as Q
import math,json
import numpy as np

def require(ok,msg):
 if not ok:raise ValueError(msg)
def fq(v):
 x=float(v);require(math.isfinite(x),'Nonfinite certificate scalar');return Q.from_float(x)
def enc(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator),approximate=float(q))
def dec(x):
 require(set(x)=={'numerator','denominator','approximate'},'Malformed fraction')
 require(isinstance(x['numerator'],str) and isinstance(x['denominator'],str),'Fraction strings required')
 q=Q(int(x['numerator']),int(x['denominator']));require(q.denominator>0,'Positive denominator');require(float(q)==x['approximate'],'Fraction approximation mismatch');return q

def verify(raw,guards,free,active,norms,direction,dual,ids):
 raw=np.asarray(raw);guards=np.asarray(guards);free=np.asarray(free);active=np.asarray(active);norms=np.asarray(norms);d=np.asarray(direction);dual=np.asarray(dual)
 require(raw.dtype==np.float64 and raw.ndim==2 and guards.dtype==np.float64 and guards.shape==raw.shape,'Full binary64 matrices')
 n,m=raw.shape;require(len(ids)==n and len(set(ids))==n,'Ordered unique IDs');require(free.dtype==np.bool_ and free.shape==(m,),'Boolean fixed subspace');require(active.dtype.kind in 'iu' and active.ndim==1 and active.tolist()==sorted(set(active.tolist())) and all(0<=int(i)<n for i in active),'Canonical active rows');require(norms.dtype==np.float64 and norms.shape==(n,),'Binary64 row norms');require(d.dtype==np.float64 and d.shape==(m,) and dual.dtype==np.float64 and dual.shape==(len(active),),'Witness shapes')
 for a in (raw,guards,norms,d,dual):require(np.isfinite(a).all(),'Nonfinite certificate input')
 require(np.all(norms>=0),'Nonnegative row norms');require(np.max(abs(d),initial=0)<=1 and np.all(d[~free]==0),'Direction outside declared box/subspace');require(np.all(dual>=0),'Negative dual weights')
 D=[fq(v) for v in d];rr=[[fq(v) for v in row] for row in raw];gg=[[fq(v) for v in row] for row in guards];rowout=[];B=[]
 for i in active:
  norm=fq(norms[i]);projected=[v if free[j] else Q(0) for j,v in enumerate(rr[i])]
  if norm==0:require(all(v==0 for v in projected),'Zero norm hiding nonzero row');B.append([Q(0)]*m)
  else:B.append([v/norm for v in projected])
 dots=[sum((a*b for a,b in zip(row,D)),Q(0)) for row in B]
 for i in range(n):
  r=sum((a*b for a,b in zip(rr[i],D)),Q(0));g=sum((a*b for a,b in zip(gg[i],D)),Q(0));rowout.append(dict(state_id=ids[i],active=i in active,raw_response_dot=enc(r),guard_dot=enc(g),strict_raw_response_descent=r<0,strict_guard_descent=g<0))
 if not len(active):
  require(not np.any(d) and not len(dual),'Empty-active witness must be zero');return dict(classification='NO_ACTIVE_RESPONSE_LOSS_NO_DIRECTION_REQUIRED',active_count=0,strict_recorded_raw_common_descent=False,no_strict_recorded_raw_common_descent_proved=False,optimal_margin_lower=None,optimal_margin_upper=None,optimal_margin_interval_width=None,exact_dual_weights=[],exact_normalized_row_dots=[],rows=rowout,scope='Recorded gradients only; no finite or model acceptance.')
 q=[fq(v) for v in dual];total=sum(q,Q(0));require(total>0,'Zero dual mass');w=[v/total for v in q];require(sum(w)==1 and all(v>=0 for v in w),'Exact simplex')
 residual=[sum((w[i]*B[i][j] for i in range(len(B))),Q(0)) for j in range(m)];lower=-max(dots);upper=sum((abs(v) for v in residual),Q(0));require(lower<=upper,'Exact weak duality failure')
 strict=all(rowout[int(i)]['strict_raw_response_descent'] for i in active)
 require(strict==(lower>0),'Positive scaling must preserve strict signs')
 impossible=upper==0;require(not(strict and impossible),'Contradictory exact witnesses')
 classification='RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED' if strict else 'RECORDED_GRADIENT_STRICT_COMMON_DESCENT_IMPOSSIBLE_CERTIFIED' if impossible else 'RECORDED_GRADIENT_FEASIBILITY_UNRESOLVED'
 return dict(classification=classification,active_count=len(active),strict_recorded_raw_common_descent=strict,no_strict_recorded_raw_common_descent_proved=impossible,optimal_margin_lower=enc(lower),optimal_margin_upper=enc(upper),optimal_margin_interval_width=enc(upper-lower),exact_dual_weights=[enc(v) for v in w],exact_normalized_row_dots=[dict(state_id=ids[int(i)],dot=enc(v)) for i,v in zip(active,dots)],exact_dual_residual=[enc(v) for v in residual],rows=rowout,scope='Exact statement about archived binary64 gradients in the declared two-sided coefficient subspace only; true derivatives and finite writes not certified.')

def verify_saved(arrays,certificate,ids):
 require(set(arrays)=={'raw','guards','free','active','norms','direction','dual'},'Exact certificate array inventory')
 expected=verify(**arrays,ids=ids)
 require(expected==certificate,'Saved rational certificate differs from independent reconstruction')
 return expected
