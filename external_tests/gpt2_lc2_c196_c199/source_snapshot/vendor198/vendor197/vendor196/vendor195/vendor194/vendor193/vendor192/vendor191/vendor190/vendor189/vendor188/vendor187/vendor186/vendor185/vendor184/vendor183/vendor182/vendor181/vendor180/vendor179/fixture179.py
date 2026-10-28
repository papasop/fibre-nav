"""Synthetic C178 archive; nonzero continuation anchor and one actual gate rescue."""
import copy,contextlib,io,shutil
from core179 import *
from fixture178 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack,observed

def fixture(root):
 from worker178 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 # Upstream is explicitly injected: create a controlled near-budget state so the
 # continuation crosses a native response gate, while other failures stay visible.
 ctx['data']['d']['points'][IDS[0]]['raw'][:6]=.00205
 chain=control.source168.old.parent.parent.previous.constants(ctx['data']);p=ctx['C176_final_head'];frames=[];native=[]
 for sid in IDS:
  pt,e=mapping(chain,p,sid);case=sid.split('__')[0];o=observed(pt,sid);met=archive155.full_metric(observed(pt.astype(np.float32),sid),0.,pt.astype(np.float32),case);q=archive155.endpoint_gates(met,case,pt.astype(np.float32),chain[sid+'__prepared'].astype(np.float32),True);native.append(dict(state_id=sid,case=case,metrics=met,qualification=q));frames.append(dict(state_id=sid,observed=o.tolist(),KL=0.))
 ctx.update(C176_final_FP64=frames,C176_final_native={'0':copy.deepcopy(native),'1':copy.deepcopy(native)})
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C178',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C178 generation: '+str(s['errors']));need(s['paired_by_round']['0']['rescued']==1,'Synthetic fixture must exercise an actual native rescue');need(s['paired_by_round']['0']['destroyed']==0,'Synthetic fixture retains previously passing states');shutil.copytree(ROOT/'vendor178',root/'C178/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C178']=pack(root/'C178',root/'C178.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C178'
