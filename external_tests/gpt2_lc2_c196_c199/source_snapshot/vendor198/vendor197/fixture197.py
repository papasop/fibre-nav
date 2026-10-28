"""Synthetic whole C196 archive; no real science originals or model execution."""
import copy,contextlib,io,shutil
from core197 import *
from fixture196 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker196 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C196',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C196 generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor196',root/'C196/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C196']=pack(root/'C196',root/'C196.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C196'
