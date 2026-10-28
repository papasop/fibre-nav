"""Synthetic whole C188 archive; no real science originals or model execution."""
import copy,contextlib,io,shutil
from core189 import *
from fixture188 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker188 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C188',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C188 generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor188',root/'C188/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C188']=pack(root/'C188',root/'C188.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C188'
