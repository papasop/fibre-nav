"""Synthetic whole C194 archive; no real science originals or model execution."""
import copy,contextlib,io,shutil
from core195 import *
from fixture194 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker194 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C194',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C194 generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor194',root/'C194/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C194']=pack(root/'C194',root/'C194.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C194'
