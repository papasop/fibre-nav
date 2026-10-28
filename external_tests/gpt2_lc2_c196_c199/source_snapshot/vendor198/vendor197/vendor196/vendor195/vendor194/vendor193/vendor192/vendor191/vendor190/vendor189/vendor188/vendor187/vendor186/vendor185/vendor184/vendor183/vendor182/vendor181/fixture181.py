"""Synthetic whole C180 archive; no real science originals or model execution."""
import copy,contextlib,io,shutil
from core181 import *
from fixture180 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker180 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C180',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C180 generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor180',root/'C180/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C180']=pack(root/'C180',root/'C180.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C180'
