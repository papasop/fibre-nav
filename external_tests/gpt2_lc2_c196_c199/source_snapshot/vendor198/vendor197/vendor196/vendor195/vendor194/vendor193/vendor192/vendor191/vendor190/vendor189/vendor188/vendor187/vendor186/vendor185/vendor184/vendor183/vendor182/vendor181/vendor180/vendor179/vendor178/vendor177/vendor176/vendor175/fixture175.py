"""Synthetic C174 whole archive; upstream contexts injected, no model or solver."""
import copy,contextlib,io,shutil
from core175 import *
from fixture174 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker174 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C174',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C174 schema generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor174',root/'C174/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C174']=pack(root/'C174',root/'C174.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C174'
