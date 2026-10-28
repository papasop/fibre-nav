"""Synthetic whole C184 archive; no real science originals or model execution."""
import copy,contextlib,io,shutil
from core185 import *
from fixture184 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker184 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C184',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C184 generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor184',root/'C184/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C184']=pack(root/'C184',root/'C184.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C184'
