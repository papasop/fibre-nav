"""Synthetic whole C186 archive; no real science originals or model execution."""
import copy,contextlib,io,shutil
from core187 import *
from fixture186 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker186 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C186',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C186 generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor186',root/'C186/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C186']=pack(root/'C186',root/'C186.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C186'
