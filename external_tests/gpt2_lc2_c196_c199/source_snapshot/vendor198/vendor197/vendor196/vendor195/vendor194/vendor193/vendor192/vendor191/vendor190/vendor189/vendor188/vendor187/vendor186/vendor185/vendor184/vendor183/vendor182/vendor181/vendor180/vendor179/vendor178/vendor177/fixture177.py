"""Synthetic C176 whole archive; upstream contexts injected, no model/LP solve."""
import copy,contextlib,io,shutil
from core177 import *
from fixture176 import fixture as upstream,Analytic,synthetic_solver
from fixture171 import pack

def fixture(root):
 from worker176 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C176',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C176 schema generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor176',root/'C176/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C176']=pack(root/'C176',root/'C176.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C176'
