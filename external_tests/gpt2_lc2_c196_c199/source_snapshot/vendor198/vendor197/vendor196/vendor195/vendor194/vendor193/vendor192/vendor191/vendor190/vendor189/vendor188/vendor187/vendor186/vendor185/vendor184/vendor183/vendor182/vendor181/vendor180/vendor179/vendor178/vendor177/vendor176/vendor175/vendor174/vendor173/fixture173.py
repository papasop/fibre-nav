"""NumPy-generated original C172 layout; all upstream contexts injected."""
import copy,contextlib,io,shutil
from core173 import *
from fixture172 import context_fixture
from fixture171 import Analytic as Base,synthetic_solver,pack
class Analytic(Base):
 def __init__(self,*a):
  super().__init__(*a);f=self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json';r=clean_json(f.read_bytes());r['protocol']=prior.PROTOCOL;dump(f,r)
def fixture(root):
 from worker172 import run
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=context_fixture(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C172',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C172 worker schema generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor172',root/'C172/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C172']=pack(root/'C172',root/'C172.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C172'
