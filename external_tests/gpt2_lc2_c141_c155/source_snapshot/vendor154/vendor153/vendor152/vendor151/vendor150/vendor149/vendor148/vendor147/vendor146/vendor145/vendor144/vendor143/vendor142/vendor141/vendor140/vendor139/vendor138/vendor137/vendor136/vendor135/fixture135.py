"""Three synthetic capsules generated with delivered C133/C134 protocols."""
import contextlib,io,shutil
from core135 import *
from fixture134 import make_fixture as fixture134,Toy
from fixture133 import pack
from worker134 import run as run134

def make_fixture(folder):
    folder=Path(folder);paths,c14=fixture134(folder/'parents');out=folder/'C134'
    with contextlib.redirect_stdout(io.StringIO()):s=run134(paths,out,c14,Toy)
    need(s['complete'] and s['path_interpretation_qualified'],'Synthetic C134 generation incomplete')
    shutil.copytree(ROOT/'vendor134',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p134=pack(out,folder/'C134_SYNTHETIC.zip')
    paths=dict(paths,C134=p134);c=dict(contract(),production=False,pins=dict(c14['pins'],C134=dict(path=str(p134),sha256=sha(p134))),source132=c14['source132'],source133=c14['source133'])
    return paths,c
