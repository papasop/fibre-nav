"""Full84 synthetic fixtures only, never scientific measurements."""
from core161 import *
from fixture160 import context as old_context
def context():
 ctx=old_context();ctx['completed160']=dict(decision='C160_FULL84_FP64_OBJECTIVE_GRADIENT_BASELINE_COMPLETE_NO_TRAINING');return ctx
def inputs(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
 for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Synthetic fixture only');paths[k]=p
 c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
