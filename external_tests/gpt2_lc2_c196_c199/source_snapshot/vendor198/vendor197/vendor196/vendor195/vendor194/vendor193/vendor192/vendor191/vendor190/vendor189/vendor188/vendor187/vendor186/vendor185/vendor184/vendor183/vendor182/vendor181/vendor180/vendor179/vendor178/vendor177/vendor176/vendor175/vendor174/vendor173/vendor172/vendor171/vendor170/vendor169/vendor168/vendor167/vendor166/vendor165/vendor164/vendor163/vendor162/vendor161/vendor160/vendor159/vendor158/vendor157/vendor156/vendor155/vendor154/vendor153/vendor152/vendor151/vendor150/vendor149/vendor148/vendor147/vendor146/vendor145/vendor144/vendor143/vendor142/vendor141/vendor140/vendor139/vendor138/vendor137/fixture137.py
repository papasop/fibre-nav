"""Analytic synthetic fixtures; never claims a real GPT2 or AD validation."""
import contextlib,io,shutil
from core137 import *
from fixture136 import make_fixture as fixture136,Analytic as PriorAnalytic,Toy
from fixture133 import pack
from worker136 import run as run136

def make_fixture(folder):
    folder=Path(folder);paths,c36=fixture136(folder/'parents');out=folder/'C136'
    with contextlib.redirect_stdout(io.StringIO()):s=run136(paths,out,c36,Toy,PriorAnalytic)
    need(s['complete'],'Synthetic C136 generation incomplete')
    shutil.copytree(ROOT/'vendor136',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    p=pack(out,folder/'C136_SYNTHETIC.zip')
    c=dict(c36,pins=dict(c36['pins'],C136=dict(path=str(p),sha256=sha(p))),source136=contract()['source136'])
    return dict(paths,C136=p),c

class Analytic(PriorAnalytic):
    def __init__(self,*args,**kw):super().__init__(*args,**kw);self.warm_calls=0
    def zero(self,spec,a):
        warm=self.phase.phase=='functional_warm';row=super().zero(spec,a)
        if warm:
            self.z-=1;self.warm_calls+=1
            if self.mode=='warm_drift':row['observed'][0]+=.001
        return row
    def gradient(self,a,step,out,budget):
        _,rows=super().gradient(a,step,out,budget)
        if self.mode=='missing_state':return np.zeros_like(a),rows
        for row in rows:
            row['C137_membership']=row.pop('C136_membership');row['state_gradient']=(12*np.array(row['gradient_contribution'])).tolist()
        g,agg=aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows])
        for row,w in zip(rows,agg['weights']):row.update(aggregate_weight=w,gradient_contribution=(w*np.array(row['state_gradient'])).tolist())
        if self.mode=='nonfinite':g[0]=np.nan
        if self.mode=='wrong_weight':rows[0]['aggregate_weight']+=.01
        if self.mode=='wrong_state_gradient':rows[0]['state_gradient'][0]+=.01
        dump(Path(out)/'AGGREGATION.json',agg);dump(Path(out)/'GRADIENT_PROGRESS.json',rows)
        return g,rows
    def stats(self):return dict(super().stats(),functional_warm_calls=self.warm_calls)
