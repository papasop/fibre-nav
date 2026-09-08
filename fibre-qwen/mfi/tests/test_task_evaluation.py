import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_tasks import evaluate_stage, summarize

class Chart:
    def __init__(self): self.initial=torch.zeros(1); self.v=torch.ones(1); self.calls=[]
    def vector(self): return self.v.clone()
    def set_vector(self,v): self.v=v.clone()
    def read(self): return {'value':0}  # deliberately wrong committed memory for target 1
    def single(self,s): return {'A':0,'B':1}[s]
    def logits(self,text):
        self.calls.append((float(self.v[0]),text))
        detailed='Saved user preference: detailed answers.' in text
        a_brief='A = brief' in text
        return torch.tensor([0.,1.]) if detailed==a_brief else torch.tensor([1.,0.])

class Tests(unittest.TestCase):
    def test_no_reference_leakage_and_baselines_use_initial_state(self):
        chart=Chart(); rows=evaluate_stage(chart,chart,1,True)
        scores=summarize([{'items':rows}])
        self.assertEqual(scores['mfi']['correct'],0)
        self.assertEqual(scores['external']['correct'],8)
        self.assertTrue(all(v==1 for v,_ in chart.calls[:8]))
        self.assertTrue(all(v==0 for v,_ in chart.calls[8:]))
        self.assertTrue(torch.equal(chart.vector(),torch.ones(1)))
    def test_failures_kept_in_denominator(self):
        chart=Chart(); rows=evaluate_stage(chart,chart,1,False)
        mfi=[r for r in rows if r['arm']=='mfi']
        self.assertEqual(len(mfi),8)
        self.assertTrue(all(r['prediction'] is None and not r['correct'] for r in mfi))
        self.assertEqual(len(chart.calls),16)
    def test_exception_restores_committed_state(self):
        chart=Chart()
        def fail(text): raise RuntimeError('scorer failure')
        chart.logits=fail
        with self.assertRaises(RuntimeError): evaluate_stage(chart,chart,1,False)
        self.assertTrue(torch.equal(chart.vector(),torch.ones(1)))

if __name__=='__main__': unittest.main()
