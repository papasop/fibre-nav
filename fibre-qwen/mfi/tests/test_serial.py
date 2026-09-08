import unittest
import torch
from evaluate_serial import run_sequence

class Chart:
    def __init__(self): self.initial=torch.tensor([0.]); self.v=self.initial.clone()
    def vector(self): return self.v
    def audit(self,target): return {'response':0.,'kl':0.,'margin':2.}
class Controller:
    def __init__(self,chart,fail=None): self.chart=chart;self.calls=[];self.fail=fail
    def read(self): return {'value':0}
    def execute(self,op,target):
        self.calls.append((op,target))
        if len(self.calls)==self.fail: return {'accepted':False}
        self.chart.v+=1
        return {'accepted':True}
    def passes(self,audit): return True
class SerialTests(unittest.TestCase):
    def test_same_state_twenty_alternating_operations(self):
        c=Chart();k=Controller(c)
        rows=run_sequence(c,k,lambda *a:[],lambda r:None)
        self.assertEqual(len(rows),20);self.assertEqual(float(c.v),20)
        self.assertEqual(k.calls,[('WRITE' if i==0 else 'OVERWRITE',1 if i%2==0 else 0) for i in range(20)])
        self.assertEqual(float(c.initial),0)
    def test_failure_stops_but_preserves_planned_denominator(self):
        c=Chart();k=Controller(c,fail=3)
        rows=run_sequence(c,k,lambda c,k,t,ok:[{'correct':ok}],lambda r:None)
        self.assertEqual(len(k.calls),3);self.assertEqual(len(rows),20)
        self.assertTrue(rows[2]['rollback_verified']);self.assertTrue(all(r['skipped'] for r in rows[3:]))
        self.assertEqual(sum(r['operation_pass'] for r in rows),2)
    def test_eval_mutation_is_rejected(self):
        c=Chart();k=Controller(c)
        def bad(c,*args): c.v+=1;return []
        with self.assertRaisesRegex(RuntimeError,'evaluation changed'): run_sequence(c,k,bad,lambda r:None)
if __name__=='__main__': unittest.main()
