"""New worker's nonempty LP/certificate stages, with explicit synthetic upstream stubs."""
import tempfile,unittest,contextlib,io,zipfile,copy
from unittest.mock import patch
from core167 import *
import worker167
class Checks(unittest.TestCase):
 def test_nonempty_full84_worker_stages_LP_receipts_exact_certificate(self):
  # Upstream archive reconstruction is covered separately by test_archive167.
  # This test stubs ONLY upstream work, and exercises actual current LP and certificate code.
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=np.zeros((3,2,36));rows=[]
   for i,sid in enumerate(IDS):
    g=np.zeros_like(p)
    if i<9:g.ravel()[i]=1.
    rows.append(dict(stage='anchor_gradient',arm='anchor',state_id=sid,objective_parts=dict(response_loss=1. if i<9 else 0.),response_gradient=g.tolist(),guard_gradient=np.zeros_like(p).tolist()))
   bio=io.BytesIO();np.savez(bio,P=p);head=bio.getvalue();bio=io.BytesIO();np.savez(bio,direction=np.zeros_like(p));old=bio.getvalue();zpath=root/'synthetic_upstream.zip'
   with zipfile.ZipFile(zpath,'w') as z:
    z.writestr('ALL1848_RECORDS.jsonl','\n'.join(json.dumps(r) for r in rows));z.writestr('ANCHOR_C163_FIXED_FINAL_HEAD.npz',head);z.writestr('ONE_FIXED_DIRECTION.npz',old)
   paths={k:zpath for k in INPUT_KEYS};c=contract();c.update(production=False,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});fresh=dict(complete=True,errors=[],parent_decision='SYNTHETIC_C165_UPSTREAM_STUB',native_summaries={'anchor':{'0':{'passed':79},'1':{'passed':79}}})
   with contextlib.redirect_stdout(io.StringIO()),patch('worker166.run',return_value=fresh),patch.object(worker167,'bind166',return_value=dict(decision='SYNTHETIC_C166_UPSTREAM_STUB')):
    s=worker167.run(paths,root/'out',c,parent_loader=lambda *a:None)
   self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['all_twelve_originals_independently_bound']);self.assertEqual(s['linear_program_calls_attempted'],2);self.assertTrue(s['new_offline_direction_optimization']);self.assertTrue(s['strict_recorded_raw_common_descent']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['guard_preservation_certified']);self.assertEqual(s['new_model_forwards'],0)
   for n in ('LP_PRIMAL_RESULT.json','LP_DUAL_RESULT.json','BOTH_LP_RESULTS.json','EXACT_RATIONAL_CERTIFICATE.json','DIRECTION_CERTIFICATE_ARRAYS.npz'):self.assertTrue((root/'out'/n).is_file())
   self.assertEqual(s['optimal_margin_lower']['numerator'],'1');self.assertEqual(s['active_response_loss_rows'],9)
if __name__=='__main__':unittest.main(verbosity=2)
