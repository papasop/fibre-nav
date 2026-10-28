"""Independent replay arithmetic controls; no model or archive measurements."""
import copy,tempfile,unittest
from core193 import *
from replay193 import native_pair,replay_report

def fixture():
 old=[];fp=[]
 for sid in IDS:
  old.append(dict(state_id=sid,metrics=dict(observed=[.001]*6+[.2]*16,KL=0.,gates={'content':True,'response':True},theta_sha256='0'*64),qualification=dict(complete_pass=True,failed_gates=[])))
  fp.append(dict(state_id=sid,observed=[.001]*6+[.2]*16,KL=0.,FP64_point_sha256='1'*64))
 ctx=dict(C190_final_native={'0':copy.deepcopy(old),'1':copy.deepcopy(old)},C190_final_FP64=copy.deepcopy(fp));native={}
 for name in ('native_initial0','native_initial1','native_final0','native_final1'):native[name]=copy.deepcopy(old)
 for rnd in (0,1):
  for i,r in enumerate(native['native_initial'+str(rnd)]):r['C190_final_native_replay']=archive155.repeat_check(ctx['C190_final_native'][str(rnd)][i]['metrics'],r['metrics'])
 initial=copy.deepcopy(fp)
 for r in initial:r['C190_final_FP64_replay']=dict(observed_Linf=0.,KL_absolute=0.,within_original_replay_tolerance=True)
 return dict(native=native,frames=[initial]),ctx
class Checks(unittest.TestCase):
 def run_report(self,a,c):
  with tempfile.TemporaryDirectory() as t:return replay_report(a,c,t)
 def test_same_point_nonzero_observable_difference_can_pass_tolerance(self):
  a,c=fixture();x=c['C190_final_native']['0'][0]['metrics'];y=copy.deepcopy(x);y['observed'][0]+=.0001;r,d=native_pair(x,y);self.assertTrue(r['same_point_identity']);self.assertTrue(r['within_replay_tolerance']);self.assertFalse(d['observables_exactly_equal']);self.assertGreater(r['observed_error_Linf'],0);self.assertEqual(r,archive155.repeat_check(x,y))
 def test_equal_observables_do_not_imply_same_point(self):
  a,c=fixture();x=c['C190_final_native']['0'][0]['metrics'];y=copy.deepcopy(x);y['theta_sha256']='2'*64;r,d=native_pair(x,y);self.assertTrue(d['observables_exactly_equal']);self.assertFalse(r['same_point_identity'])
 def test_metric_gate_change_blocks_registered_replay(self):
  a,c=fixture();x=c['C190_final_native']['0'][0]['metrics'];y=copy.deepcopy(x);y['gates']['response']=False;r,d=native_pair(x,y);self.assertFalse(r['within_replay_tolerance'])
 def test_original_native_thresholds_kept(self):
  a,c=fixture();x=c['C190_final_native']['0'][0]['metrics'];x['observed']=[0.]*22;y=copy.deepcopy(x);y['observed'][0]=.0002;y['KL']=5e-6;self.assertTrue(native_pair(x,y)[0]['within_replay_tolerance']);y['observed'][0]=np.nextafter(.0002,1.);self.assertFalse(native_pair(x,y)[0]['within_replay_tolerance'])
 def test_KL_above_original_tolerance_blocks(self):
  a,c=fixture();x=c['C190_final_native']['0'][0]['metrics'];y=copy.deepcopy(x);y['KL']=5.0001e-6;self.assertFalse(native_pair(x,y)[0]['within_replay_tolerance'])
 def test_full168_native84_FP64_and_no_new_acceptance(self):
  a,c=fixture();s=self.run_report(a,c);self.assertEqual(s['native_pairs'],168);self.assertEqual(s['FP64_pairs'],84);self.assertTrue(s['all_native_same_point_identity']);self.assertTrue(s['all_native_observables_exactly_equal']);self.assertFalse(s['acceptance_thresholds_changed']);self.assertFalse(s['LC2_closed']);self.assertTrue(s['no_accuracy_bound_claim'])
 def test_nonzero_cross_run_shift_separate_from_same_run_training(self):
  a,c=fixture();rr=a['native']['native_initial0'][0];rr['metrics']['observed'][0]+=.0001;rr['C190_final_native_replay']=archive155.repeat_check(c['C190_final_native']['0'][0]['metrics'],rr['metrics']);s=self.run_report(a,c);self.assertEqual(s['native_pairs_with_nonzero_observable_difference'],1);self.assertFalse(s['all_native_observables_exactly_equal']);self.assertGreater(s['maxima_by_round']['0']['cross_run_anchor_maximum_shift'],0);self.assertLess(s['maxima_by_round']['0']['same_run_training_maximum_change'],0)
 def test_forged_original_replay_metadata_blocks(self):
  a,c=fixture();a['native']['native_initial0'][0]['C190_final_native_replay']['observed_error_Linf']=.00001
  with self.assertRaises(ValueError):self.run_report(a,c)
 def test_wrong_point_hash_blocks_whole_report(self):
  a,c=fixture();rr=a['native']['native_initial0'][0];rr['metrics']['theta_sha256']='2'*64;rr['C190_final_native_replay']=archive155.repeat_check(c['C190_final_native']['0'][0]['metrics'],rr['metrics'])
  with self.assertRaises(ValueError):self.run_report(a,c)
 def test_FP64_original_tolerance_not_silently_relaxed(self):
  a,c=fixture();rr=a['frames'][0][0];rr['observed'][0]+=.00001;rr['C190_final_FP64_replay']=dict(observed_Linf=float(abs(rr['observed'][0]-.001)),KL_absolute=0.,within_original_replay_tolerance=False)
  with self.assertRaises(ValueError):self.run_report(a,c)
 def test_FP64_point_identity_report_does_not_add_a_gate(self):
  a,c=fixture();c['C190_final_FP64'][0]['FP64_point_sha256']='3'*64;self.assertTrue(self.run_report(a,c)['all_FP64_registered_replay_rules_passed'])
 def test_wrong_or_missing_registry_blocks(self):
  a,c=fixture();a['native']['native_initial1'].reverse()
  with self.assertRaises(ValueError):self.run_report(a,c)
  a,c=fixture();c['C190_final_FP64'].pop()
  with self.assertRaises(ValueError):self.run_report(a,c)
 def test_nonfinite_values_rejected(self):
  a,c=fixture();x=c['C190_final_native']['0'][0]['metrics'];y=copy.deepcopy(x);y['KL']=float('nan')
  with self.assertRaises(ValueError):native_pair(x,y)
if __name__=='__main__':unittest.main(verbosity=2)
