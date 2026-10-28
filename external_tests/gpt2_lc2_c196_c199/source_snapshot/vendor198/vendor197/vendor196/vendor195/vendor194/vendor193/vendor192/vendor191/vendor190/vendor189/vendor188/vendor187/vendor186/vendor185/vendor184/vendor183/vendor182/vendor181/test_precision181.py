"""Precision consensus stays descriptive; neither precision overrides full gates."""
import copy,tempfile,unittest
from core181 import *
from precision181 import reconstruct,bind_csv,audit_precision
from endpoint180 import precision_margins as production_report
class FakeArchive:
 def __init__(self,p):self.root=Path(p)
 def json(self,n):return clean_json((self.root/n).read_bytes())
 def raw(self,n):return (self.root/n).read_bytes()
class Checks(unittest.TestCase):
 def fixture(self):
  ff=[dict(state_id=s,observed=[.001]*6+[.2]*16) for s in IDS];native={}
  for arm in ('initial','final'):
   for rnd in (0,1):native['native_'+arm+str(rnd)]=[dict(state_id=s,metrics=dict(observed=[.001]*6+[.2]*16),qualification=dict(complete_pass=True,failed_gates=[])) for s in IDS]
  return native,copy.deepcopy(ff),copy.deepcopy(ff)
 def test_independent_matches_original_production_report_and_all336_rows(self):
  n,i,f=self.fixture();i[0]['observed'][0]=.002001
  with tempfile.TemporaryDirectory() as t:
   expected=production_report(n,i,f,t);s=audit_precision(FakeArchive(t),dict(endpoint_precision_margins=expected),n,i,f,Path(t)/'independent');self.assertEqual(s['original_precision_rows_independently_reconstructed'],336);self.assertEqual(s['by_round']['0']['resolved_to_both_response_pass_ids'],[IDS[0]]);self.assertEqual(s['by_round']['1']['resolved_to_both_response_pass_ids'],[IDS[0]]);self.assertFalse(s['FP64_full_acceptance_evaluated'])
 def test_unchanged_disagreement_not_invented_resolved(self):
  n,i,f=self.fixture();i[0]['observed'][0]=f[0]['observed'][0]=.002001;rr,s=reconstruct(n,i,f);self.assertEqual(s['by_head']['final']['0']['response_gate_disagreements'],1);self.assertEqual(s['by_head']['final']['0']['FP64_response_only_passed'],83)
 def test_both_fail_resolution_not_counted_both_pass(self):
  n,i,f=self.fixture();i[0]['observed'][0]=.002001;f[0]['observed'][0]=.0021
  for t in (0,1):n['native_final'+str(t)][0]['metrics']['observed'][0]=.0021;n['native_final'+str(t)][0]['qualification']=dict(complete_pass=False,failed_gates=['response'])
  with tempfile.TemporaryDirectory() as t:
   expected=production_report(n,i,f,t);s=audit_precision(FakeArchive(t),dict(endpoint_precision_margins=expected),n,i,f,Path(t)/'audit');self.assertEqual(s['by_round']['0']['resolved_to_both_response_pass_ids'],[]);self.assertEqual(s['by_round']['0']['resolved_to_both_response_fail_ids'],[IDS[0]])
 def test_KL_failed_native_not_full_pass_despite_response_agreement(self):
  n,i,f=self.fixture();n['native_final0'][0]['qualification']=dict(complete_pass=False,failed_gates=['KL']);rr,s=reconstruct(n,i,f);q=s['by_head']['final']['0'];self.assertEqual(q['native_response_only_passed'],84);self.assertEqual(q['FP64_response_only_passed'],84);self.assertEqual(q['native_full_pass_and_FP64_response_pass'],83)
 def test_exact_budget_boundary_inclusive_without_extra_margin(self):
  n,i,f=self.fixture();f[0]['observed'][0]=.002;n['native_final0'][0]['metrics']['observed'][0]=.002;rr,s=reconstruct(n,i,f);self.assertEqual(s['by_head']['final']['0']['FP64_response_only_passed'],84);self.assertEqual(s['by_head']['final']['0']['minimum_FP64_response_slack'],0.)
 def test_new_disagreement_remains_visible(self):
  n,i,f=self.fixture();f[0]['observed'][0]=.002001;rr,s=reconstruct(n,i,f);self.assertEqual(s['by_head']['initial']['0']['response_gate_disagreements'],0);self.assertEqual(s['by_head']['final']['0']['disagreement_state_ids'],[IDS[0]])
 def test_nonfinite_rejected(self):
  n,i,f=self.fixture();f[0]['observed'][0]=float('nan')
  with self.assertRaises(ValueError):reconstruct(n,i,f)
 def test_permuted_states_rejected(self):
  n,i,f=self.fixture();f.reverse()
  with self.assertRaises(ValueError):reconstruct(n,i,f)
 def test_missing_original_CSV_record_rejected(self):
  n,i,f=self.fixture()
  with tempfile.TemporaryDirectory() as t:
   production_report(n,i,f,t);z=FakeArchive(t);rows,_=reconstruct(n,i,f);data=z.raw('ALL336_ENDPOINT_PRECISION_MARGINS.csv');data=b'\n'.join(data.splitlines()[:-1])+b'\n'
   with self.assertRaises(ValueError):bind_csv(data,rows)
 def test_CSV_boolean_forgery_rejected(self):
  n,i,f=self.fixture()
  with tempfile.TemporaryDirectory() as t:
   production_report(n,i,f,t);rows,_=reconstruct(n,i,f);data=FakeArchive(t).raw('ALL336_ENDPOINT_PRECISION_MARGINS.csv').replace(b'False',b'True',1)
   with self.assertRaises(ValueError):bind_csv(data,rows)
if __name__=='__main__':unittest.main(verbosity=2)
