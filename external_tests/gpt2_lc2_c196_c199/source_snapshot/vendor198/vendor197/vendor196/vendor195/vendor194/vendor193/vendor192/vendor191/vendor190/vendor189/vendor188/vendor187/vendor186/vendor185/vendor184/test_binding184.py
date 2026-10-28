"""Whole synthetic C183 audit binding, C182 reconstruction and continuation anchor."""
import copy,contextlib,io,tempfile,unittest,zipfile,shutil
from unittest.mock import patch
from core184 import *
from inputs184 import bind183,load
from fixture183 import fixture
from fixture171 import pack
from worker183 import run as run183
from audit183 import audit
from diagnostics183 import diagnostics
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cp,cls.ctx,src=fixture(cls.root/'C182_fixture');cls.c=contract();cls.c.update(production=False,parent=cp,pins=copy.deepcopy(cp['pins']))
  with contextlib.redirect_stdout(io.StringIO()):s=run183(cls.paths,cls.root/'C183',cp,lambda *a:copy.deepcopy(cls.ctx))
  need(s['complete'],'Synthetic C183 audit generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor183',cls.root/'C183/source',ignore=shutil.ignore_patterns('__pycache__'));cls.paths['C183']=pack(cls.root/'C183',cls.root/'C183.zip');cls.c['pins']['C183']=dict(path=str(cls.paths['C183']),sha256=sha(cls.paths['C183']))
  cls.recon=cls.root/'independent_C183'
  with contextlib.redirect_stdout(io.StringIO()):cls.a=audit(cls.paths['C182'],cls.ctx,cls.recon/'C182_arithmetic',cp)
  cls.d=diagnostics(cls.a,cls.recon/'diagnostics')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def changed(self,name,fn):
  with zipfile.ZipFile(self.paths['C183']) as z:m={n:z.read(n) for n in z.namelist()}
  m[name]=fn(m[name]);m['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in m.items() if n!='RESULT_MANIFEST.json'})+'\n').encode();p=self.root/(self._testMethodName+'.zip')
  with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
   for n,v in m.items():z.writestr(n,v)
  return p
 def j(self,fn):
  def edit(b):v=clean_json(b);fn(v);return (json.dumps(v)+'\n').encode()
  return edit
 def reject(self,n,fn):
  p=self.changed(n,fn)
  with self.assertRaises((ValueError,KeyError)):bind183(p,self.a,self.d,self.root/(self._testMethodName+'_out'),self.c,self.recon)
 def test_full_C183_summary_source_diagnostics_and_all_reconstructed_artifacts(self):
  with archive155.prior.no_models(),safe.no_solvers():r=bind183(self.paths['C183'],self.a,self.d,self.root/'complete',self.c,self.recon)
  self.assertEqual(r['decision'],prior.DECISION);self.assertTrue(r['C183_scientific_summary_reconstructed']);self.assertGreaterEqual(len(r['reconstructed_artifacts_bound']),29);self.assertEqual(sum(n.startswith('C182_arithmetic/precision/') for n in r['reconstructed_artifacts_bound']),5);self.assertEqual(r['new_LP_calls'],0)
 def test_loader_adopts_C182_step8_not_C178_or_intermediate(self):
  import inputs184
  with patch.object(inputs184,'load182',lambda *a:copy.deepcopy(self.ctx)),contextlib.redirect_stdout(io.StringIO()):ctx=load(self.paths,self.root/'full_loader',self.c)
  np.testing.assert_array_equal(ctx['C182_final_head'],self.a['heads'][-1]);self.assertFalse(np.array_equal(ctx['C182_final_head'],ctx['C178_final_head']));self.assertEqual(ctx['C182_final_FP64'],self.a['frames'][-1]);self.assertEqual(ctx['C182_final_native']['1'],self.a['native']['native_final1']);self.assertEqual(ctx['C183_receipt']['decision'],prior.DECISION)
 def test_original_C183_source_tamper_with_rebound_manifest_blocks(self):self.reject('source/core183.py',lambda b:b+b'\n# changed\n')
 def test_initial_older_head_as_final_anchor_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('fixed_final_step8_head_digest',v['initial_head_digest'])))
 def test_original_parent_decision_reclassification_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('parent_decision','forged_all_pass')))
 def test_forged_guard_regularizer_diagnosis_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v['diagnostics'].__setitem__('observed_guard_change_from_regularizer_only',not v['diagnostics']['observed_guard_change_from_regularizer_only'])))
 def test_incomplete_C183_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('complete',False)))
 def test_C182_ancestry_tamper_blocks(self):self.reject('ORIGINAL_INPUTS.json',self.j(lambda v:v['C182'].__setitem__('sha256','0'*64)))
 def test_duplicate_diagnostic_tamper_blocks(self):self.reject('diagnostics/DIAGNOSTIC_SUMMARY.json',self.j(lambda v:v.__setitem__('LC2_closed',True)))
 def test_reconstructed_history_artifact_tamper_blocks(self):self.reject('C182_arithmetic/ALL8_UPDATE_HISTORY_RECONSTRUCTED.json',self.j(lambda v:v[0]['update'].__setitem__('step_size',.002)))
 def test_reconstructed_csv_tamper_blocks(self):
  def change(b):
   lines=b.decode().splitlines();parts=lines[1].split(',');parts[0]='999';lines[1]=','.join(parts);return ('\n'.join(lines)+'\n').encode()
  self.reject('diagnostics/ALL8_STEP_STATISTICS.csv',change)
 def test_precision_summary_duplicate_tamper_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v['endpoint_precision_reconstruction'].__setitem__('FP64_full_acceptance_evaluated',True)))
 def test_precision_rows_reconstruction_tamper_blocks(self):self.reject('C182_arithmetic/precision/ALL336_PRECISION_ROWS_INDEPENDENT.json',self.j(lambda v:v[0].__setitem__('native_response_slack',.5)))
 def test_precision_transition_reconstruction_tamper_blocks(self):self.reject('C182_arithmetic/precision/ALL168_PRECISION_TRANSITIONS.json',self.j(lambda v:v[0].__setitem__('native_rescued',not v[0]['native_rescued'])))
 def test_recorded_Jacobian_count_cannot_be_hidden(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('archived_response_Jacobian_blocks_checked',0)))
 def test_unknown_solver_internal_count_remains_null(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('original_solver_internal_linear_system_count',8)))
 def test_activity_summary_reclassification_blocks(self):self.reject('SUMMARY.json',self.j(lambda v:v['diagnostics']['activity_set_dynamics'].__setitem__('final_active_count',999)))
 def test_activity_transition_artifact_tamper_blocks(self):self.reject('diagnostics/ALL672_ACTIVITY_TRANSITIONS.json',self.j(lambda v:v[0].__setitem__('state_id','foreign_state')))
 def test_activity_summary_duplicate_tamper_blocks(self):self.reject('diagnostics/ACTIVITY_SET_SUMMARY.json',self.j(lambda v:v.__setitem__('final_active_count',999)))
 def test_true_model_measurements_cannot_be_invented(self):self.reject('SUMMARY.json',self.j(lambda v:v.__setitem__('model_outputs_independently_recomputed',True)))
if __name__=='__main__':unittest.main(verbosity=2)
