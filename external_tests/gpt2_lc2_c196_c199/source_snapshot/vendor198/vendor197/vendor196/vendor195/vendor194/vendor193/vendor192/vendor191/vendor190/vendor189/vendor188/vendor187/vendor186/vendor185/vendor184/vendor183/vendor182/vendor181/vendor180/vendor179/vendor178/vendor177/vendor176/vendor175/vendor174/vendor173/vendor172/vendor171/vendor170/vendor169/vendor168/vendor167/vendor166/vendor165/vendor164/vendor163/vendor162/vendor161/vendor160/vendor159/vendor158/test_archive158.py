"""Full-size synthetic original-archive integration; no GPT-2 measurement claims."""
import contextlib,copy,io,shutil,tempfile,unittest,zipfile
from core158 import *
from inputs158 import load,audit_failed157
from test_archive157 import Checks as ParentChecks
from inputs157 import load as load157
from baseline157 import record_comparison
from fixture156 import pack
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ParentChecks.setUpClass();cls.root=ParentChecks.root;cls.paths=dict(ParentChecks.paths)
        with contextlib.redirect_stdout(io.StringIO()):cls.context=load157(ParentChecks.paths,cls.root/'context157',ParentChecks.c)
        out=cls.root/'c157';out.mkdir();dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',previous.design());previous.freeze(cls.context['data'],out)
        shutil.copytree(ROOT/'vendor157',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()});mr=out/'model_runtime';mr.mkdir();reports={}
        with contextlib.redirect_stdout(io.StringIO()):
            for label,expected in [('NATIVE_FP32_BASELINE_LOGITS',cls.context['baseline_logits']['FP32_logits_cast64']),('NATIVE_FP32_CACHED_RESPONSE',np.asarray(cls.context['baseline']['FP32_cached_response_reference'],float))]:
                reports[label]=record_comparison(mr,label,expected.astype(np.float32)+np.float32(.0001),expected)
        dump(mr/'BASELINE_RUNTIME_PROVENANCE.json',dict(archived_C155_MODEL_PINS=cls.context['modelpins'],synthetic_fixture=True))
        summary=dict(protocol=previous.PROTOCOL,complete=False,decision='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION',delivery_revision='R2_BASELINE_RECORDING_ONLY',synthetic_fixture=True,stage='pinned_model_and_baseline_replay',new_model_forwards=1,new_backward_calls=0,focus_training_states_evaluated=0,new_training_steps=0,new_head_fits=0,excluded_input_states_evaluated=0,LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False,thresholds_changed=False,parent_decision_reclassified=False,original_inputs_unchanged=dict(C155=True,C156=True),baseline_replay_diagnostics=reports,errors=['Synthetic native replay mismatch; not an actual model run.'])
        dump(out/'SUMMARY.json',summary);p=cls.root/'C157.zip';pack(out,p);cls.paths['C157']=p;cls.c=contract();cls.c.update(production=False,parent=ParentChecks.c,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
    @classmethod
    def tearDownClass(cls):ParentChecks.tearDownClass()
    def test_complete_three_original_binding_and_failed_C157_arithmetic(self):
        with contextlib.redirect_stdout(io.StringIO()):data=load(self.paths,self.root/'binding158',self.c)
        self.assertEqual(len(data['data']['records']),1530);self.assertEqual(len(data['data']['points']),588);self.assertEqual(data['failed157']['decision'],'C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION')
    def test_rebound_diagnostic_forgery_is_recomputed(self):
        out=self.root/'tampered157';shutil.copytree(self.root/'c157',out);p=out/'model_runtime/NATIVE_FP32_BASELINE_LOGITS.npz'
        with np.load(p,allow_pickle=False) as z:a=z['actual'];b=z['archived']
        a=a.copy();a.flat[0]+=.01;np.savez_compressed(p,actual=a,archived=b);new=self.root/'tampered157.zip';pack(out,new)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit_failed157(new,self.context,self.root/'tampered_check',self.c)
    def test_rebound_point_forgery_blocks(self):
        out=self.root/'tampered_point';shutil.copytree(self.root/'c157',out);p=out/'ALL14_FP64_CHORD_POINTS.npz'
        with np.load(p,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        arrays[next(iter(arrays))][0]+=.01;np.savez_compressed(p,**arrays);seals=clean_json((out/'SEALED_BEFORE_MODEL.json').read_bytes());seals['artifact_seals'][p.name]=sha(p);dump(out/'SEALED_BEFORE_MODEL.json',seals);new=self.root/'tampered_point.zip';pack(out,new)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit_failed157(new,self.context,self.root/'tampered_point_check',self.c)
if __name__=='__main__':unittest.main(verbosity=2)
