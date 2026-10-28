"""Full-size synthetic C155-C158 binding; archived gradients remain measurements, not derivatives."""
import contextlib,copy,io,shutil,tempfile,unittest
from core159 import *
import test_archive158 as parent_archive
from fixture156 import pack
from fixture158 import Analytic
from worker158 import run as run158
from inputs159 import load,audit158
from baseline158 import record
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.parent=parent_archive.Checks;cls.parent.setUpClass();cls.root=cls.parent.root;cls.context=copy.deepcopy(cls.parent.context);cls.context['failed157']=dict(decision='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');cls.paths=dict(cls.parent.paths);cls.parent_config=cls.parent.c;out=cls.root/'c158';context=cls.context
  class Backend(Analytic):
   def __init__(self,mr,ctx,points,directions,calls):
    super().__init__(mr,ctx,points,directions,calls);mr=Path(mr);mr.mkdir(parents=True,exist_ok=True);self.baseline_records={};first=ctx['baseline_logits']['FP64_logits'];pairs=[('native_logits','NATIVE_FP32_BASELINE_LOGITS',ctx['baseline_logits']['FP32_logits_cast64'].astype(np.float32),ctx['baseline_logits']['FP32_logits_cast64'],'cross_run_descriptive_only'),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE',np.asarray(ctx['baseline']['FP32_cached_response_reference'],np.float32),np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'cross_run_descriptive_only'),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS',first,first,'cross_run_descriptive_only'),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',first,first,'within_run_repeat')]
    for k,n,a,b,role in pairs:self.baseline_records[k]=record(mr,n,a,b,role)
    self.references=dict(current_FP64=first[:,ctx['modelpins']['response_ids']].mean(0),archived_FP64=np.asarray(ctx['baseline']['FP64_response_reference'],float),archived_FP32=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float));self.reference_ids={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in self.references.items()};dump(mr/'REFERENCES_FROZEN_BEFORE_FOCUS.json',dict(protocol=previous.PROTOCOL,model_forwards_so_far=3,state_probes_so_far=0,references={n:a.tolist() for n,a in self.references.items()},reference_tensor_sha256=self.reference_ids));dump(mr/'BASELINE_RUNTIME_PROVENANCE.json',dict(archived_C155_MODEL_PINS=ctx['modelpins'],synthetic_fixture=True))
  with contextlib.redirect_stdout(io.StringIO()):s=run158(cls.paths,out,cls.parent_config,Backend,lambda *a:copy.deepcopy(context))
  need(s['complete'],'Synthetic C158 must complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor158',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()});b=cls.root/'C158.zip';pack(out,b);cls.paths['C158']=b;cls.c=contract();cls.c.update(production=False,parent=cls.parent_config,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
 @classmethod
 def tearDownClass(cls):cls.parent.tearDownClass()
 def test_complete_four_originals_and_all24_C158_records(self):
  with contextlib.redirect_stdout(io.StringIO()):data=load(self.paths,self.root/'bound159',self.c)
  self.assertEqual(len(data['data']['records']),1530);self.assertEqual(len(data['data']['points']),588);self.assertEqual(data['completed158']['decision'],'C158_SAME_RUN_TWO_STATE_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED');self.assertEqual(len(data['completed158']['endpoints']),4)
 def test_rebound_C158_gradient_difference_summary_forgery_blocks(self):
  out=self.root/'forged158';shutil.copytree(self.root/'c158',out);p=out/'SUMMARY.json';s=clean_json(p.read_bytes());s['diagnostics'][0]['scales'][0]['central_minus_AD_Linf']+=1.;dump(p,s);b=self.root/'forged158.zip';pack(out,b)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit158(b,self.context,self.root/'check_forged158',self.c)
 def test_rebound_C158_reference_tensor_forgery_blocks(self):
  out=self.root/'forged_ref158';shutil.copytree(self.root/'c158',out);p=out/'model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json';s=clean_json(p.read_bytes());s['references']['archived_FP64'][0]+=.01;arr=np.asarray(s['references']['archived_FP64'],float);s['reference_tensor_sha256']['archived_FP64']=hashlib.sha256(arr.tobytes()).hexdigest();dump(p,s);b=self.root/'forged_ref158.zip';pack(out,b)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit158(b,self.context,self.root/'check_ref158',self.c)
if __name__=='__main__':unittest.main(verbosity=2)
