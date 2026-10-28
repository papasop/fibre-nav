"""Full-size synthetic five-archive input binding; no real model measurement claim."""
import contextlib,copy,io,shutil,unittest
from core160 import *
import test_archive159 as parent_archive
from inputs159 import load as load159
from inputs160 import load,audit159
from worker159 import run as run159
from fixture159 import Analytic
from fixture156 import pack
from baseline158 import record
from map159 import chain_arithmetic
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.parent=parent_archive.Checks;cls.parent.setUpClass();cls.root=cls.parent.root;cls.paths=dict(cls.parent.paths);cls.pc=cls.parent.c
  with contextlib.redirect_stdout(io.StringIO()):ctx=load159(cls.paths,cls.root/'context159',cls.pc)
  cls.context=ctx;out=cls.root/'c159'
  class Backend(Analytic):
   def __init__(self,mr,context,points,directions,calls):
    super().__init__(mr,context,points,directions,calls);mr=Path(mr);mr.mkdir(parents=True,exist_ok=True);first=context['baseline_logits']['FP64_logits'];self.references=dict(current_FP64=first[:,context['modelpins']['response_ids']].mean(0),archived_FP64=np.asarray(context['baseline']['FP64_response_reference'],float),archived_FP32=np.asarray(context['baseline']['FP32_cached_response_reference'],float));self.ids={n:array_digest(a) for n,a in self.references.items()};self.baseline_records={}
    for key,n,a,b,role in [('native_logits','NATIVE_FP32_BASELINE_LOGITS',context['baseline_logits']['FP32_logits_cast64'].astype(np.float32),context['baseline_logits']['FP32_logits_cast64'],'cross_run_descriptive_only'),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE',self.references['archived_FP32'].astype(np.float32),self.references['archived_FP32'],'cross_run_descriptive_only'),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS',first,first,'cross_run_descriptive_only'),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',first,first,'within_run_repeat')]:self.baseline_records[key]=record(mr,n,a,b,role)
    dump(mr/'REFERENCES_FROZEN_BEFORE_FOCUS.json',dict(protocol=previous.PROTOCOL,references={n:a.tolist() for n,a in self.references.items()},reference_tensor_sha256={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in self.references.items()},model_forwards_so_far=3,state_probes_so_far=0))
   def ad(self,sid):
    self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;self.calls['backward_calls']+=9;self.calls['backward_attempts']+=9;absolute=self.points[(sid,'zero')][:6];g=self.directions[sid][:6];u=self.c['V']@(self.c[sid+'__raw']-self.c[sid+'__prepared']);chain={};losses={}
    for n,b in self.references.items():
     rr=absolute-b;i=int(np.argmax(abs(rr)));factor=2*max(abs(rr[i])/.0018-1,0)*np.sign(rr[i])/.0018;ge=np.stack([factor*self.c['V'][:,i]*u,factor*self.c['V'][:,i]*np.roll(u,1)]);gp=self.c[sid+'__weights'][:,None,None]*ge[None];losses[n]=float(factor*g[i]);chain[n]=chain_arithmetic(losses[n],ge,gp,self.c,sid)
    return dict(state_id=sid,label='zero',FP64_point_sha256=point_hash(self.points[(sid,'zero')]),absolute_response=absolute.tolist(),coordinate_derivatives=g.tolist(),response_loss_derivatives=losses,chain_pullbacks=chain)
  with contextlib.redirect_stdout(io.StringIO()):s=run159(cls.paths,out,cls.pc,Backend,lambda *a:copy.deepcopy(ctx))
  need(s['complete'],'Synthetic C159 must complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor159',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()});p=cls.root/'C159.zip';pack(out,p);cls.paths['C159']=p;cls.c=contract();cls.c.update(production=False,parent=cls.pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
 @classmethod
 def tearDownClass(cls):cls.parent.tearDownClass()
 def test_all_five_archives_and_C159_full_chain_binding(self):
  with contextlib.redirect_stdout(io.StringIO()):ctx=load(self.paths,self.root/'full160',self.c)
  self.assertEqual(len(ctx['data']['records']),1530);self.assertEqual(len(ctx['data']['points']),588);self.assertEqual(ctx['completed159']['decision'],'C159_COEFFICIENT_CHAIN_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED')
 def test_rebound_C159_gradient_summary_forgery_blocks(self):
  out=self.root/'forged159';shutil.copytree(self.root/'c159',out);p=out/'SUMMARY.json';s=clean_json(p.read_bytes());s['diagnostics'][0]['chain_pullbacks']['current_FP64']['explicit_dot_minus_alpha']+=1.;dump(p,s);b=self.root/'forged159.zip';pack(out,b)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit159(b,self.context,self.root/'check_forged159',self.c)
 def test_rebound_C159_point_forgery_blocks(self):
  out=self.root/'forged_point159';shutil.copytree(self.root/'c159',out);p=out/'ALL14_FP64_CHAIN_POINTS.npz'
  with np.load(p,allow_pickle=False) as z:ar={n:z[n] for n in z.files}
  ar[next(iter(ar))][0]+=.01;np.savez_compressed(p,**ar);b=self.root/'forged_point159.zip';pack(out,b)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):audit159(b,self.context,self.root/'check_point159',self.c)
if __name__=='__main__':unittest.main(verbosity=2)
