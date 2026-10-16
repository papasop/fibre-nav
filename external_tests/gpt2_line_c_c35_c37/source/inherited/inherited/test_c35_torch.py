"""Synthetic tensor import, compiled forward and80-state pipeline; no GPT-2 claim."""
from pathlib import Path
import sys
from c35_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
import contextlib,io,json,tempfile,unittest,zipfile,types
import numpy as np
import torch
from c35_protocol import *
from c35_parent import prepare,validate
from c35_fit import fit_all,check_frozen,records,input_data
from c35_writer import ComposedWriter
from c31_runtime import load,save_tensor,vector,tensor_hash
from c31_protocol import specs as old_specs
from worker_c34 import saved_state
from worker_c35 import main,proposals,evaluate,prepare_new,counts
from c35_math import predict,geometric

class SyntheticCore(torch.nn.Module):
    def forward(self,x,r):
        out=torch.zeros((len(x),36864),device=x.device);out[:,:6]=x[:,:6];return out

class MockRuntime:
    instances=[]
    def __init__(self,out,root):
        self.out=Path(out);self.device=torch.device('cpu');self.writer=SyntheticCore();self.evaluation_calls=0;self.jcalls=0;self.frozen=sha(self.out/'WRITER_FROZEN.json');self.instances.append(self)
    def sync(self):pass
    def invariant(self):require(sha(self.out/'WRITER_FROZEN.json')==self.frozen,'Synthetic writer identity changed')
    def evaluate(self,theta,target,start=None):
        self.evaluation_calls+=1;e=theta[:6].double().tolist();drift=max(abs(x) for x in e)
        gates={g:True for g in GATES if start is not None or g not in ('output_flip','unselected_preserved')};gates['response']=drift<=.002
        return {'theta_sha256':tensor_hash(theta),'bits':list(target),'response_residual':e,'drift':drift,'kl':.001,
                'evaluation':{'exact_margins':[.1]*4,'heldout_margins':[.1]*12},'gates':gates,'pass':all(gates.values())}
    def jacobian(self,theta,path):
        if Path(path).exists():return load(path)['J'].numpy()
        self.jcalls+=1;J=np.zeros((6,36864));J[:,:6]=np.eye(6);save_tensor(path,{'J':torch.from_numpy(J),'theta_sha256':tensor_hash(theta),'preparation_geometry_only':True});return J
    def correct_preparation(self,initial,folder):return initial,[],0.
    def fresh_input(self,theta,desired):
        route=int(round(float(theta[7])))//12;x=torch.zeros(3147);x[:6]=.05+float(theta[16])*.1
        extra=torch.zeros(70);extra[:6]=theta[:6]/.002
        return {'features':torch.zeros(3077),'extras':extra,'input':x,'route':torch.tensor(route)}

def fixture_archive(path):
    with tempfile.TemporaryDirectory() as d:
        r=Path(d);pins=read(ROOT/'C34_SOURCE_PINS.json');c34settings=read(ROOT/'C34_SETTINGS.json')
        dump(r/'WRITER_FROZEN.json',{'synthetic':True});wf=sha(r/'WRITER_FROZEN.json')
        fixed={'new_training':old_specs('new_training'),'prospective':old_specs('prospective'),'old_inspected16_used':False,'resampling_allowed':False};dump(r/'FIXED_STATE_SPECS.json',fixed)
        dump(r/'identity.json',{'protocol':PARENT_PROTOCOL,'settings':c34settings,'source_hashes':pins})
        for n in ('MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt','C31_SUMMARY.json','C31_IDENTITY.json','C31_RUNTIME.json'):
            if n.endswith('.json'):dump(r/'parent'/n,{'synthetic':True})
            else:save_tensor(r/'parent'/n,{'synthetic':torch.tensor(1.)})
        for j,s in enumerate(fixed['new_training']+fixed['prospective']):
            folder=r/'states'/s['cohort']/s['id'];prep=torch.zeros(36864);prep[7]=j
            x=torch.zeros(3147);x[:6]=.02 if j%4==0 else (.06 if j%4!=3 else -.07)
            raw=prep.clone();raw[:6]=x[:6]*SETTINGS['output_scale']
            inp={'features':torch.zeros(3077),'extras':torch.zeros(70),'input':x,'route':torch.tensor(CASES.index(s['case'])),
                 'spec':s,'prepared_sha256':tensor_hash(prep),'desired_bit':STARTS[s['case']][0]^STARTS[s['case']][1]}
            save_tensor(folder/'INPUT.pt',inp);save_tensor(folder/'prepared.pt',{'theta':prep,'spec':s})
            save_tensor(folder/'RAW_WRITTEN.pt',{'theta':raw,'prepared_sha256':tensor_hash(prep),'input_sha256':tensor_hash(x)})
            start=STARTS[s['case']];target=start[:3]+[start[0]^start[1]]
            rt=MockRuntime.__new__(MockRuntime);rt.evaluation_calls=0;m=rt.evaluate(raw,target,start)
            dump(folder/'COMPLETE.json',{'spec':s,'writer_frozen_sha256':wf,'preparation_valid':True,'generation_retries':0,'post_write_correction_steps':0,
                 'preparation':{'theta_sha256':tensor_hash(prep)},'raw_written':m,
                 'files':{n:sha(folder/n) for n in ('INPUT.pt','prepared.pt','RAW_WRITTEN.pt')}})
            dump(r/'evaluation'/s['id']/'COMPLETE.json',{'arms':{'raw':{'metrics':m}}})
        for case in CASES:
            J=np.zeros((6,36864));J[:,:6]=np.eye(6);folder=r/'basis'/case
            save_tensor(folder/'BASES.pt',{'response6':torch.from_numpy(J.T.copy()),'common1':torch.from_numpy(J.sum(0)[:,None].copy())})
            for sid in [x for x in SETTINGS['prior_basis_states'] if x.startswith(case+'__')]:
                raw=load(r/'states/new_training'/sid/'RAW_WRITTEN.pt')['theta']
                save_tensor(folder/(sid+'_J.pt'),{'J':torch.from_numpy(J.copy()),'theta_sha256':tensor_hash(raw),'geometry_stage':'C34_saved_raw_written_endpoint'})
        files={p.relative_to(r).as_posix():sha(p) for p in (r/'basis').rglob('*') if p.is_file()}
        dump(r/'BASES_FROZEN.json',{'basis_training_ids':SETTINGS['prior_basis_states'],'frozen_before_nonbasis_endpoint_evaluation':True,'writer_frozen_sha256':wf,'files':files})
        dump(r/'SUMMARY.json',{'protocol':PARENT_PROTOCOL,'complete':True,'decision':'C34_LOW_DIMENSIONAL_ORACLE_COMPARISON_COMPLETE',
          'all64_requested_states_retained':True,'original_C25_C27_writer_unchanged':True,
          'bases_frozen_sha256':sha(r/'BASES_FROZEN.json'),'original_writer_sha256':wf})
        manifest={}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for p in r.rglob('*'):
                if p.is_file():n=p.relative_to(r).as_posix();z.write(p,n);manifest[n]=sha(p)
            for n,h in pins.items():z.write(ROOT/'inherited'/n,'source/'+n);manifest['source/'+n]=h
            z.writestr('MANIFEST.json',json.dumps(manifest))

def setup(r):
    r=Path(r);parent=r/'parent.zip';fixture_archive(parent);out=r/'session/run';out.mkdir(parents=True)
    dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT)})
    prepare(parent,out,ROOT);dump(out/'NEW_STATE_SPECS.json',specs());return parent,out

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_full80_freeze_order_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            p,out=setup(d);h=sha(p)
            class InterruptRuntime(MockRuntime):
                def evaluate(self,*a,**kw):
                    if self.evaluation_calls==7:raise RuntimeError('synthetic mid-audit interruption')
                    return super().evaluate(*a,**kw)
            with self.assertRaisesRegex(RuntimeError,'mid-audit'):main(out,runtime_factory=InterruptRuntime)
            frozen=sha(out/'COEFFICIENTS_FROZEN.json')
            sid=old_specs('new_training')[2]['id'];sealed=sha(out/'proposals'/sid/'POINTS.pt')
            s=main(out,runtime_factory=MockRuntime)
            self.assertEqual(sha(out/'COEFFICIENTS_FROZEN.json'),frozen);self.assertEqual(sha(out/'proposals'/sid/'POINTS.pt'),sealed)
            self.assertTrue(s['complete']);self.assertEqual(sha(p),h);self.assertTrue(s['LC1_local_development_protocol_passed'])
            self.assertEqual([s['cohorts'][g]['selected6']['n_requested'] for g in ('training','inspected_development','new_local')],[48,16,16])
            self.assertEqual(s['cohorts']['new_local']['selected6']['complete_passed'],16);self.assertFalse(s['internal_parameter_update'])
            self.assertEqual(s['online_write_Jacobians'],0)
            def forbidden(*a):raise AssertionError('Completed resume reloaded model')
            self.assertEqual(main(out,runtime_factory=forbidden),s)
    def test_no_model_evaluation_during_composed_proposal(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=setup(d);fit_all(out);rt=MockRuntime(out,ROOT);head=ComposedWriter(rt.writer,records(out),rt.device)
            s=old_specs('new_training')[0];prep=vector(load(out/'states/new_training'/s['id']/'prepared.pt'));inp=input_data(out,s)
            value,p,ready=proposals(rt,head,out,s,prep,inp)
            self.assertEqual(rt.evaluation_calls,0);self.assertTrue(ready.exists());self.assertEqual(set(value['points']),set(ARMS))
            self.assertFalse(value['postwrite_response_used']);self.assertEqual(value['core_forward_calls'],1)
            before=sha(p);v2,_,_=proposals(rt,head,out,s,prep,inp);self.assertEqual(sha(p),before)
    def test_new_generation_requires_frozen_head(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=setup(d);rt=MockRuntime(out,ROOT)
            with self.assertRaises(FileNotFoundError):prepare_new(rt,out,specs()[0])
    def test_frozen_head_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=setup(d);fit_all(out);p=out/'fit/xor_00/HEAD.pt';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):check_frozen(out)
    def test_parent_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=setup(d);p=out/'states/new_training/xor_00__c31_train00/INPUT.pt';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):validate(out,ROOT)
    def test_compiled_torch_matches_numpy_coefficient_prediction(self):
        rng=np.random.default_rng(350050);rec={
          'J_mean':rng.normal(0,1e-4,(6,36864)),'point_center':rng.normal(0,.2,36864),'response_center':rng.normal(0,.002,6),
          'B6':np.pad(np.eye(6),((0,36858),(0,0))),'mean':rng.normal(size=12),'scale':np.ones(12),
          'active':np.ones(12,dtype=bool),'offset':rng.normal(0,.0001,6),'beta':rng.normal(0,.0001,(12,6))}
        head=ComposedWriter(SyntheticCore(),[rec]*4,torch.device('cpu'));pre=torch.tensor(rng.normal(0,.001,6),dtype=torch.float64)
        theta=torch.from_numpy(rng.normal(0,.1,36864).astype(np.float32))
        _,diag=head(theta,torch.zeros(3147),torch.tensor(0),pre)
        p=predict(rec,geometric(rec,theta.double().numpy())[None,:],pre.numpy()[None,:])[0]
        np.testing.assert_allclose(diag['selected6']['predicted_residual'].numpy(),p,atol=1e-12)
        self.assertNotIn('J_mean',dict(head.named_buffers()))
    def test_invalid_preparation_remains_in_denominator(self):
        m={'gates':{g:True for g in GATES},'pass':True,'drift':0.}
        rs=[{'preparation_valid':False,'arms':{a:{'metrics':m,'complete_accepted':False} for a in ARMS}}]
        r=counts(rs);self.assertEqual(r['selected6']['n_requested'],1);self.assertEqual(r['selected6']['complete_passed'],0)
    def test_no_grad_even_when_caller_enables_grad(self):
        rec={'J_mean':np.zeros((6,36864)),'point_center':np.zeros(36864),'response_center':np.zeros(6),
          'B6':np.zeros((36864,6)),'mean':np.zeros(12),'scale':np.ones(12),'active':np.ones(12,dtype=bool),'offset':np.zeros(6),'beta':np.zeros((12,6))}
        h=ComposedWriter(SyntheticCore(),[rec]*4,torch.device('cpu'))
        points,_=h(torch.zeros(36864,requires_grad=True),torch.zeros(3147,requires_grad=True),torch.tensor(0),torch.zeros(6))
        self.assertTrue(all(not t.requires_grad for t in points.values()))
    def test_large_predicted_increment_is_clipped(self):
        rec={'J_mean':np.zeros((6,36864)),'point_center':np.zeros(36864),'response_center':np.ones(6),
          'B6':np.pad(np.eye(6),((0,36858),(0,0))),'mean':np.zeros(12),'scale':np.ones(12),'active':np.ones(12,dtype=bool),'offset':np.zeros(6),'beta':np.zeros((12,6))}
        h=ComposedWriter(SyntheticCore(),[rec]*4,torch.device('cpu'));p,d=h(torch.zeros(36864),torch.zeros(3147),torch.tensor(0),torch.zeros(6))
        self.assertLess(float(d['selected6']['radial_scale']),1.);self.assertAlmostEqual(float(p['selected6'].double().norm()),.05,places=7)
if __name__=='__main__':unittest.main()
