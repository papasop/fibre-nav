"""Synthetic64-state actual-tensor pipeline; no real GPT-2/A100 run."""
import contextlib,io,tempfile,unittest,zipfile,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from c34_protocol import *
from c34_parent import prepare,probe,validate,resume_score
from c34_runtime import Runtime,save_tensor,load,vector,tensor_hash
import c31_runtime as old_runtime
from worker_c34 import main,build_bases,check_frozen,evaluate_state,measure_raw,check_metrics
from c31_protocol import specs

ROOT=Path(__file__).resolve().parent

def fixture_archive(path):
    """Authentic source pins, internally consistent synthetic tensors/endpoints."""
    with tempfile.TemporaryDirectory() as d:
        r=Path(d);pins=read(ROOT/'C31_SOURCE_PINS.json');writer={'synthetic_writer':True};dump(r/'WRITER_FROZEN.json',writer);wf=sha(r/'WRITER_FROZEN.json')
        fixed={'new_training':specs('new_training'),'prospective':specs('prospective'),'old_inspected16_used':False,'resampling_allowed':False};dump(r/'FIXED_STATE_SPECS.json',fixed)
        dump(r/'identity.json',{'protocol':PARENT_PROTOCOL,'settings':read(ROOT/'C31_SETTINGS.json'),'source_hashes':pins})
        dump(r/'PREDICTORS_FROZEN.json',{'writer_frozen_sha256':wf})
        dump(r/'RUNTIME.json',{'synthetic':True})
        for n in ('MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt'):
            p=r/'parent'/n;p.parent.mkdir(exist_ok=True)
            if n.endswith('.json'):dump(p,{'synthetic':True})
            else:save_tensor(p,{'synthetic':torch.tensor(1.)})
        counts={'new_training':0,'prospective':0}
        for j,s in enumerate(fixed['new_training']+fixed['prospective']):
            folder=r/'states'/s['cohort']/s['id'];folder.mkdir(parents=True)
            raw=torch.zeros(36864);raw[7]=j
            raw[:6]=.001 if j%4==0 else (.003 if j%4!=3 else -.004)
            if j%4==1:raw[:6]+=torch.tensor([-.003,.003,-.003,.003,-.003,.003])
            prep=raw.clone();prep[:6]=0
            inp={'input':torch.zeros(3147),'features':torch.zeros(3077),'route':torch.tensor(CASES.index(s['case'])),
                 'spec':s,'desired_bit':STARTS[s['case']][0]^STARTS[s['case']][1],'prepared_sha256':tensor_hash(prep)}
            save_tensor(folder/'INPUT.pt',inp);save_tensor(folder/'prepared.pt',{'theta':prep,'spec':s})
            save_tensor(folder/'PERTURBATION.pt',{'theta':torch.zeros(36864)})
            save_tensor(folder/'RAW_WRITTEN.pt',{'theta':raw,'prepared_sha256':tensor_hash(prep),'input_sha256':tensor_hash(inp['input'])})
            valid=j!=48;m=MockRuntime.metric(raw,s['case']);counts[s['cohort']]+=int(valid and m['pass'])
            dump(folder/'COMPLETE.json',{'spec':s,'writer_frozen_sha256':wf,'preparation_valid':valid,
                 'generation_retries':0,'post_write_correction_steps':0,'raw_written':m,'preparation':{'theta_sha256':tensor_hash(prep)},
                 'files':{n:sha(folder/n) for n in ('INPUT.pt','prepared.pt','RAW_WRITTEN.pt','PERTURBATION.pt')}})
        dump(r/'SUMMARY.json',{'protocol':PARENT_PROTOCOL,'complete':True,'decision':'C31_RESIDUAL_COVERAGE_COMPARISON_COMPLETE',
          'C27_writer_unchanged':True,'all48_training_and16_prospective_states_retained':True,
          'frozen_predictor_sha256':sha(r/'PREDICTORS_FROZEN.json'),
          'new_training':{'all':{'raw_complete_write_passed':counts['new_training']}},
          'prospective_local_development':{'all':{'raw_complete_write_passed':counts['prospective']}}})
        manifest={}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for p in r.rglob('*'):
                if p.is_file():n=p.relative_to(r).as_posix();z.write(p,n);manifest[n]=sha(p)
            for n,h in pins.items():z.write(ROOT/'inherited'/n,'source/'+n);manifest['source/'+n]=h
            z.writestr('MANIFEST.json',json.dumps(manifest))

class MockRuntime:
    instances=[]
    def __init__(self,out,root):
        self.out=Path(out);self.calls=[];self.jcalls=[];self.frozen=sha(self.out/'WRITER_FROZEN.json');self.instances.append(self)
    @staticmethod
    def metric(theta,case):
        e=theta[:6].double().tolist();drift=max(abs(x) for x in e);gates={g:True for g in GATES};gates['response']=drift<=.002
        start=STARTS[case];target=start[:3]+[start[0]^start[1]]
        return {'theta_sha256':tensor_hash(theta),'response_residual':e,'drift':drift,'kl':.001,'bits':target,
                'evaluation':{'exact_margins':[.1]*4,'heldout_margins':[.1]*12},'gates':gates,'pass':all(gates.values())}
    def evaluate(self,theta,target,start):
        case=CASES[2*start[0]+start[1]];j=int(theta[7]);self.calls.append(j)
        spec=(specs('new_training')+specs('prospective'))[j]
        if spec['id'] not in SETTINGS['basis_training_ids']:require((self.out/'BASES_FROZEN.json').exists(),'Nonbasis state evaluated before basis freeze')
        return self.metric(theta,case)
    def jacobian(self,theta,path):
        self.jcalls.append(int(theta[7]));J=np.zeros((6,36864));J[:,:6]=np.eye(6)
        save_tensor(path,{'J':torch.from_numpy(J),'theta_sha256':tensor_hash(theta),'geometry_stage':'C34_saved_raw_written_endpoint'});return J
    def invariant(self):require(sha(self.out/'WRITER_FROZEN.json')==self.frozen,'Mock writer changed')

def prepared(r):
    p=Path(r)/'parent.zip';fixture_archive(p);out=Path(r)/'session/run';out.mkdir(parents=True)
    dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT)})
    prepare(p,out,ROOT);return p,out

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_full64_pipeline_no_invalid_drop(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            p,out=prepared(d);h=sha(p);s=main(out,runtime_factory=MockRuntime)
            self.assertTrue(s['complete']);self.assertEqual(sha(p),h);self.assertEqual(len(MockRuntime.instances[-1].jcalls),12)
            self.assertEqual([s['cohorts'][n]['raw']['n_requested'] for n in ('basis_training','other_training','inspected_development')],[12,36,16])
            self.assertEqual(s['cohorts']['inspected_development']['response6_oracle']['complete_passed'],15)
            self.assertEqual(s['cohorts']['inspected_development']['response6_oracle']['response_gate_passed_all'],16)
            self.assertLess(s['cohorts']['other_training']['common1_oracle']['complete_passed'],36)
            self.assertFalse(s['LC1_closed']);self.assertTrue(s['privileged_raw_endpoint_response_used_for_coefficients'])
    def test_partial_resume_uses_saved_geometry_and_endpoints(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            p,out=prepared(d);ss=validate(out,ROOT);rt=MockRuntime(out,ROOT/'inherited');build_bases(rt,out,ss,'')
            first=evaluate_state(rt,out,ss[0]);h=sha(out/'BASES_FROZEN.json')
            s=main(out,runtime_factory=MockRuntime)
            self.assertEqual(MockRuntime.instances[-1].jcalls,[]);self.assertEqual(sha(out/'BASES_FROZEN.json'),h)
            self.assertEqual(read(out/'evaluation'/ss[0]['id']/'COMPLETE.json'),first);self.assertTrue(s['complete'])
    def test_completed_resume_does_not_reload_runtime(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=prepared(d);s=main(out,runtime_factory=MockRuntime)
            def forbidden(*a):raise AssertionError('Runtime reloaded on completed resume')
            self.assertEqual(main(out,runtime_factory=forbidden),s)
    def test_parent_bytes_tamper_fails(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=prepared(d);p=out/'states/new_training/xor_00__c31_train00/RAW_WRITTEN.pt';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):validate(out,ROOT)
    def test_basis_tamper_fails(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=prepared(d);ss=validate(out,ROOT);rt=MockRuntime(out,ROOT/'inherited');build_bases(rt,out,ss,'')
            p=out/'basis/xor_00/BASES.pt';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):check_frozen(out)
    def test_missing_gate_or_wrong_bit_rejected(self):
        m=MockRuntime.metric(torch.zeros(36864),'xor_00');m['bits'][3]=1
        with self.assertRaises(ValueError):check_metrics(m,'xor_00')
    def test_raw_replay_mismatch_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            _,out=prepared(d);rt=MockRuntime(out,ROOT/'inherited');original=rt.evaluate
            def wrong(*a):
                m=original(*a);m['response_residual'][0]+=.001;m['drift']=max(abs(x) for x in m['response_residual']);m['gates']['response']=m['drift']<=.002;m['pass']=all(m['gates'].values());return m
            rt.evaluate=wrong
            with self.assertRaises(ValueError):measure_raw(rt,out,specs('new_training')[0])
    def test_source_bundle_is_original(self):check_sources(ROOT)

    def tiny_autograd_runtime(self):
        rt=Runtime.__new__(Runtime);rt.params=[torch.nn.Parameter(torch.zeros(36864),requires_grad=False)]
        rt.device=torch.device('cpu');rt.model=None;rt.tok=None;rt.anchors=[];rt.responseids=list(range(6))
        rt.invariant=lambda:require(all(not p.requires_grad and p.grad is None for p in rt.params),'Gradient freeze not restored')
        def assign(params,theta):
            with torch.no_grad():params[0].copy_(theta)
        def response(*args):return rt.params[0][:6]**2+torch.sin(rt.params[0][6:12])
        return rt,types.SimpleNamespace(assign_flat=assign,response_vector=response)

    def test_real_autograd_jacobian_and_cache(self):
        rt,w=self.tiny_autograd_runtime();theta=torch.zeros(36864);theta[:12]=torch.linspace(.1,1.2,12)
        with tempfile.TemporaryDirectory() as d,patch.object(old_runtime,'w',w,create=True):
            p=Path(d)/'J.pt';J=rt.jacobian(theta,p)
            wanted=np.zeros((6,36864));wanted[np.arange(6),np.arange(6)]=(2*theta[:6]).numpy()
            wanted[np.arange(6),np.arange(6,12)]=torch.cos(theta[6:12]).numpy()
            np.testing.assert_allclose(J,wanted,atol=1e-8);rt.invariant()
            w.response_vector=lambda *a:(_ for _ in ()).throw(AssertionError('Cached Jacobian recomputed'))
            np.testing.assert_array_equal(J,rt.jacobian(theta,p))
            with self.assertRaises(ValueError):rt.jacobian(theta+1,p)

    def test_autograd_failure_restores_gradient_freeze(self):
        rt,w=self.tiny_autograd_runtime()
        w.response_vector=lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic derivative failure'))
        with tempfile.TemporaryDirectory() as d,patch.object(old_runtime,'w',w,create=True):
            with self.assertRaises(RuntimeError):rt.jacobian(torch.zeros(36864),Path(d)/'J.pt')
            rt.invariant();self.assertFalse((Path(d)/'J.pt').exists())

if __name__=='__main__':unittest.main()
