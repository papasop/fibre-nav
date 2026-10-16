"""Synthetic tests of the real pipeline; not GPT-2 experimental evidence."""
import contextlib, hashlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import types
import c31_runtime as runtime_module
from c31_protocol import *
from c31_prediction import fit_arm, freeze_all, check_frozen, forecast
from c31_runtime import generate_sketch, vector, save_tensor, load, tensor_hash
from worker_c31 import main, collect_state, arrays, validate_cache


def fixture(out):
    out=Path(out);root=Path(__file__).resolve().parent
    dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)})
    tr=[{'id':c+'__v%d'%i if i<3 else c+'__old%02d'%i,'case':c} for c in CASES for i in range(19)]
    dv=[{'id':c+'__unimported_old%d'%i,'case':c} for c in CASES for i in range(4)]
    dump(out/'parent/SPLIT.json',{'training':tr,'prospective_new_states':dv})
    dump(out/'parent/C29_FROZEN_IDENTITIES.json',{'synthetic':True})
    dump(out/'parent/C29_SUMMARY.json',{'endpoint_replay':{'training':{'c27':{'n':76,'valid':76,'passed':76,'maximum_R':.0003}}}})
    rng=np.random.default_rng(310077)
    for t in tr:
        sid,case=t['id'],t['case'];folder=out/'data/training'/sid
        x=torch.zeros(3147);x[3077:]=torch.from_numpy(rng.normal(size=70).astype(np.float32))
        f={'features':torch.zeros(3077),'input':x,'route':torch.tensor(CASES.index(case))}
        if not sid.endswith('v0'):f['extras']=torch.zeros(70)
        save_tensor(folder/'INPUT.pt',f);dump(folder/'REFERENCE.json',{'valid':True})
        dump(folder/'C29_INPUT_AUDIT.json',{'missing_extras_reconstructed':False,'extras_present':'extras' in f})
        m={'theta_sha256':'mock'+sid,'response_residual':[.0003]*6,'drift':.0003,'kl':.001,'gates':{k:True for k in GATES},'bits':STARTS[case][:3]+[STARTS[case][0]^STARTS[case][1]]}
        dump(folder/'C29_COMPLETE.json',{'id':sid,'case':case,'cohort':'training','reference_valid':True,'frozen':{'synthetic':True},'theta_sha256':{'c27':m['theta_sha256']},'endpoints':{'c27':{'reference_valid':True,'metrics':m}}})
        if '__v' in sid:
            theta=torch.zeros(36864);theta[0]=3*CASES.index(case)+int(sid[-1]);save_tensor(folder/'prepared.pt',theta)
    pv={p.relative_to(out).as_posix():{'sha256':sha(p),'member':'synthetic/'+p.relative_to(out).as_posix()} for p in out.rglob('*') if p.is_file() and p.name!='identity.json'}
    dump(out/'PARENT_PROVENANCE.json',{'members':pv,'synthetic_only':True})


class MockRuntime:
    """Full-size chart and actual stored inputs; no real model or geometry claims."""
    def __init__(self,out,root):
        self.out=Path(out);self.original={}
        for c in CASES:
            for i in range(3):self.original[3*CASES.index(c)+i]=load(self.out/'data/training'/(c+'__v%d'%i)/'INPUT.pt')
        fr={'synthetic_writer':True,'unchanged':True}
        p=self.out/'WRITER_FROZEN.json'
        if p.exists():require(read(p)==fr,'Changed mock writer')
        else:dump(p,fr)
    def fresh_input(self,theta,desired):
        idx=int(round(float(theta[0])));old=self.original[idx]
        f={k:v.clone() for k,v in old.items()};f['input'][-6:]+=theta[16:22];return f
    def raw_write(self,prep,inp):
        theta=prep.clone();theta[8]=1
        return theta,{'proposal_count':1,'online_Jacobians':0,'online_SVD_calls':0,'post_write_correction_steps':0}
    def evaluate(self,theta,target,start=None):
        written=bool(theta[8]>.5);e=.0003+.4*float(theta[16]) if written else 0.
        gates={'bits_match':True,'response':abs(e)<=.002,'KL':True,'exact_read_margin':True,'development_read_margin':True}
        # Deliberately invalid preparation group: must remain in all denominators.
        if not written and int(round(float(theta[0])))==0:gates['bits_match']=False
        if start is not None:gates.update(output_flip=True,unselected_preserved=True)
        return {'theta_sha256':tensor_hash(theta),'bits':list(target),'response_residual':[e]*6,'drift':abs(e),'kl':.001,'gates':gates,'pass':all(gates.values())}
    def jacobian(self,theta,path):
        J=np.zeros((6,36864));J[:,:6]=np.eye(6);save_tensor(path,{'J':torch.from_numpy(J),'theta_sha256':tensor_hash(theta)});return J
    def correct_preparation(self,initial,folder):return initial,[],0.
    def invariant(self):pass


class Tests(unittest.TestCase):
    def test_sketch_exact_reconstruction(self):
        s=generate_sketch();self.assertEqual(tuple(s.shape),(36864,64));self.assertEqual(tensor_hash(s),SETTINGS['sketch_tensor_sha256'])

    def test_vector_rejects_wrong_dtype(self):
        with self.assertRaises(ValueError):vector(torch.zeros(36864,dtype=torch.float64))

    def test_vector_rejects_nan(self):
        v=torch.zeros(36864);v[0]=float('nan')
        with self.assertRaises(ValueError):vector(v)

    def test_predictor_frozen_before_prospective_generation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';fixture(out);rt=MockRuntime(out,Path(__file__).parent)
            with self.assertRaises(FileNotFoundError):collect_state(rt,out,specs('prospective')[0])

    def test_real_saved_state_and_input_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';fixture(out);rt=MockRuntime(out,Path(__file__).parent)
            with contextlib.redirect_stdout(io.StringIO()):r=collect_state(rt,out,specs('new_training')[0])
            self.assertFalse(r['preparation_valid'])
            X,Y,rs=arrays(out,[r]);self.assertEqual(len(rs),1);self.assertFalse(rs[0]['preparation_valid'])
            p=out/'states/new_training'/r['spec']['id']/'RAW_WRITTEN.pt';p.write_bytes(p.read_bytes()+b'tamper')
            with self.assertRaises(ValueError):validate_cache(out,r,r['spec'])

    def test_cached_state_preserves_actual_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';fixture(out);rt=MockRuntime(out,Path(__file__).parent);t=specs('new_training')[0]
            with contextlib.redirect_stdout(io.StringIO()):r=collect_state(rt,out,t);s=collect_state(rt,out,t)
            self.assertEqual(r,s)

    def test_response_prediction_is_not_raw_write_acceptance(self):
        Y=np.full((4,6),.01);r=predictor.report(Y,Y,['xor_00__%d'%i for i in range(4)],[True]*4)
        self.assertEqual(r['prediction_error_within_response_budget'],4);self.assertEqual(r['actual_endpoint_response_passed'],0)

    def test_actual_autograd_jacobian_and_cleanup(self):
        rt=runtime_module.Runtime.__new__(runtime_module.Runtime)
        rt.device=torch.device('cpu');p=torch.nn.Parameter(torch.zeros(36864),requires_grad=False);rt.params=[p]
        rt.model=object();rt.tok=None;rt.anchors=[];rt.responseids=list(range(6));rt.invariant=lambda:None
        def assign(params,theta):
            with torch.no_grad():params[0].copy_(theta)
        fake=types.SimpleNamespace(assign_flat=assign,response_vector=lambda *args:p[:6].square()+2*p[:6])
        theta=torch.zeros(36864);theta[:6]=torch.arange(6)/10
        with tempfile.TemporaryDirectory() as d,patch.object(runtime_module,'w',fake,create=True):
            J=rt.jacobian(theta,Path(d)/'J.pt')
            np.testing.assert_allclose(J[:,:6],np.diag(2*theta[:6].numpy()+2))
            self.assertEqual(np.count_nonzero(J[:,6:]),0);self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
            np.testing.assert_array_equal(J,rt.jacobian(theta,Path(d)/'J.pt'))

    def test_actual_fresh_input_feature_layout_and_route(self):
        rt=runtime_module.Runtime.__new__(runtime_module.Runtime)
        rt.device=torch.device('cpu');rt.params=[];rt.tok=None;rt.anchors=[];rt.responseids=list(range(6));rt.readids=[1,2]
        rt.response0=torch.zeros(6);rt.exact_batch={'attention_mask':torch.ones((4,3),dtype=torch.long)}
        rt.gnorm={'mean':torch.zeros(3077),'scale':torch.ones(3077)};rt.gextra={'mean':torch.zeros(70),'scale':torch.ones(70)}
        rt.sketch=torch.zeros((36864,64));rt.sync=lambda:None;rt.invariant=lambda:None
        logits=torch.zeros((4,3,8));logits[:,-1,2]=torch.tensor([1.,-1.,1.,-1.])
        rt.model=lambda **kw:types.SimpleNamespace(hidden_states=[torch.ones((4,3,768))],logits=logits)
        fake=types.SimpleNamespace(assign_flat=lambda *args:None,response_vector=lambda *args:torch.ones(6)*.001)
        with patch.object(runtime_module,'w',fake,create=True):f=rt.fresh_input(torch.zeros(36864),1)
        self.assertEqual(f['features'].shape,(3077,));self.assertEqual(f['input'].shape,(3147,));self.assertEqual(int(f['route']),2)
        torch.testing.assert_close(f['input'][3077:3083],torch.ones(6)*.5);self.assertEqual(float(f['features'][-1]),1.)

    def test_end_to_end_interrupt_resume_freeze_and_all_denominators(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';fixture(out);real=collect_state;seen=[]
            def interrupted(rt,folder,t):
                if len(seen)==5:raise RuntimeError('simulated interrupt')
                value=real(rt,folder,t);seen.append(t['id']);return value
            with patch('worker_c31.collect_state',side_effect=interrupted),contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'simulated interrupt'):
                main(out,runtime_factory=MockRuntime)
            keep={s:sha(out/'states/new_training'/s/'COMPLETE.json') for s in seen}
            self.assertFalse((out/'PREDICTORS_FROZEN.json').exists());self.assertFalse((out/'states/prospective').exists())
            with contextlib.redirect_stdout(io.StringIO()):s=main(out,runtime_factory=MockRuntime)
            self.assertEqual(s['new_training']['all']['n_requested'],48);self.assertEqual(s['prospective_local_development']['all']['n_requested'],16)
            self.assertLess(s['new_training']['all']['valid_preparations'],48)
            self.assertEqual(s['prospective_prediction_comparison']['expanded124']['train_selected']['n_requested'],16)
            for sid,h in keep.items():self.assertEqual(sha(out/'states/new_training'/sid/'COMPLETE.json'),h)
            fr=check_frozen(out)
            self.assertEqual(len(fr['arms']['expanded124']['training_ids']),124)
            self.assertEqual(len(fr['arms']['historical76']['training_ids']),76)
            self.assertFalse(s['LC1_closed']);self.assertEqual(s['new_writer_training_steps'],0)
            with contextlib.redirect_stdout(io.StringIO()):again=main(out,runtime_factory=MockRuntime)
            self.assertEqual(s,again)
            key=next(iter(fr['arms']['expanded124']['files']));p=out/key;p.write_bytes(p.read_bytes()+b'bad')
            with self.assertRaisesRegex(ValueError,'modified'):check_frozen(out)


if __name__=='__main__':unittest.main(verbosity=2)
