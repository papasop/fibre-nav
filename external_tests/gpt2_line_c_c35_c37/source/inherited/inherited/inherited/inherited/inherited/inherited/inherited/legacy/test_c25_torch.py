import copy,hashlib,io,json,tempfile,textwrap,time,unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import torch
from test_c18_torch import Tests as InheritedIsolationTests
from raw_writer import StateWriter,one_raw_proposal
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from joint_calibration import prepare_system
from budget_calibration import interval_coefficients,reconstruct_direction
from c25_protocol import CASES,GATES,SETTINGS,training_specs,original60_specs,START_HEAD_SHA
from c25_identity import head_hash,tensor_hash,state_hash
import c25_identity as ident
from c25_guard import quality
from c25_scan import scan_round
ROOT=Path(__file__).parent

def dump(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,allow_nan=False))
def save_tensor(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
def vector(v):return v['theta'].clone() if isinstance(v,dict) else v.clone()

class PhysicalScanTests(unittest.TestCase):
 def exercise(self,case='xor_00',fail=False,audit_fail=False,twice=False):
  torch.manual_seed(25);h=StateWriter(1,2,2)
  for r in range(4):set_layer_matrix(h,r,torch.tensor([[0.,0.,.04],[0.,0.,1.]]))
  for p in h.parameters():p.requires_grad_(False)
  original_ids={s['id'] for s in original60_specs()}
  data=[{'id':s['id'],'case':s['case'],'input':torch.tensor([.1]),'route':torch.tensor(CASES.index(s['case'])),'prep':torch.tensor([0. if s['id'] in original_ids else .04,0.]),'reference_valid':True} for s in training_specs()]
  route=CASES.index(case);group=[t for t in data if t['case']==case];old_hash=head_hash(h);calls={'eval':0,'audit':0,'snap':0}
  def metric(theta):
   d=float(theta[0]**2);content=abs(float(theta[1])-1)<1e-6;g={k:content for k in GATES};g['response']=d<=.002
   return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values()),'response_residual':[d],'theta_sha256':tensor_hash(theta)}
  def measure(writer):
   calls['audit']+=1;rows={t['id']:{'reference_valid':True,'metrics':metric(t['prep']+one_raw_proposal(writer,t['input'],t['route'],torch.tensor(1.)))} for t in data}
   if audit_fail:rows[data[0]['id']]['metrics']['gates']['KL']=False
   return rows
  current={t['id']:{'reference_valid':True,'metrics':metric(t['prep']+one_raw_proposal(h,t['input'],t['route'],torch.tensor(1.)))} for t in data}
  states=[{'theta':t['prep']+one_raw_proposal(h,t['input'],t['route'],torch.tensor(1.))} for t in group]
  system={'J':np.array([[[2*float(r['theta'][0]),0.]] for r in states]),'H':np.stack([hidden_augmented(h,t['input'],route).numpy() for t in group]),'E':np.array([[float(r['theta'][0]**2)] for r in states]),'scale':1.,'info':{}}
  bases=[np.array([[0.,0.,-.05],[0.,0.,0.]]) for _ in range(3)]
  def evaluate(theta,t):
   calls['eval']+=1
   if fail:raise RuntimeError('injected physical interruption')
   return metric(theta)
  original={n:v.clone() for n,v in h.state_dict().items()}
  def invariant():
   for n,v in h.state_dict().items():
    if not n.startswith('experts.%d.2.'%route):self.assertTrue(torch.equal(v,original[n]))
   self.assertFalse(any(p.requires_grad or p.grad is not None for p in h.parameters()))
  def snapshot():calls['snap']+=1
  with tempfile.TemporaryDirectory() as d:
   if fail:
    with self.assertRaisesRegex(RuntimeError,'injected'):scan_round(h,route,group,states,system,bases,current,d,torch.tensor(1.),evaluate,measure,invariant,dump,snapshot)
    self.assertEqual(head_hash(h),old_hash);return
   with redirect_stdout(io.StringIO()):winner,info=scan_round(h,route,group,states,system,bases,current,d,torch.tensor(1.),evaluate,measure,invariant,dump,snapshot)
   self.assertEqual(head_hash(h),old_hash);self.assertEqual(info['n_completed_candidate_specs'],90);self.assertEqual(calls['eval'],19*info['n_unique_candidate_heads']);self.assertLess(info['n_unique_candidate_heads'],90);self.assertEqual(calls['snap'],18)
   if audit_fail:self.assertIsNone(winner);self.assertGreater(calls['audit'],1)
   else:self.assertTrue(winner['full76_verified']);self.assertEqual(len(winner['rows']),76);self.assertEqual(calls['audit'],1);self.assertTrue(original_ids.issubset(quality(winner['rows'])['passed_ids']))
   if twice:
    calls['eval']=calls['audit']=0
    with redirect_stdout(io.StringIO()):winner2,info2=scan_round(h,route,group,states,system,bases,current,d,torch.tensor(1.),evaluate,measure,invariant,dump,snapshot)
    self.assertEqual(info,info2);self.assertEqual(calls['eval'],0);self.assertEqual(calls['audit'],0);self.assertEqual(winner['head_sha256'],winner2['head_sha256'])
 def test_actual_scan_all_four_routes_19_states_full76_and_restore(self):
  for case in CASES:self.exercise(case)
 def test_actual_scan_resume_is_identity_checked(self):self.exercise(twice=True)
 def test_actual_interruption_restores(self):self.exercise(fail=True)
 def test_actual_failed_full76_never_commits(self):self.exercise(audit_fail=True)
 def test_interval114_constraints_real_solver(self):
  r=np.random.default_rng(2519);J=r.normal(size=(19,6,10));H=r.normal(size=(19,5));E=r.normal(size=(19,6))*.004;s=prepare_system(J,H,E,.07);c,i=interval_coefficients(s,1e-4,.0018,8192,1e-8);self.assertEqual(c.shape,(114,));self.assertTrue(np.isfinite(reconstruct_direction(s,c)).all());self.assertFalse(i['finite_acceptance_certified'])

class IdentityTests(unittest.TestCase):
 def fixture(self):
  torch.manual_seed(2500);h=StateWriter(1,36864,64);sha=head_hash(h);cp={'head':copy.deepcopy(h.state_dict()),'rounds':8,'accepted_updates':8,'done':True,'stop':'ALL_60_TRAINING_STATES_ACCEPTED','optimiser':None,'start_head_sha256':ident.PARENT_START_HEAD_SHA,'train_ids':[s['id'] for s in ident.training_specs()],'head_sha256':sha};return cp,sha
 def test_actual_head_tensors_copied_without_history(self):
  cp,sha=self.fixture()
  with patch.object(ident,'START_HEAD_SHA',sha):state=ident.validate_start(cp)
  self.assertEqual(state_hash(state),sha);state[next(iter(state))].add_(.1);self.assertEqual(state_hash(cp['head']),sha)
 def test_corrupt_real_tensor_rejected(self):
  cp,sha=self.fixture();cp['head']['experts.0.0.bias'][0]+=.01
  with patch.object(ident,'START_HEAD_SHA',sha),self.assertRaises(ValueError):ident.validate_start(cp)
 def test_nonfinite_tensor_rejected(self):
  cp,sha=self.fixture();cp['head']['experts.0.0.bias'][0]=float('nan')
  with patch.object(ident,'START_HEAD_SHA',sha),self.assertRaises(ValueError):ident.validate_start(cp)
 def test_wrong_checkpoint_stage_rejected(self):
  cp,sha=self.fixture();cp['rounds']=7
  with patch.object(ident,'START_HEAD_SHA',sha),self.assertRaises(ValueError):ident.validate_start(cp)
 def test_wrong_original_training_ids_rejected(self):
  cp,sha=self.fixture();cp['train_ids']=[]
  with patch.object(ident,'START_HEAD_SHA',sha),self.assertRaises(ValueError):ident.validate_start(cp)

class ActualWorkerBlocks(unittest.TestCase):
 def test_actual_geometry19_state_derivatives_resume_and_head_change(self):
  torch.manual_seed(2519);head=StateWriter(1,36864,64)
  for p in head.parameters():p.requires_grad_(False)
  data=[{'id':t['id'],'case':t['case'],'input':torch.tensor([.1]),'route':torch.tensor(CASES.index(t['case'])),'prep':torch.zeros(36864)} for t in training_specs()]
  case='xor_11';device='cpu';model=anchor_batch=spec=None;response_ids=list(range(6));response0=torch.zeros(6);c9norm={'output_scale':torch.tensor(.07)};calls={'functional':0};sync=frozen_model_check=baseline_check=lambda:None
  def last_logits(model,batch,theta,spec):calls['functional']+=1;return (theta[:6]**2).unsqueeze(0)
  current={t['id']:{'metrics':{'response_residual':(one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale'])[:6]**2).double().tolist()}} for t in data}
  snapshot=lambda *args:None;args=type('Args',(),{'drive_dir':''})()
  text=(ROOT/'worker_c25.py').read_text();a=text.index('  def build_route_geometry');b=text.index("  if not (out/'WRITER_FROZEN.json')",a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals())
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);ns['out']=out;exec(compile(code,'<actual C25 geometry>','exec'),ns)
   with redirect_stdout(io.StringIO()):group,states=ns['build_route_geometry'](case,out/'round')
   self.assertEqual(len(states),19);self.assertEqual(calls['functional'],19)
   for r in states:torch.testing.assert_close(r['J'][:,:6],torch.diag(2*r['theta'][:6]));self.assertEqual(int(torch.count_nonzero(r['J'][:,6:])),0)
   with redirect_stdout(io.StringIO()):ns['build_route_geometry'](case,out/'round')
   self.assertEqual(calls['functional'],19);set_layer_matrix(head,3,layer_matrix(head,3)+.001)
   with self.assertRaisesRegex(RuntimeError,'wrong head'):ns['build_route_geometry'](case,out/'round')
 def test_actual_online_one_proposal_cached_input_and_state_identity(self):
  device='cpu';writer=StateWriter(1,2,2);prep=torch.zeros(2);inputs={'input':torch.tensor([.1]),'route':torch.tensor(2)};c9norm={'output_scale':torch.tensor(1.)};expected_hash=head_hash(writer);calls={'proposals':0,'eval':0};sync=writer_frozen_check=lambda:None;original_proposal=globals()['one_raw_proposal']
  def one_raw_proposal(*args):calls['proposals']+=1;return original_proposal(*args)
  def evaluate(theta,*args):calls['eval']+=1;return {'drift':.001,'kl':.001,'episode_pass':True}
  text=(ROOT/'worker_c25.py').read_text();a=text.index('  def online_raw_arm');b=text.index('  trainrows=[]',a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals());exec(compile(code,'<actual C25 online>','exec'),ns)
  with tempfile.TemporaryDirectory() as d:
   r=ns['online_raw_arm'](writer,prep,inputs,[],[],Path(d),expected_hash);self.assertEqual(calls,{'proposals':1,'eval':1});self.assertEqual(r['proposals'],1);self.assertEqual(r['Jacobian_builds'],0)
   self.assertEqual(r,ns['online_raw_arm'](writer,prep,inputs,[],[],Path(d),expected_hash));self.assertEqual(calls,{'proposals':1,'eval':1})
   with self.assertRaisesRegex(RuntimeError,'Cached online head'):ns['online_raw_arm'](writer,prep+1,inputs,[],[],Path(d),expected_hash)
 def test_actual_checkpoint_preserves_route_scheduler(self):
  head=StateWriter(1,2,2);rounds=3;accepted=2;history=[];current={};train_ids=[];extra_norm={'mean':torch.zeros(70),'scale':torch.ones(70)};stop='ALL_REMAINING_ROUTE_GRIDS_STALLED';stalled=['xor_01'];last_case_index=2;START_HEAD_SHA='fixture';cpu_state=lambda h:{n:v.clone() for n,v in h.state_dict().items()};cpu_tree=lambda x:x;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  text=(ROOT/'worker_c25.py').read_text();a=text.index('  def checkpoint');b=text.index('  if progress.exists():',a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals())
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);progress=folder/'PROGRESS.json';ns.update(folder=folder,progress=progress);exec(compile(code,'<actual C25 checkpoint>','exec'),ns);ns['checkpoint']();ns['checkpoint'](True);p=json.loads(progress.read_text());cp=torch.load(folder/p['checkpoint'],weights_only=True)
   self.assertEqual(cp['stalled_cases'],stalled);self.assertEqual(cp['last_case_index'],2);self.assertTrue(cp['done']);self.assertIsNone(cp['optimiser']);self.assertEqual(cp['head_sha256'],head_hash(head));self.assertEqual(p['checkpoint_sha256'],sha(folder/p['checkpoint']))
 def test_actual_checkpoint_restore_all_outputs_frozen_inputs(self):
  from test_c25 import cohort
  head=StateWriter(1,2,2);state=copy.deepcopy(head.state_dict());expected=head_hash(head);device='cpu';train_ids=[];START_HEAD_SHA='fixture';extra_norm={'mean':torch.zeros(70),'scale':torch.ones(70)};initial_quality=quality(cohort());current=cohort();rounds=accepted=0;history=[];stop=None;stalled=[];last_case_index=-1;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  text=(ROOT/'worker_c25.py').read_text();a=text.index('  if progress.exists():');b=text.index('  else:checkpoint()',a);body=textwrap.dedent(text[a:b]);code='def actual_restore():\n'+textwrap.indent(body,' ')+'\n return rounds,accepted,stalled,last_case_index\n';ns=dict(globals());ns.update(locals())
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);progress=folder/'PROGRESS.json';cp_path=folder/'checkpoint_000003.pt';cp={'head':state,'train_ids':train_ids,'start_head_sha256':START_HEAD_SHA,'rounds':3,'accepted_updates':2,'history':[],'current_training':current,'extra_norm':extra_norm,'stop':None,'head_sha256':expected,'stalled_cases':['xor_00'],'last_case_index':3}
   save_tensor(cp_path,cp);dump(progress,{'checkpoint':cp_path.name,'checkpoint_sha256':sha(cp_path),'rounds':3});ns.update(folder=folder,progress=progress);exec(compile(code,'<actual C25 restore>','exec'),ns)
   for route in range(4):set_layer_matrix(head,route,layer_matrix(head,route)+.01)
   with redirect_stdout(io.StringIO()):r=ns['actual_restore']()
   self.assertEqual(r,(3,2,['xor_00'],3));self.assertEqual(head_hash(head),expected)
   cp['head']['experts.0.0.bias'][0]+=.01;save_tensor(cp_path,cp);dump(progress,{'checkpoint':cp_path.name,'checkpoint_sha256':sha(cp_path),'rounds':3})
   with self.assertRaisesRegex(RuntimeError,'frozen writer input'):ns['actual_restore']()
 def test_actual_new_state_preparation_frozen_phase_no_replacement_and_resume(self):
  t={'id':'xor_00__c25_new0','case':'xor_00','base_id':'xor_00__v0','seed':2515031,'norm':.375,'cohort':'new'};out=None;head=StateWriter(1,2,2);frozen_hash=head_hash(head);START_HEAD_SHA='origin';base_data={'xor_00__v0':(torch.zeros(36864),torch.ones(36864))};starts={'xor_00':[0,0,0,1]};targets={'xor_00':[0,0,0,0]};calls={'projection':0,'correct':0,'features':0}
  def load_obj(*args):return {'J':torch.zeros(6,36864)}
  capsule=None
  def null_project(j,v):calls['projection']+=1;return v,{'fixture':True}
  def offline_correct(folder,theta):calls['correct']+=1;return theta,[],0.
  def evaluate(theta,target,start=None):return {'pass':True,'episode_pass':True}
  def fresh_inputs(theta,desired):calls['features']+=1;return {'route':torch.tensor(0),'input':torch.zeros(3147)}
  baseline_check=frozen_model_check=writer_frozen_check=lambda:None
  text=(ROOT/'worker_c25.py').read_text();a=text.index('  def prepare_state');b=text.index('  def measure_training',a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals());exec(compile(code,'<actual C25 prepare>','exec'),ns)
  with tempfile.TemporaryDirectory() as d:
   ns['out']=Path(d)
   with self.assertRaisesRegex(RuntimeError,'actual final frozen'):ns['prepare_state'](t,frozen_hash)
   (Path(d)/'WRITER_FROZEN.json').write_text('{}');folder,r=ns['prepare_state'](t,frozen_hash);self.assertEqual(r['generation_retries'],0);self.assertFalse(r['replacement_used']);self.assertEqual(r['writer_frozen_before_generation'],frozen_hash);self.assertEqual(calls,{'projection':1,'correct':2,'features':1})
   ns['prepare_state'](t,frozen_hash);self.assertEqual(calls,{'projection':1,'correct':2,'features':1})
   with self.assertRaisesRegex(RuntimeError,'actual final frozen'):ns['prepare_state'](t,'wrong')
 def test_actual_worker_import_HF_boundary_only_stubbed(self):
  import importlib,sys,types
  tf=types.ModuleType('transformers');tf.__version__='4.56.2';tf.AutoTokenizer=object;tf.AutoModelForCausalLM=object;pe=types.ModuleType('peft');pe.__version__='0.17.1';pe.LoraConfig=object;pe.get_peft_model=lambda *args:None
  with patch.dict(sys.modules,{'transformers':tf,'peft':pe}):m=importlib.import_module('worker_c25')
  self.assertTrue(callable(m.main));self.assertEqual(m.PROTOCOL,'GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1')

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
