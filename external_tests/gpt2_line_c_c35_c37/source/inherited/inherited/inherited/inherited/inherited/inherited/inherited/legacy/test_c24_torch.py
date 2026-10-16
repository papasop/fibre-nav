import copy,hashlib,io,json,tempfile,textwrap,time,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from test_c18_torch import Tests as InheritedIsolationTests
from raw_writer import StateWriter,one_raw_proposal
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from joint_calibration import prepare_system
from budget_calibration import interval_coefficients,reconstruct_direction
from c24_identity import head_hash,tensor_hash,state_hash
import c24_identity as ident
from c24_scan import scan_round
from c24_protocol import CASES,SETTINGS,GATES,training_specs,quality,START_SPEC
ROOT=Path(__file__).parent

def dump(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,allow_nan=False))
def save_tensor(path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
def vector(v):return v['theta'].clone() if isinstance(v,dict) else v.clone()

class PhysicalScanTests(unittest.TestCase):
 def exercise(self,fail=False,audit_fail=False,twice=False):
  torch.manual_seed(24);h=StateWriter(1,2,2)
  for route in range(4):set_layer_matrix(h,route,torch.tensor([[0.,0.,.09],[0.,0.,1.]]))
  for p in h.parameters():p.requires_grad_(False)
  data=[{'id':t['id'],'case':t['case'],'input':torch.tensor([.1]),'route':torch.tensor(CASES.index(t['case'])),'prep':torch.zeros(2),'reference_valid':True} for t in training_specs()]
  case=SETTINGS['scan_case'];route=CASES.index(case);group=[t for t in data if t['case']==case];old_hash=head_hash(h);calls={'eval':0,'audit':0,'snap':0}
  def metric(theta):
   d=float(theta[0]**2);content=abs(float(theta[1])-1)<1e-6;g={k:content for k in GATES};g['response']=d<=.002
   return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values()),'response_residual':[d],'theta_sha256':tensor_hash(theta)}
  def measure(writer):
   calls['audit']+=1;rows={t['id']:{'reference_valid':True,'metrics':metric(one_raw_proposal(writer,t['input'],t['route'],torch.tensor(1.)))} for t in data}
   if audit_fail:rows[data[0]['id']]['metrics']['gates']['KL']=False
   return rows
  current={t['id']:{'reference_valid':True,'metrics':metric(one_raw_proposal(h,t['input'],t['route'],torch.tensor(1.)))} for t in data}
  states=[{'theta':one_raw_proposal(h,t['input'],t['route'],torch.tensor(1.))} for t in group]
  system={'J':np.repeat(np.array([[[.18,0.]]]),15,axis=0),'H':np.stack([hidden_augmented(h,t['input'],route).numpy() for t in group]),'E':np.full((15,1),.0081),'scale':1.,'info':{}}
  bases=[np.array([[0.,0.,-.045],[0.,0.,0.]]) for _ in range(3)]
  def evaluate(theta,t):
   calls['eval']+=1
   if fail:raise RuntimeError('injected physical interruption')
   return metric(theta)
  original={n:v.clone() for n,v in h.state_dict().items()}
  def invariant():
   for n,v in h.state_dict().items():
    if not n.startswith('experts.2.2.'):self.assertTrue(torch.equal(v,original[n]))
   self.assertFalse(any(p.requires_grad or p.grad is not None for p in h.parameters()))
  def snapshot():calls['snap']+=1
  with tempfile.TemporaryDirectory() as d:
   if fail:
    with self.assertRaisesRegex(RuntimeError,'injected'):scan_round(h,route,group,states,system,bases,current,d,torch.tensor(1.),evaluate,measure,invariant,dump,snapshot)
    self.assertEqual(head_hash(h),old_hash);return
   with redirect_stdout(io.StringIO()):winner,info=scan_round(h,route,group,states,system,bases,current,d,torch.tensor(1.),evaluate,measure,invariant,dump,snapshot)
   self.assertEqual(head_hash(h),old_hash);self.assertEqual(info['n_completed_candidate_specs'],90);self.assertEqual(calls['eval'],15*info['n_unique_candidate_heads']);self.assertLess(info['n_unique_candidate_heads'],90);self.assertTrue(info['origin_restored']);self.assertEqual(calls['snap'],18)
   if audit_fail:self.assertIsNone(winner);self.assertGreater(calls['audit'],1)
   else:self.assertTrue(winner['full60_verified']);self.assertEqual(len(winner['rows']),60);self.assertEqual(calls['audit'],1)
   if twice:
    calls['eval']=calls['audit']=0
    with redirect_stdout(io.StringIO()):winner2,info2=scan_round(h,route,group,states,system,bases,current,d,torch.tensor(1.),evaluate,measure,invariant,dump,snapshot)
    self.assertEqual(info,info2);self.assertEqual(calls['eval'],0);self.assertEqual(calls['audit'],0);self.assertEqual(winner['head_sha256'],winner2['head_sha256']);self.assertEqual(head_hash(h),old_hash)
 def test_actual_all90_scan_aliases_full60_and_origin_restore(self):self.exercise()
 def test_actual_scan_resume_reuses_same_identity_measurements(self):self.exercise(twice=True)
 def test_actual_interruption_restores_round_origin(self):self.exercise(fail=True)
 def test_actual_failed_full60_audit_never_commits(self):self.exercise(audit_fail=True)
 def test_fresh_interval_direction_for_curved_response(self):
  J=np.repeat(np.array([[[.18]]]),15,axis=0);H=np.ones((15,1));E=np.full((15,1),.0081);s=prepare_system(J,H,E,1.)
  c,_=interval_coefficients(s,1e-8,.0018);D=reconstruct_direction(s,c);x=.09+D.item();self.assertLess(x*x,.0081);self.assertNotAlmostEqual(x*x,.0018,places=6)

class IdentityTests(unittest.TestCase):
 def setup_identity(self):
  torch.manual_seed(2400);h=StateWriter(1,36864,64);cp={'head':copy.deepcopy(h.state_dict()),'rounds':15,'accepted_updates':13,'done':True};origin=state_hash(cp['head']);layer=layer_matrix(h,2);layer[0,-1]+=.01;set_layer_matrix(h,2,layer);expected=head_hash(h)
  r={'layer':layer,'origin_head_sha256':origin,'candidate_head_sha256':expected,'spec':START_SPEC,'diagnostic_only_not_committed':True};return cp,r,origin,expected
 def test_real_shapes_reconstruct_actual_head_and_only_selected_layer(self):
  cp,r,origin,expected=self.setup_identity()
  with patch.object(ident,'ORIGIN_HEAD_SHA',origin),patch.object(ident,'START_HEAD_SHA',expected):s=ident.reconstruct_start(cp,r)
  self.assertEqual(state_hash(s),expected)
  for n in s:
   if not n.startswith('experts.2.2.'):self.assertTrue(torch.equal(s[n],cp['head'][n]))
  self.assertEqual(state_hash(cp['head']),origin)
 def test_actual_corrupt_layer_rejected(self):
  cp,r,origin,expected=self.setup_identity();r['layer'][1,0]+=.01
  with patch.object(ident,'ORIGIN_HEAD_SHA',origin),patch.object(ident,'START_HEAD_SHA',expected),self.assertRaisesRegex(ValueError,'identity mismatch'):ident.reconstruct_start(cp,r)
 def test_nonfinite_real_tensor_rejected(self):
  cp,r,origin,expected=self.setup_identity();r['layer'][0,0]=float('nan')
  with patch.object(ident,'ORIGIN_HEAD_SHA',origin),patch.object(ident,'START_HEAD_SHA',expected),self.assertRaisesRegex(ValueError,'finite|Finite'):ident.reconstruct_start(cp,r)
 def test_wrong_source_head_rejected(self):
  cp,r,origin,expected=self.setup_identity();cp['head']['experts.0.0.bias'][0]+=.001
  with patch.object(ident,'ORIGIN_HEAD_SHA',origin),patch.object(ident,'START_HEAD_SHA',expected),self.assertRaisesRegex(ValueError,'origin'):ident.reconstruct_start(cp,r)
 def test_wrong_saved_spec_rejected(self):
  cp,r,origin,expected=self.setup_identity();r['spec']={}
  with patch.object(ident,'ORIGIN_HEAD_SHA',origin),patch.object(ident,'START_HEAD_SHA',expected),self.assertRaisesRegex(ValueError,'identity'):ident.reconstruct_start(cp,r)

class ActualWorkerBlocks(unittest.TestCase):
 def test_actual_geometry_block_derivatives_and_same_head_resume(self):
  torch.manual_seed(2415);head=StateWriter(1,36864,64)
  for p in head.parameters():p.requires_grad_(False)
  data=[{'id':t['id'],'case':t['case'],'input':torch.tensor([.1]),'route':torch.tensor(CASES.index(t['case'])),'prep':torch.zeros(36864)} for t in training_specs()]
  case='xor_10';device='cpu';model=anchor_batch=spec=None;response_ids=list(range(6));response0=torch.zeros(6);c9norm={'output_scale':torch.tensor(.07)};calls={'functional':0};sync=frozen_model_check=baseline_check=lambda:None
  def last_logits(model,batch,theta,spec):calls['functional']+=1;return (theta[:6]**2).unsqueeze(0)
  current={t['id']:{'metrics':{'response_residual':(one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale'])[:6]**2).double().tolist()}} for t in data}
  snapshot=lambda *args:None;args=type('Args',(),{'drive_dir':''})()
  text=(ROOT/'worker_c24.py').read_text();a=text.index('  def build_route_geometry');b=text.index("  if not (out/'WRITER_FROZEN.json')",a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals())
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);ns['out']=out;exec(compile(code,'<actual C24 geometry>','exec'),ns)
   with redirect_stdout(io.StringIO()):group,states=ns['build_route_geometry'](case,out/'round')
   self.assertEqual(len(states),15);self.assertEqual(calls['functional'],15)
   for r in states:
    torch.testing.assert_close(r['J'][:,:6],torch.diag(2*r['theta'][:6]));self.assertEqual(int(torch.count_nonzero(r['J'][:,6:])),0)
   with redirect_stdout(io.StringIO()):ns['build_route_geometry'](case,out/'round')
   self.assertEqual(calls['functional'],15)
   set_layer_matrix(head,2,layer_matrix(head,2)+.001)
   with self.assertRaisesRegex(RuntimeError,'wrong head'):ns['build_route_geometry'](case,out/'round')
 def test_actual_online_arm_one_proposal_and_cached_identity(self):
  device='cpu';writer=StateWriter(1,2,2);prep=torch.zeros(2);inputs={'input':torch.tensor([.1]),'route':torch.tensor(2)};c9norm={'output_scale':torch.tensor(1.)};expected_hash=head_hash(writer);calls={'proposals':0,'eval':0};sync=writer_frozen_check=lambda:None
  original_proposal=globals()["one_raw_proposal"]
  def one_raw_proposal(*args):calls['proposals']+=1;return original_proposal(*args)
  def evaluate(theta,*args):calls['eval']+=1;return {'drift':.001,'kl':.001,'episode_pass':True}
  text=(ROOT/'worker_c24.py').read_text();a=text.index('  def online_raw_arm');b=text.index('  trainrows=[]',a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals());exec(compile(code,'<actual C24 online arm>','exec'),ns)
  with tempfile.TemporaryDirectory() as d:
   r=ns['online_raw_arm'](writer,prep,inputs,[],[],Path(d),expected_hash);self.assertEqual(calls,{'proposals':1,'eval':1});self.assertEqual(r['proposals'],1);self.assertEqual(r['Jacobian_builds'],0);self.assertFalse(r['teacher_fallback'])
   self.assertEqual(r,ns['online_raw_arm'](writer,prep,inputs,[],[],Path(d),expected_hash));self.assertEqual(calls,{'proposals':1,'eval':1})
   with self.assertRaisesRegex(RuntimeError,'Cached online head'):ns['online_raw_arm'](writer,prep+1,inputs,[],[],Path(d),expected_hash)
 def test_actual_checkpoint_final_pointer_and_saved_weights(self):
  head=StateWriter(1,2,2);rounds=3;accepted=2;history=[];current={};train_ids=[];extra_norm={'mean':torch.zeros(70),'scale':torch.ones(70)};stop='FULL_FIXED_GRID_EXHAUSTED_AT_CURRENT_HEAD';START_HEAD_SHA='fixture';cpu_state=lambda h:{n:v.clone() for n,v in h.state_dict().items()};cpu_tree=lambda x:x;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  text=(ROOT/'worker_c24.py').read_text();a=text.index('  def checkpoint');b=text.index('  if progress.exists():',a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals())
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);progress=folder/'PROGRESS.json';ns.update(folder=folder,progress=progress);exec(compile(code,'<actual C24 checkpoint>','exec'),ns);ns['checkpoint']();ns['checkpoint'](True);p=json.loads(progress.read_text());cp=torch.load(folder/p['checkpoint'],weights_only=True)
   self.assertTrue(cp['done']);self.assertIsNone(cp['optimiser']);self.assertEqual(cp['rounds'],3);self.assertEqual(cp['accepted_updates'],2);self.assertEqual(cp['head_sha256'],head_hash(head));self.assertEqual(p['checkpoint_sha256'],sha(folder/p['checkpoint']));self.assertEqual(len(list(folder.glob('checkpoint_*.pt'))),1)

 def test_actual_checkpoint_restore_and_domain_guard(self):
  from test_c24_parent import cohort
  head=StateWriter(1,2,2);state=copy.deepcopy(head.state_dict());expected=head_hash(head);device='cpu';train_ids=[];START_HEAD_SHA='fixture';extra_norm={'mean':torch.zeros(70),'scale':torch.ones(70)};initial_quality=quality(cohort());current=cohort();rounds=accepted=0;history=[];stop=None
  sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  text=(ROOT/'worker_c24.py').read_text();a=text.index('  if progress.exists():');b=text.index('  else:checkpoint()',a);body=textwrap.dedent(text[a:b]);code='def actual_restore():\n'+textwrap.indent(body,' ')+'\n return rounds,accepted,current\n';ns=dict(globals());ns.update(locals())
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);progress=folder/'PROGRESS.json';cp_path=folder/'checkpoint_000003.pt';cp={'head':state,'train_ids':train_ids,'start_head_sha256':START_HEAD_SHA,'rounds':3,'accepted_updates':2,'history':[],'current_training':current,'extra_norm':extra_norm,'stop':None,'head_sha256':expected}
   save_tensor(cp_path,cp);dump(progress,{'checkpoint':cp_path.name,'checkpoint_sha256':sha(cp_path),'rounds':3});ns.update(folder=folder,progress=progress);exec(compile(code,'<actual C24 restore>','exec'),ns)
   set_layer_matrix(head,2,layer_matrix(head,2)+.01)
   with redirect_stdout(io.StringIO()):r=ns['actual_restore']()
   self.assertEqual(r[:2],(3,2));self.assertEqual(head_hash(head),expected)
   cp['head']['experts.1.2.bias'][0]+=.01;save_tensor(cp_path,cp);dump(progress,{'checkpoint':cp_path.name,'checkpoint_sha256':sha(cp_path),'rounds':3})
   with self.assertRaisesRegex(RuntimeError,'unselected/frozen'):ns['actual_restore']()
 def test_actual_worker_import_with_only_HF_boundary_stubbed(self):
  import importlib,sys,types
  tf=types.ModuleType('transformers');tf.__version__='4.56.2';tf.AutoTokenizer=object;tf.AutoModelForCausalLM=object
  pe=types.ModuleType('peft');pe.__version__='0.17.1';pe.LoraConfig=object;pe.get_peft_model=lambda *args:None
  with patch.dict(sys.modules,{'transformers':tf,'peft':pe}):m=importlib.import_module('worker_c24')
  self.assertTrue(callable(m.main));self.assertEqual(m.PROTOCOL,'GPT2_C24_RELINEARISED_CALIBRATION_A100_V1')

 def test_actual_physical_evaluator_emits_candidate_parameter_identity(self):
  import types
  device='cpu';params=[];model=tok=anchors=exact=dev=None;ids=[0,1];response_ids=list(range(6));logits0=torch.zeros(2,6);response0=torch.zeros(6);sync=frozen_model_check=lambda:None
  audit=types.SimpleNamespace(content_bits=lambda *a:[1,0,0,1],evaluate=lambda *a:{'exact_margins':[.1]*4,'heldout_margins':[.1]*12})
  w=types.SimpleNamespace(assign_flat=lambda *a:None,anchor_metrics=lambda *a:(.001,0.),next_logits=lambda *a:torch.full((2,6),.001))
  from c24_protocol import endpoint_gate,episode_gate
  text=(ROOT/'worker_c24.py').read_text();a=text.index(' @torch.no_grad()\n def evaluate');b=text.index(' with ExitStack()',a);code=textwrap.dedent(text[a:b]);ns=dict(globals());ns.update(locals());exec(compile(code,'<actual C24 physical evaluator>','exec'),ns)
  theta=torch.tensor([.1,.2]);r=ns['evaluate'](theta,[1,0,0,1],[1,0,0,0]);self.assertEqual(r['theta_sha256'],tensor_hash(theta));self.assertTrue(r['episode_pass']);self.assertEqual(set(r['gates']),set(GATES))

if __name__=='__main__':
 torch.set_num_threads(1);unittest.main()
