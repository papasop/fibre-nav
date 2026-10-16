import copy,hashlib,io,json,tempfile,textwrap,unittest
from pathlib import Path
import numpy as np
import torch
from test_c18_torch import Tests
from raw_writer import StateWriter,one_raw_proposal
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from joint_calibration import prepare_system,solve_direction
from radius_calibration import ridge_coefficients,reconstruct_direction,scale_direction
from radius_guard import select_candidate,merge_route,check_replay
import c21_protocol as p

class ComparisonTorchTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(19031)
 def test_later_candidate_commit_does_not_leave_last_tried_layer(self):
  h=StateWriter(1,2,2)
  for route in range(4):set_layer_matrix(h,route,torch.tensor([[0.,0.,.2],[0.,0.,1.]]))
  for v in h.parameters():v.requires_grad_(False)
  data=[(t['id'],t['case'],torch.tensor([.1])) for t in p.training_specs()]
  def measure():
   rows={}
   for sid,case,x in data:
    v=one_raw_proposal(h,x,torch.tensor(p.CASES.index(case)),torch.tensor(1.));drift=float(v[0]**2);content=abs(float(v[1])-1)<1e-6
    gates={g:content for g in p.GATES};gates['response']=drift<=.002
    rows[sid]={'reference_valid':True,'metrics':{'drift':drift,'kl':.001,'gates':gates,'episode_pass':all(gates.values())}}
   return rows
  current=measure();old=layer_matrix(h,0);others={n:v.clone() for n,v in h.state_dict().items() if not n.startswith('experts.0.2.')};candidates=[]
  for ri,value in enumerate([.1,.02,.08]):
   layer=old.clone();layer[0,-1]=value;set_layer_matrix(h,0,layer);rows=measure()
   candidates.append({'rows':rows,'layer':layer,'radius_index':0,'ridge_index':ri,'backtrack_index':0})
  winner=select_candidate(current,candidates);self.assertEqual(winner['ridge_index'],1)
  set_layer_matrix(h,0,winner['layer']);audited=measure();check_replay(winner['rows'],audited)
  self.assertEqual(len(p.quality(audited)['passed_ids']),15)
  self.assertTrue(all(torch.equal(h.state_dict()[n],v) for n,v in others.items()))
  buf=io.BytesIO();torch.save({'head':h.state_dict(),'rounds':1,'chosen_ridge_index':1},buf);buf.seek(0)
  cp=torch.load(buf,weights_only=True);h2=StateWriter(1,2,2);h2.load_state_dict(cp['head'])
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(h.parameters(),h2.parameters())))
 def test_route_merged_measurements_equal_full_physical_fixture(self):
  h=StateWriter(1,2,2);x=torch.tensor([.3]);before=[one_raw_proposal(h,x,torch.tensor(i),torch.tensor(.07)) for i in range(4)]
  set_layer_matrix(h,2,layer_matrix(h,2)+.01)
  after=[one_raw_proposal(h,x,torch.tensor(i),torch.tensor(.07)) for i in range(4)]
  for i in (0,1,3):self.assertTrue(torch.equal(before[i],after[i]))
  self.assertFalse(torch.equal(before[2],after[2]))
 def test_version_guard_detects_accidental_cross_route_write(self):
  h=StateWriter(1,2,2);versions={n:v._version for n,v in h.named_parameters() if not n.startswith('experts.0.2.')}
  set_layer_matrix(h,1,layer_matrix(h,1)+.01)
  self.assertTrue(any(v._version!=versions[n] for n,v in h.named_parameters() if n in versions))
 def test_checkpoint_head_is_actual_winner_not_direction(self):
  h=StateWriter(1,2,2);layer=layer_matrix(h,1);D=torch.full_like(layer,.01,dtype=torch.float64);trial=(layer.double()+.5*D).float()
  set_layer_matrix(h,1,trial);buf=io.BytesIO();torch.save({'head':h.state_dict()},buf);buf.seek(0);state=torch.load(buf,weights_only=True)['head']
  self.assertTrue(torch.equal(state['experts.1.2.weight'],trial[:,:-1]));self.assertTrue(torch.equal(state['experts.1.2.bias'],trial[:,-1]))
 def test_tiny_gram_all_ridges_produce_finite_saved_directions(self):
  r=np.random.default_rng(19);J=r.normal(size=(9,2,4));H=r.normal(size=(9,3));E=r.normal(size=(9,2))*.003;system=prepare_system(J,H,E,.07)
  for ridge in p.SETTINGS['ridge_grid']:
   D,info=solve_direction(system,ridge,.5,.05);buf=io.BytesIO();torch.save({'D':torch.from_numpy(D),'info':info},buf);buf.seek(0);v=torch.load(buf,weights_only=True)
   self.assertTrue(torch.isfinite(v['D']).all());self.assertEqual(v['info']['ridge'],ridge);self.assertLessEqual(v['D'].norm().item(),.5+1e-12)
 def test_half_step_and_full_step_resume_byte_identical(self):
  old=torch.tensor([[1.,2.,3.]],dtype=torch.float32);D=torch.tensor([[.003,.004,.005]],dtype=torch.float64)
  for alpha in (1.,.5):
   trial=(old.double()+alpha*D).float();buf=io.BytesIO();torch.save({'old':old,'D':D,'alpha':alpha,'layer':trial},buf);buf.seek(0);v=torch.load(buf,weights_only=True)
   self.assertTrue(torch.equal((v['old'].double()+v['alpha']*v['D']).float(),v['layer']))
 def run_worker_block(self,break_audit=False):
  """Execute the delivered worker's actual candidate block on a nonlinear fixture."""
  from joint_calibration import response_prediction,linear_protection
  h=StateWriter(1,2,2)
  for route in range(4):set_layer_matrix(h,route,torch.tensor([[0.,0.,.09],[0.,0.,1.]]))
  for v in h.parameters():v.requires_grad_(False)
  data=[{'id':t['id'],'case':t['case'],'input':torch.tensor([.1]),'route':torch.tensor(p.CASES.index(t['case'])),'prep':torch.zeros(2),'reference_valid':True} for t in p.training_specs()]
  def metrics(v):
   d=float(v[0]**2);content=abs(float(v[1])-1)<1e-6;g={k:content for k in p.GATES};g['response']=d<=.002
   return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values()),'response_residual':[d]}
  def measure():return {t['id']:{'reference_valid':True,'metrics':metrics(one_raw_proposal(h,t['input'],t['route'],torch.tensor(1.)))} for t in data}
  current=measure();case=p.CASES[0];route=0;group=[t for t in data if t['case']==case];states=[]
  for t in group:
   theta=one_raw_proposal(h,t['input'],t['route'],torch.tensor(1.));states.append({'theta':theta.clone(),'J':torch.tensor([[2*float(theta[0]),0.]]),'H':hidden_augmented(h,t['input'],route),'residual':torch.tensor([float(theta[0]**2)],dtype=torch.float64)})
  J=np.stack([v['J'].double().numpy() for v in states]);H=np.stack([v['H'].double().numpy() for v in states]);E=np.stack([v['residual'].numpy() for v in states]);scale=1.;system=prepare_system(J,H,E,scale)
  calls={'route':0,'full':0};old=layer_matrix(h,0)
  def evaluate(theta,*args):calls['route']+=1;return metrics(theta)
  def measure_training(writer):
   calls['full']+=1;result=measure()
   if break_audit:result[data[-1]['id']]['metrics']['gates']['KL']=False
   return result
  def dump(path,obj):Path(path).write_text(json.dumps(obj))
  def save_tensor(path,obj):torch.save(obj,path)
  def head_hash(head):return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in head.parameters())).hexdigest()
  before_hash=head_hash(h);q=p.quality(current);source=(Path(__file__).parent/'worker_c21.py').read_text();block=textwrap.dedent(source[source.index('    passing=[i for i,t in enumerate(group)'):source.index('    rounds+=1;')])
  code='def actual_worker_comparison():\n'+textwrap.indent(block,' ')+'\n return updated,chosen,trials\n'
  with tempfile.TemporaryDirectory() as d:
   out=round_dir=Path(d);head=h;old_layer=old;c9norm={'output_scale':torch.tensor(1.)};branches={case:{'trust':1.}};starts={case:[]};targets={case:[]};SETTINGS=p.SETTINGS
   baseline_check=lambda:None;frozen_model_check=lambda:None;guard=p.guard
   namespace=dict(globals());namespace.update(locals());exec(compile(code,'<actual C21 candidate block>','exec'),namespace)
   if break_audit:
    with self.assertRaises(ValueError):namespace['actual_worker_comparison']()
    self.assertTrue(torch.equal(layer_matrix(h,0),old));return
   result,chosen,trials=namespace['actual_worker_comparison']()
   self.assertEqual({v['ridge'] for v in trials},set(p.SETTINGS['ridge_grid']));self.assertEqual(chosen['compared_admissible_pairs'],9);self.assertEqual(len({(t['increment_radius'],t['ridge']) for t in trials}),9)
   self.assertEqual(calls['route'],135);self.assertEqual(calls['full'],1);check_replay(result,measure())
   frozen=copy.deepcopy(h.state_dict());set_layer_matrix(h,0,old);calls.update(route=0,full=0)
   result2,chosen2,trials2=namespace['actual_worker_comparison']()
   self.assertEqual(chosen,chosen2);self.assertEqual(calls,{'route':0,'full':0})
   self.assertTrue(all(torch.equal(v,frozen[n]) for n,v in h.state_dict().items()))
 def test_actual_worker_block_compares_all_ridges_and_resumes_saved_trials(self):self.run_worker_block()
 def test_actual_worker_block_rolls_back_failed_full60_audit(self):self.run_worker_block(True)

class PreparationTorchTests(unittest.TestCase):
 def exercise(self,valid=True,evaluation=False):
  """Real worker preparation control flow; model/geometry callbacks are synthetic."""
  spec=p.test_specs()[0] if evaluation else p.supplemental_training_specs()[0]
  frozen_hash='frozen-test-head' if evaluation else None
  source=(Path(__file__).parent/'worker_c21.py').read_text();a=source.index('  def prepare_state(');b=source.index('\n  def measure_training',a)
  code=textwrap.dedent(source[a:b]);calls={'correction':0,'features':0,'project':0}
  def offline_correct(folder,initial):calls['correction']+=1;folder.mkdir(parents=True,exist_ok=True);return initial,[],0.
  def evaluate(*args):return {'pass':valid,'episode_pass':valid}
  def fresh_inputs(*args):calls['features']+=1;return {'route':torch.tensor(0),'input':torch.ones(3147),'features':torch.ones(3077)}
  def null_project(j,v):calls['project']+=1;return v,{'synthetic':True}
  def vector(v):
   if isinstance(v,dict):v=v['theta']
   assert v.shape==(36864,) and torch.isfinite(v).all();return v.clone()
  def load_obj(*args):return {'J':torch.zeros(6,36864)}
  def dump(path,value):Path(path).write_text(json.dumps(value))
  def save_tensor(path,value):torch.save(value,path)
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);base_data={spec['base_id']:(torch.zeros(36864),torch.ones(36864))};capsule=None;starts={spec['case']:[]};targets={spec['case']:[0,0,1,0]};CASES=p.CASES;START_HEAD_SHA=p.START_HEAD_SHA;baseline_check=lambda:None;frozen_model_check=lambda:None;head=None;head_hash=lambda _:frozen_hash
   namespace=dict(globals());namespace.update(locals());exec(compile(code,'<actual C21 preparation>','exec'),namespace)
   folder,record=namespace['prepare_state'](spec,frozen_hash)
   self.assertEqual(record['valid'],valid);self.assertEqual(record['generation_retries'],0);self.assertFalse(record['replacement_used']);self.assertEqual(calls,{'correction':2,'features':1,'project':1})
   saved=torch.load(folder/'perturbation.pt',weights_only=True);self.assertAlmostEqual(float(saved['theta'].norm()),spec['norm'],places=5)
   self.assertEqual(record['writer_frozen_before_generation'],frozen_hash)
   _,again=namespace['prepare_state'](spec,frozen_hash);self.assertEqual(again,record);self.assertEqual(calls,{'correction':2,'features':1,'project':1})
   if evaluation:
    with self.assertRaises(RuntimeError):namespace['prepare_state'](spec,'different-head')
 def test_new_state_resume_uses_same_saved_perturbation(self):self.exercise(True,True)
 def test_invalid_reference_is_saved_without_replacement(self):self.exercise(False,True)
 def test_final_evaluation_preparation_is_bound_to_frozen_head(self):self.exercise(True,True)

if __name__=='__main__':unittest.main()
