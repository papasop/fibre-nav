import copy,hashlib,io,json,tempfile,textwrap,time,unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import torch
from test_c18_torch import Tests as InheritedIsolationTests
from raw_writer import StateWriter,one_raw_proposal
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from radius_calibration import scale_direction
from joint_calibration import response_prediction
from c23_protocol import candidate_specs,merge_route,guard,check_replay,candidate_key,quality
import c23_protocol as p
ROOT=Path(__file__).parent

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def head_hash(h):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in h.parameters())).hexdigest()
def dump(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj))
def save_tensor(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(obj,path)

class ScanTorchTests(unittest.TestCase):
 def setup_scan(self):
  torch.manual_seed(2301);head=StateWriter(1,2,2)
  for route in range(4):set_layer_matrix(head,route,torch.tensor([[0.,0.,.09],[0.,0.,1.]]))
  for v in head.parameters():v.requires_grad_(False)
  data=[{'id':t['id'],'case':t['case'],'input':torch.tensor([.1]),'route':torch.tensor(p.CASES.index(t['case'])),'prep':torch.zeros(2),'reference_valid':True} for t in p.training_specs()]
  def metrics(theta):
   d=float(theta[0]**2);content=abs(float(theta[1])-1)<1e-6;g={k:content for k in p.GATES};g['response']=d<=.002
   return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values()),'response_residual':[d],
    'theta_sha256':tensor_hash(theta),'float64_aggregation_gate_diagnostic':d<=.002,'primary_response_gate':d<=.002,'maximum_aggregation_difference':0.}
  def measured():return {t['id']:{'reference_valid':True,'metrics':metrics(one_raw_proposal(head,t['input'],t['route'],torch.tensor(1.)))} for t in data}
  return head,data,metrics,measured
 def exercise_scan(self,fail=False):
  head,data,metrics,measured=self.setup_scan();current=measured();case=p.SETTINGS['scan_case'];route=p.CASES.index(case);group=[t for t in data if t['case']==case];group_ids=[t['id'] for t in group];old_layer=layer_matrix(head,route);origin=head_hash(head);states=[]
  for t in group:
   theta=one_raw_proposal(head,t['input'],t['route'],torch.tensor(1.));states.append({'theta':theta.clone()})
  H=np.stack([hidden_augmented(head,t['input'],route).numpy() for t in group]);J=np.repeat(np.array([[[.18,0.]]]),15,axis=0);E=np.full((15,1),.0081);scale=1.;bases=[np.array([[0.,0.,-.045],[0.,0.,0.]]) for _ in range(3)]
  system={'J':J,'H':H,'E':E,'scale':scale,'info':{}};calls={'evaluate':0,'restore':0,'zero':[]};c9norm={'output_scale':torch.tensor(1.)};START_HEAD_SHA=origin;SETTINGS=p.SETTINGS;starts={case:[]};targets={case:[]};invariant_check=lambda *args:None
  def restore_origin():set_layer_matrix(head,route,old_layer);calls['restore']+=1;self.assertEqual(head_hash(head),origin)
  def zero_phase(phase):calls['zero'].append(phase);restore_origin()
  def evaluate(theta,*args):
   calls['evaluate']+=1
   if fail:raise RuntimeError('injected interrupted evaluation')
   return metrics(theta)
  snapshot=lambda *args:None;args=type('Args',(),{'drive_dir':''})()
  s=(ROOT/'worker_c23.py').read_text();a=s.index("  zero_phase('before_scan');records=[]");b=s.index('  best=None;',a);block=textwrap.dedent(s[a:b]);code='def actual_scan():\n'+textwrap.indent(block,' ')+'\n return records,eligible\n'
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);namespace=dict(globals());namespace.update(locals());exec(compile(code,'<actual C23 scan>','exec'),namespace)
   if fail:
    with self.assertRaisesRegex(RuntimeError,'injected'):namespace['actual_scan']()
    self.assertEqual(head_hash(head),origin);return
   with redirect_stdout(io.StringIO()):records,eligible=namespace['actual_scan']()
   self.assertEqual(len(records),90);unique=len({r['candidate_head_sha256'] for r in records});self.assertEqual(calls['evaluate'],15*unique);self.assertLess(unique,90);self.assertEqual(head_hash(head),origin)
   frozen=copy.deepcopy(head.state_dict());calls['evaluate']=0
   with redirect_stdout(io.StringIO()):again,eligible_again=namespace['actual_scan']()
   self.assertEqual(calls['evaluate'],0);self.assertEqual(records,again);self.assertEqual(len(eligible),len(eligible_again));self.assertTrue(all(torch.equal(v,frozen[n]) for n,v in head.state_dict().items()))
   return namespace,records,eligible
 def test_actual_scan_all90_aliases_and_resume(self):self.exercise_scan()
 def test_actual_scan_restores_origin_after_exception(self):self.exercise_scan(True)
 def exercise_audit(self,fail=False):
  namespace,records,eligible=self.exercise_scan();head=namespace['head'];origin=namespace['START_HEAD_SHA'];calls={'full':0};old_layer=namespace['old_layer'];case=namespace['case'];route=namespace['route'];data=namespace['data'];metrics=namespace['metrics']
  def measure_training(writer):
   calls['full']+=1;rows={t['id']:{'reference_valid':True,'metrics':metrics(one_raw_proposal(writer,t['input'],t['route'],torch.tensor(1.)))} for t in data}
   if fail:rows[data[0]['id']]['metrics']['gates']['KL']=False
   return rows
  s=(ROOT/'worker_c23.py').read_text();a=s.index('  best=None;');b=s.index("  zero_phase('after_scan')",a);block=textwrap.dedent(s[a:b]);code='def actual_audit():\n'+textwrap.indent(block,' ')+'\n return best,audits\n'
  with tempfile.TemporaryDirectory() as d:
   namespace.update(out=Path(d),measure_training=measure_training,eligible=eligible);exec(compile(code,'<actual C23 audit>','exec'),namespace);best,audits=namespace['actual_audit']();self.assertEqual(head_hash(head),origin)
   if fail:self.assertIsNone(best);self.assertEqual(calls['full'],len({c['record']['candidate_head_sha256'] for c in eligible}))
   else:
    self.assertTrue(best['full60_verified']);self.assertFalse(best['head_committed']);self.assertEqual(calls['full'],1);self.assertTrue((Path(d)/'best_candidate'/'CANDIDATE_LAYER.pt').exists());self.assertFalse((Path(d)/'WRITER_FROZEN.json').exists())
 def test_ranked_audit_verifies_full60_and_restores_no_commit(self):self.exercise_audit()
 def test_rejected_audits_not_retried_for_aliases(self):self.exercise_audit(True)
 def test_float64_aggregation_can_differ_without_changing_primary_rule(self):
  s=(ROOT/'worker_c23.py').read_text();a=s.index(' @torch.no_grad()\n def response_probe');b=s.index(' @torch.no_grad()\n def evaluate',a);code=textwrap.dedent(s[a:b]);params=[torch.zeros(1)];device='cpu';model=tok=anchors=None;response_ids=[0]
  logits0=torch.tensor([[100.0,0.],[100.1,1.],[100.2,2.]],dtype=torch.float32);al=logits0.clone();al[:,0]+=torch.tensor([.002,.002,.00201]);response0=logits0[:,response_ids].mean(0)
  class W:
   @staticmethod
   def assign_flat(ps,v):ps[0].copy_(v)
   @staticmethod
   def param_flat(ps):return ps[0].clone()
   @staticmethod
   def next_logits(*args):return al.clone()
  w=W();sync=lambda:None;frozen_model_check=lambda:None;namespace=dict(globals());namespace.update(locals());exec(compile(code,'<actual C23 probe>','exec'),namespace);r=namespace['response_probe'](torch.zeros(1))
  self.assertEqual(r['primary_response_gate'],r['drift']<=.002);self.assertEqual(r['float64_aggregation_gate_diagnostic'],r['float64_aggregation_drift']<=.002);self.assertFalse(r['float64_changes_primary_gate']);self.assertNotEqual(r['primary_response_gate'],r['float64_aggregation_gate_diagnostic']);self.assertGreater(r['maximum_aggregation_difference'],0.);self.assertEqual(r['full_logits_sha256'],tensor_hash(al))
 def test_tiny_float32_parameter_update_can_be_identical(self):
  old=torch.ones(2,3);trial=(old.double()+1e-10).float();self.assertTrue(torch.equal(old,trial))

if __name__=='__main__':unittest.main()
