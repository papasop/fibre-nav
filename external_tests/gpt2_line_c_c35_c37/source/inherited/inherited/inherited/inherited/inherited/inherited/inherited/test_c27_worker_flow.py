"""Full worker with actual writer dimensions and a deterministic synthetic endpoint.
Only pretrained loading/A100 hardware are mocked. Not a GPT-2 experiment.
"""
import sys,types,hashlib,random,json,tempfile,importlib,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from c27_protocol import *
from c27_features import state_hash,tensor_hash
from raw_writer import StateWriter


class FakeModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.lora_A=torch.nn.Parameter(torch.zeros(8));self.lora_B=torch.nn.ModuleDict({'default':torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(2304,8)),torch.nn.Parameter(torch.zeros(2304,8))])})
  self.config=types.SimpleNamespace(_commit_hash=REVISION,use_cache=False)


def main():
 torch.set_num_threads(2);torch.manual_seed(123)
 fakew=types.ModuleType('writer_core');fakeaudit=types.ModuleType('run_l5_audit');fakeruntime=types.ModuleType('runtime_a100')
 fakew.ANCHORS=['A%d'%i for i in range(40)];tokenizer=types.SimpleNamespace(vocab_size=200);readids=[5,6]
 cfg=json.loads((Path(__file__).parent/'legacy/config_audit.json').read_text());rr=random.Random(82931);rr.sample(fakew.ANCHORS,cfg['anchor_count']);responseids=rr.sample(list(range(100,200)),cfg['response_dim'])
 calls={'load':0,'model_forwards':0};active={}
 def make_model(cfg,device):
  calls['load']+=1;m=FakeModel().to(device);active['model']=m;return tokenizer,m,list(m.lora_B['default'])
 def flat(params):return torch.cat([p.detach().reshape(-1) for p in params])
 def assign(params,v):
  with torch.no_grad():
   for i,p in enumerate(params):p.copy_(v[i*18432:(i+1)*18432].view_as(p))
 def logits(theta,prompts):
  result=torch.zeros(len(prompts),200)
  if prompts[0].startswith('A'):
   residual=theta[4:10]+theta[3]*theta[10]
   result[:,responseids]=residual
  else:
   for j in range(len(prompts)):
    bitlogit=2*theta[j%4]-1;result[j,5]=-bitlogit/2;result[j,6]=bitlogit/2
  return result
 def next_logits(model,tok,prompts,device):
  calls['model_forwards']+=1;return logits(flat(list(model.lora_B['default'])),prompts)
 fakew.make_model=make_model;fakew.set_seed=lambda seed:torch.manual_seed(seed);fakew.param_flat=flat;fakew.assign_flat=assign;fakew.next_logits=next_logits
 fakew.single_token_pair=lambda tok:((' red',5),(' blue',6))
 fakeaudit.prompts=lambda:(['00','01','10','11'],['E%d'%i for i in range(4)],[],['D%d'%i for i in range(12)])
 fakeruntime.a100_device=lambda torch:(torch.device('cpu'),'CPU_SYNTHETIC_NOT_A100')
 mockmods={'writer_core':fakew,'run_l5_audit':fakeaudit,'runtime_a100':fakeruntime,
           'transformers':types.SimpleNamespace(__version__='4.56.2'),'peft':types.SimpleNamespace(__version__='0.17.1')}
 with patch.dict(sys.modules,mockmods):
  sys.modules.pop('worker_c27',None);worker=importlib.import_module('worker_c27')
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp)/'session/run';out.mkdir(parents=True)
   data=[];new=[];ids=[];original=[];baseline={};Jdata={}
   base=StateWriter().eval()
   with torch.no_grad():
    for route,case in enumerate(CASES):
     base.experts[route][0].weight.zero_();base.experts[route][0].bias.fill_(20.)
     base.experts[route][2].weight.zero_();base.experts[route][2].bias.zero_()
     target=STARTS[case][0]^STARTS[case][1]
     base.experts[route][2].bias[3]=(target-STARTS[case][3])/SETTINGS['output_scale']
   hsha=state_hash(base.state_dict());model0=FakeModel();params=list(model0.lora_B['default'])
   norm={'mean':torch.zeros(3077),'scale':torch.ones(3077),'output_scale':torch.tensor(SETTINGS['output_scale'])};extra={'mean':torch.zeros(70),'scale':torch.ones(70)}
   def save(path,value):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save(value,path)
   def sample(case,i,cohort):
    route=CASES.index(case);sid=case+('__v%d'%i if cohort=='training' else '__c25_new%d'%i)
    q=(.006 if i>=17 else .001) if case=='xor_10' and cohort=='training' else ([.001,.0025,.004,.007][i] if case=='xor_10' else .0005)
    prep=torch.zeros(36864);prep[:4]=torch.tensor(STARTS[case]);prep[10]=q;prep[4:10]=-prep[3]*prep[10]
    f=torch.zeros(3077);f[0]=q*100;f[3072]=2*STARTS[case][0]-1;f[3073]=2*STARTS[case][1]-1;f[3076]=STARTS[case][0]^STARTS[case][1]
    x=torch.cat([f,torch.zeros(70)]);route_t=torch.tensor(route)
    folder=out/'data'/cohort/sid;save(folder/'prepared.pt',prep);save(folder/'INPUT.pt',{'input':x,'route':route_t,'features':f,**({'extras':torch.zeros(70)} if i%2 else {})});dump(folder/'REFERENCE.json',{'valid':True})
    with torch.no_grad():theta=prep+base(x.unsqueeze(0),route_t.reshape(1))[0]*SETTINGS['output_scale']
    target=STARTS[case][:3]+[STARTS[case][0]^STARTS[case][1]]
    el=logits(theta,['E0','E1','E2','E3'])[:,readids];dl=logits(theta,['D%d'%j for j in range(12)])[:,readids];al=logits(theta,['A0']*cfg['anchor_count'])
    e=al[:,responseids].mean(0);kl=max(0.,float((torch.softmax(torch.zeros_like(al),-1)*(torch.log_softmax(torch.zeros_like(al),-1)-torch.log_softmax(al,-1))).sum(-1).mean()))
    bits=el.argmax(-1).tolist();em=[float(row[b]-row[1-b]) for row,b in zip(el,target)];dm=[float(row[b]-row[1-b]) for row,b in zip(dl,target*3)]
    gates={'bits_match':bits==target,'output_flip':bits[3]!=STARTS[case][3],'unselected_preserved':bits[:3]==STARTS[case][:3],
           'response':float(e.abs().max())<=.002,'KL':kl<=.005,'exact_read_margin':min(em)>=.05,'development_read_margin':min(dm)>=.05}
    metrics={'theta_sha256':tensor_hash(theta),'bits':bits,'drift':float(e.abs().max()),'kl':kl,'gates':gates,'response_residual':e.double().tolist(),
             'evaluation':{'exact_margins':em,'heldout_margins':dm},'episode_pass':all(gates.values())}
    r={'reference_valid':True,'metrics':metrics}
    if cohort=='training':
     baseline[sid]=r
     if case=='xor_10':
      J=torch.zeros(6,36864)
      for k in range(6):J[k,4+k]=1.;J[k,3]=theta[10];J[k,10]=theta[3]
      with torch.no_grad():H=torch.cat([base.experts[2][1](base.experts[2][0](x)),torch.ones(1)])
      save(out/'geometry'/(sid+'.pt'),{'id':sid,'head_sha256':hsha,'input_sha256':tensor_hash(x),'theta':theta,'J':J,'H':H,'residual':e.double()})
    else:dump(folder/'C25_COMPLETE.json',{'reference_valid':True,'arms':{'calibrated':metrics}})
    return {'id':sid,'case':case}
   for case in CASES:
    for i in range(19):
     t=sample(case,i,'training');data.append(t);ids.append(t['id'])
     if i<15:original.append({'id':t['id']})
    for i in range(4):new.append(sample(case,i,'inspected'))
   require(sum(complete(r) for r in baseline.values())==74,'Synthetic baseline must be74, not a vacuous all-pass')
   cp={'head':base.state_dict(),'extra_norm':extra};save(out/'parent/C25_FINAL_CHECKPOINT.pt',cp);cp_sha=file_sha(out/'parent/C25_FINAL_CHECKPOINT.pt')
   save(out/'parent/C9_CHECKPOINT.pt',{'normalisation':norm})
   dump(out/'parent/SPLIT.json',{'training':data,'original60':original,'prospective_new_states':new});dump(out/'parent/FINAL_TRAINING.json',baseline)
   pins={'lora_A_sha256':hashlib.sha256(model0.lora_A.detach().numpy().tobytes()).hexdigest(),'chart_shapes':[[2304,8],[2304,8]],'symbols':[[' red',5],[' blue',6]]}
   dump(out/'parent/MODEL_PINS.json',pins)
   dump(out/'PARENT_PROVENANCE.json',{'members':{p.relative_to(out).as_posix():{'sha256':file_sha(p)} for p in out.rglob('*') if p.is_file()}})
   dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(worker.ROOT),'parent_C25_sha256':PARENT_SHA})
   actual_load=worker.load
   def guarded_load(path):
    if '/data/inspected/' in str(path):assert (out/'WRITERS_FROZEN.json').exists(),'Development tensor accessed before both writers freeze'
    return actual_load(path)
   with patch.object(worker,'HEAD_SHA',hsha),patch.object(worker,'PARENT_CP_SHA',cp_sha),patch.object(worker,'load',guarded_load),patch.object(torch.cuda,'synchronize',lambda *a,**k:None),patch.object(sys,'argv',['worker','--out',str(out)]):
    worker.main()
    summary=json.loads((out/'SUMMARY.json').read_text());assert summary['base_head_unchanged'] and summary['all_initial74_passes_preserved']
    assert summary['training']['state_bypass']['passed']>=74
    first_matrices={arm:file_sha(out/'arms'/arm/'FINAL_CHECKPOINT.pt') for arm in ARMS}
    # Resume after a simulated interruption of final packaging: no arm is refitted.
    (out/'SUMMARY.json').unlink();worker.main()
    assert all(file_sha(out/'arms'/arm/'FINAL_CHECKPOINT.pt')==h for arm,h in first_matrices.items())
    worker.main() # Completed skip: no model reload.
   assert calls['load']==2
   assert state_hash(base.state_dict())==hsha
   print('SYNTHETIC_WORKER_FLOW_PASS',summary['training'],summary['inspected_development'])
   print('GPU/model loader mocked; actual full-size residual matrices, numerical solve, physical synthetic forwards,guarded selection,freeze order,and resume exercised. Not pretrained GPT2 evidence.')


if __name__=='__main__':main()
