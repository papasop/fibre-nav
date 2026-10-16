"""Full worker with actual writer dimensions and a deterministic synthetic endpoint.
Only pretrained loading/A100 hardware are mocked. Not a GPT-2 experiment.
"""
import sys,types,hashlib,random,json,tempfile,importlib,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from c29_protocol import *
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
 cfg=json.loads((Path(__file__).parent/'inherited/inherited/legacy/config_audit.json').read_text());rr=random.Random(82931);rr.sample(fakew.ANCHORS,cfg['anchor_count']);responseids=rr.sample(list(range(100,200)),cfg['response_dim'])
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
   residual=theta[4:10]+theta[3]*theta[10]+10*theta[11].square()
   result[:,responseids]=residual
  else:
   for j in range(len(prompts)):
    bitlogit=2*theta[j%4]-1;result[j,5]=-bitlogit/2;result[j,6]=bitlogit/2
  return result
 def next_logits(model,tok,prompts,device):
  calls['model_forwards']+=1;return logits(flat(list(model.lora_B['default'])),prompts)
 def response_vector(model,tok,prompts,responseids,device):
  theta=torch.cat([p.reshape(-1) for p in model.lora_B['default']]);return theta[4:10]+theta[3]*theta[10]+10*theta[11].square()
 fakew.response_vector=response_vector
 fakew.make_model=make_model;fakew.set_seed=lambda seed:torch.manual_seed(seed);fakew.param_flat=flat;fakew.assign_flat=assign;fakew.next_logits=next_logits
 fakew.single_token_pair=lambda tok:((' red',5),(' blue',6))
 fakeaudit.prompts=lambda:(['00','01','10','11'],['E%d'%i for i in range(4)],[],['D%d'%i for i in range(12)])
 fakeruntime.a100_device=lambda torch:(torch.device('cpu'),'CPU_SYNTHETIC_NOT_A100')
 mockmods={'writer_core':fakew,'run_l5_audit':fakeaudit,'runtime_a100':fakeruntime,
           'transformers':types.SimpleNamespace(__version__='4.56.2'),'peft':types.SimpleNamespace(__version__='0.17.1')}
 with patch.dict(sys.modules,mockmods):
  sys.modules.pop('worker_c29',None);worker=importlib.import_module('worker_c29')
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
    q=(.0005+i*.00004) if cohort=='training' else [.001,.0025,.004,.007][i]
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
    else:dump(folder/'C27_COMPLETE.json',{'reference_valid':True,'arms':{'state_bypass':{'reference_valid':True,'metrics':metrics}}})
    return {'id':sid,'case':case}
   for case in CASES:
    for i in range(19):
     t=sample(case,i,'training');data.append(t);ids.append(t['id'])
     if i<15:original.append({'id':t['id']})
    for i in range(4):new.append(sample(case,i,'inspected'))
   require(sum(complete(r) for r in baseline.values())==76,'Synthetic frozen C27 baseline must be76')
   cp={'head':base.state_dict(),'extra_norm':extra};save(out/'parent/C25_FINAL_CHECKPOINT.pt',cp);cp_sha=file_sha(out/'parent/C25_FINAL_CHECKPOINT.pt')
   save(out/'parent/C9_CHECKPOINT.pt',{'normalisation':norm})
   dump(out/'parent/SPLIT.json',{'training':data,'original60':original,'prospective_new_states':new});dump(out/'parent/FINAL_TRAINING.json',baseline)
   from c27_features import fit_map
   fm0=fit_map(torch.stack([torch.load(out/'data/training'/t['id']/'INPUT.pt',weights_only=True)['input'] for t in data if t['case']=='xor_10']).numpy(),[t['id'] for t in data if t['case']=='xor_10'],'state_bypass',8)
   zero=torch.zeros(36864,9);save(out/'parent/C27_BRANCH.pt',{'residual_matrix':zero,'training_rows':baseline});save(out/'parent/C27_FEATURE_MAP.pt',fm0)
   branch_sha=file_sha(out/'parent/C27_BRANCH.pt');matrix_sha=tensor_hash(zero)
   from c28_features import feature_map
   from c27_features import map_features
   fakepins={};arm_rows={};zero_sha=tensor_hash(zero)
   def endpoint(theta,case):
    target=STARTS[case][:3]+[STARTS[case][0]^STARTS[case][1]]
    el=logits(theta,['E0','E1','E2','E3'])[:,readids];dl=logits(theta,['D%d'%j for j in range(12)])[:,readids];al=logits(theta,['A0']*cfg['anchor_count']);e=al[:,responseids].mean(0)
    kl=max(0.,float((torch.softmax(torch.zeros_like(al),-1)*(torch.log_softmax(torch.zeros_like(al),-1)-torch.log_softmax(al,-1))).sum(-1).mean()))
    bits=el.argmax(-1).tolist();em=[float(row[b]-row[1-b]) for row,b in zip(el,target)];dm=[float(row[b]-row[1-b]) for row,b in zip(dl,target*3)]
    gates={'bits_match':bits==target,'output_flip':bits[3]!=STARTS[case][3],'unselected_preserved':bits[:3]==STARTS[case][:3],'response':float(e.abs().max())<=.002,'KL':kl<=.005,'exact_read_margin':min(em)>=.05,'development_read_margin':min(dm)>=.05}
    return {'reference_valid':True,'metrics':{'theta_sha256':tensor_hash(theta),'bits':bits,'drift':float(e.abs().max()),'kl':kl,'gates':gates,'response_residual':e.double().tolist(),'evaluation':{'exact_margins':em,'heldout_margins':dm},'episode_pass':all(gates.values())}}
   points={}
   for cohort,items in [('training',data),('inspected',new)]:
    for t in items:
     f=torch.load(out/'data'/cohort/t['id']/'INPUT.pt',weights_only=True);prep=torch.load(out/'data'/cohort/t['id']/'prepared.pt',weights_only=True)
     with torch.no_grad():theta0=prep+base(f['input'].unsqueeze(0),f['route'].reshape(1))[0]*SETTINGS['output_scale']
     points[t['id']]=(theta0,f)
     if cohort=='training':
      J=torch.zeros(6,36864,dtype=torch.float64)
      for k in range(6):J[k,4+k]=1;J[k,3]=theta0[10];J[k,10]=theta0[3]
      save(out/'geometry'/(t['id']+'.pt'),{'J':J,'residual':torch.tensor(baseline[t['id']]['metrics']['response_residual'],dtype=torch.float64),'theta_sha256':tensor_hash(theta0),'input_sha256':tensor_hash(f['input']),'C27_matrix_sha256':zero_sha})
   for arm in ARMS:
    mats=torch.zeros(4,36864,9);maps=[];fr={'arm':arm,'feature_map_sha256':[]}
    for route,case in enumerate(CASES):
     ids=[t['id'] for t in data if t['case']==case];xs=torch.stack([points[sid][1]['input'] for sid in ids]).numpy();fm=feature_map(xs,ids,arm);maps.append(fm);fp=out/'parent'/arm/case/'FEATURE_MAP.pt';save(fp,fm);fr['feature_map_sha256'].append(file_sha(fp))
     sign=(STARTS[case][0]^STARTS[case][1])-STARTS[case][3]
     mats[route,4:10,-1]=(-sign if arm=='pca_control' else sign)*.0002/SETTINGS['output_scale'];mats[route,11,-1]=.005/SETTINGS['output_scale']
    rows={}
    for t in data:
     theta0,f=points[t['id']];ta=theta0+mats[int(f['route']),:,-1]*SETTINGS['output_scale'];rows[t['id']]=endpoint(ta,t['case'])
    acp={'arm':arm,'matrices':mats,'feature_maps':maps,'parent_matrix_sha256':zero_sha,'training_rows':rows,'done':True};cpfile=out/'parent'/arm/'FINAL_CHECKPOINT.pt';save(cpfile,acp)
    fr.update(checkpoint_sha256=file_sha(cpfile),matrices_sha256=tensor_hash(mats));dump(out/'parent'/arm/'FROZEN.json',fr);fakepins[arm]={k:fr[k] for k in ('checkpoint_sha256','matrices_sha256')};arm_rows[arm]=(mats,rows)
   for t in new:
    sid=t['id'];theta0,f=points[sid];arms={'c27':endpoint(theta0,t['case'])}
    for arm,(mats,rows) in arm_rows.items():arms[arm]=endpoint(theta0+mats[int(f['route']),:,-1]*SETTINGS['output_scale'],t['case'])
    dump(out/'data/inspected'/sid/'C28_COMPLETE.json',{'reference_valid':True,'arms':arms})
   dump(out/'parent/C28_BASELINE_REPLAY.json',baseline)
   pins={'lora_A_sha256':hashlib.sha256(model0.lora_A.detach().numpy().tobytes()).hexdigest(),'chart_shapes':[[2304,8],[2304,8]],'symbols':[[' red',5],[' blue',6]]}
   dump(out/'parent/MODEL_PINS.json',pins)
   dump(out/'PARENT_PROVENANCE.json',{'members':{p.relative_to(out).as_posix():{'sha256':file_sha(p)} for p in out.rglob('*') if p.is_file()}})
   dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(worker.ROOT)})
   actual_load=worker.load
   def guarded_load(path):
    if '/data/inspected/' in str(path):assert (out/'FROZEN_IDENTITIES.json').exists(),'Inspected tensor read before frozen identities recorded'
    return actual_load(path)
   with patch.object(worker,'HEAD_SHA',hsha),patch.object(worker,'PARENT_CP_SHA',cp_sha),patch.object(worker,'PARENT_BRANCH_CP_SHA',branch_sha),patch.object(worker,'PARENT_MATRIX_SHA',matrix_sha),patch.object(worker,'PINS',fakepins),patch.object(worker,'load',guarded_load),patch.object(torch.cuda,'synchronize',lambda *a,**k:None),patch.object(sys,'argv',['worker','--out',str(out)]):
    worker.main()
    summary=json.loads((out/'SUMMARY.json').read_text());assert summary['all_writers_unchanged'] and summary['all_parent_endpoint_records_replayed']
    assert summary['committed_diagnostic_work']['new_committed_inspected_Jacobians']==16
    assert summary['committed_diagnostic_work']['diagnostic_backward_rows']==96
    assert summary['committed_diagnostic_work']['midpoint_response_probes']==32
    assert summary['direction_diagnosis']['training']['pca_control']['n']==76
    identities=copy.deepcopy(summary['frozen_identities'])
    (out/'SUMMARY.json').unlink();worker.main()
    resumed=json.loads((out/'SUMMARY.json').read_text());assert resumed['frozen_identities']==identities
    assert resumed['committed_diagnostic_work']==summary['committed_diagnostic_work']
    worker.main()
   assert calls['load']==2
   assert state_hash(base.state_dict())==hsha
   print('SYNTHETIC_WORKER_FLOW_PASS',summary['direction_diagnosis'])
   print('GPU/model loader mocked; actual full-size four-route matrices, 16 diagnostic autograd Jacobians, immutable weights, archived endpoint replay, actual displacement Taylor accounting, fixed midpoint probes and resume exercised; no fitting or candidate selection. Not pretrained GPT2 evidence.')


if __name__=='__main__':main()
