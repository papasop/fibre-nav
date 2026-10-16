"""Frozen centre-transfer protocol; generation is separate from single-proposal writing."""
import argparse,fcntl,contextlib,io
from pathlib import Path
from c37_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
import numpy as np
import torch
from c37_protocol import *
from c37_parent import validate
from c35_fit import records,check_frozen
from c35_writer import ComposedWriter
from worker_c35 import evaluate,counts,proposals
from worker_c36 import main as c36_completed,runtime_dir,deployed_identity
from c36_audit import writer_identity,compare_replay
from c31_runtime import Runtime,load,save_tensor,vector,tensor_hash
from geometry import null_project

def centre(out,c):
 f=Path(out)/'parent_c36/c35/new_states'/c['id'];r=read(f/'COMPLETE.json')
 require(r['preparation_valid'] and r['spec']['case']==c['case'],'Declared C36 centre invalid')
 for n,h in r['files'].items():require(sha(f/n)==h,'Centre evidence changed')
 theta=vector(load(f/'prepared.pt'));require(load(f/'INPUT.pt')['prepared_sha256']==tensor_hash(theta),'Centre state/input mismatch')
 return theta

def audit_parent(out,identity_checker):
 out=Path(out)
 def forbidden(*a):raise AssertionError('Completed C36 audit attempted model loading')
 with contextlib.redirect_stdout(io.StringIO()):s=c36_completed(out/'parent_c36',runtime_factory=forbidden)
 from worker_c36 import collected
 rs=collected(out/'parent_c36');src=out/'parent_c36/c35'
 for r in rs:
  spec=r['spec'];f=src/'new_states'/spec['id'];prep=vector(load(f/'prepared.pt'));inp=load(f/'INPUT.pt')
  v,p,ready=proposals(None,None,src,spec,prep,inp)
  for a in ARMS:require(tensor_hash(v['points'][a])==r['arms'][a]['metrics']['theta_sha256'],'Parent endpoint tensor identity differs')
 for c in centres():centre(out,c)
 return {'complete':True,'C36_new32_recount':s['new_local'],'C36_proposal_tensor_checks':96,
         'actual_core_identity':identity_checker(src),'new_model_evaluations':0,
         'scope':'Archival numeric/tensor/source verification; not an independent new physical replication.'}

def freeze(out):
 out=Path(out);check_frozen(out/'c35')
 value={'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT),'state_specs':specs(),
   'parent_provenance_sha256':sha(out/'PARENT_PROVENANCE.json'),'parent_audit_sha256':sha(out/'PARENT_AUDIT.json'),
   'context_members_sha256':sha(out/'CONTEXT_MEMBERS.json'),'C35_frozen_sha256':sha(out/'c35/COEFFICIENTS_FROZEN.json'),
   'centres':{c['id']:tensor_hash(centre(out,c)) for c in centres()},'new_generation_started':False,'no_refitting':True}
 p=out/'C37_FROZEN.json'
 if p.exists():require(read(p)==value,'Frozen C37 weights/specifications changed')
 else:
  require(not any((out/'c35/new_states').glob('*__c37_new*')),'Freeze must precede generation');dump(p,value)
 return value

def prepare_new(rt,out,spec):
 out=Path(out);freeze(out);data=out/'c35';f=data/'new_states'/spec['id'];f.mkdir(parents=True,exist_ok=True);cp=f/'COMPLETE.json'
 base=centre(out,{'id':spec['centre_id'],'case':spec['case']});base_sha=tensor_hash(base)
 if cp.exists():
  r=read(cp);require(r['spec']==spec and r['centre_sha256']==base_sha and r['C37_frozen_sha256']==sha(out/'C37_FROZEN.json'),'Preparation binding differs')
  for n,h in r['files'].items():require(sha(f/n)==h,'Saved preparation changed')
  return vector(load(f/'prepared.pt')),load(f/'INPUT.pt'),r['preparation_valid']
 qp=f/'PERTURBATION.pt'
 if qp.exists():
  q=load(qp);require(q['spec']==spec and q['centre_sha256']==base_sha,'Perturbation binding differs');delta=vector(q['theta'])
 else:
  J=rt.jacobian(base,out/'centre_geometry'/(spec['centre_id']+'.pt'))
  raw=np.random.default_rng(spec['seed']).normal(size=36864);q,info=null_project(J,raw,SETTINGS['preparation_null_relative_cutoff'],SETTINGS['preparation_null_absolute_cutoff'])
  norm=float(np.linalg.norm(q));require(norm>0,'Zero projected perturbation; no resampling allowed')
  delta=torch.from_numpy((q*(spec['norm']/norm)).astype(np.float32));save_tensor(qp,{'theta':delta,'spec':spec,'centre_sha256':base_sha,'null_projection':info})
 initial=base+delta;p=f/'prepared.pt'
 if p.exists():
  saved=load(p);require(saved['spec']==spec and saved['initial_sha256']==tensor_hash(initial),'Saved prepared state binding differs');prep=vector(saved)
 else:
  prep,history,seconds=rt.correct_preparation(initial,f/'preparation_correction')
  save_tensor(p,{'theta':prep,'spec':spec,'initial_sha256':tensor_hash(initial),'centre_sha256':base_sha})
  dump(f/'PREPARATION_HISTORY.json',{'history':history,'seconds':seconds,'offline_prewrite_preparation_only':True})
 start=STARTS[spec['case']];pm=rt.evaluate(prep,start);desired=start[0]^start[1];ip=f/'INPUT.pt'
 if ip.exists():
  inp=load(ip);require(inp['spec']==spec and inp['prepared_sha256']==tensor_hash(prep) and inp['desired_bit']==desired,'Saved input mismatch')
 else:
  inp={**rt.fresh_input(prep,desired),'spec':spec,'prepared_sha256':tensor_hash(prep),'desired_bit':desired};save_tensor(ip,inp)
 distance=float((prep.double()-base.double()).norm());distinct=distance>=SETTINGS['minimum_prepared_distance_fraction']*spec['norm']
 valid=bool(pm['pass'] and int(inp['route'])==CASES.index(spec['case']) and distinct)
 dump(cp,{'spec':spec,'centre_sha256':base_sha,'C37_frozen_sha256':sha(out/'C37_FROZEN.json'),
   'preparation':pm,'preparation_valid':valid,'prepared_distance_from_centre':distance,'distinctness_passed':distinct,
   'generation_retries':0,'replacement_used':False,'files':{n:sha(f/n) for n in ('prepared.pt','INPUT.pt','PERTURBATION.pt')}})
 print('C37 PREPARATION',spec['id'],'centre',spec['centre_id'],'distance',distance,'valid',valid,flush=True)
 return prep,inp,valid

def replay_parent(rt,head,out):
 out=Path(out);data=out/'parent_c36/c35';rows=[]
 for c in CASES:
  sid=c+'__c36_new00';spec=read(data/'results'/sid/'COMPLETE.json')['spec'];f=data/'new_states'/sid
  prep=vector(load(f/'prepared.pt'));inp=load(f/'INPUT.pt');v=load(data/'proposals'/sid/'POINTS.pt')
  with torch.no_grad():points,_=head(prep.to(rt.device),inp['input'].to(rt.device),inp['route'].to(rt.device),inp['extras'][:6].to(rt.device).double()*.002)
  require(all(torch.equal(points[a].cpu(),v['points'][a]) for a in ARMS),'C36 sealed proposal does not reproduce')
  p=out/'replay'/(sid+'.json');old=read(data/'results'/sid/'COMPLETE.json');binding={'freeze_sha256':sha(out/'C37_FROZEN.json'),'parent_points_sha256':sha(data/'proposals'/sid/'POINTS.pt')}
  if p.exists():
   r=read(p);require(r['binding']==binding,'Replay binding differs')
   for a in ARMS:compare_replay(r['arms'][a],old['arms'][a]['metrics'],c)
  else:
   start=STARTS[c];target=start[:3]+[start[0]^start[1]];arms={}
   for a in ARMS:
    m=rt.evaluate(v['points'][a],target,start);compare_replay(m,old['arms'][a]['metrics'],c);arms[a]=m
   r={'id':sid,'binding':binding,'arms':arms};dump(p,r)
  rows.append(r);print('C37 C36 PHYSICAL REPLAY',sid,'OK',flush=True)
 return rows

def collect(out):
 out=Path(out);data=out/'c35';rs=[]
 for s in specs():
  f=data/'new_states'/s['id'];r=read(f/'COMPLETE.json');require(r['spec']==s and r['C37_frozen_sha256']==sha(out/'C37_FROZEN.json'),'Completed preparation spec/freeze changed')
  for n,h in r['files'].items():require(sha(f/n)==h,'Completed preparation tensor changed')
  prep=vector(load(f/'prepared.pt'));inp=load(f/'INPUT.pt');v,_,_=proposals(None,None,data,s,prep,inp)
  rec=evaluate(None,None,data,s,prep,inp,r['preparation_valid'],'new_local')
  for a in ARMS:
   require(rec['arms'][a]['complete_accepted']==bool(r['preparation_valid'] and rec['arms'][a]['metrics']['pass']),'Acceptance flag changed')
   require(rec['arms'][a]['metrics']['theta_sha256']==tensor_hash(v['points'][a]),'Actual endpoint tensor mismatch')
  rs.append(rec)
 return rs

def summarize(out,rs):
 out=Path(out);require(len(rs)==32 and {r['spec']['id'] for r in rs}=={s['id'] for s in specs()},'Requested denominator changed')
 co=counts(rs);passed={a:co[a]['complete_passed']==32 for a in ('geometry6','selected6')}
 dec='C37_BOTH_FROZEN_WRITERS_CENTRE_TRANSFER_PASS' if all(passed.values()) else ('C37_ONE_BRANCH_CENTRE_TRANSFER_PASS' if any(passed.values()) else 'C37_CENTRE_TRANSFER_FAILURE')
 return {'protocol':PROTOCOL,'complete':True,'decision':dec,'new_local':co,
   'by_case':{c:counts([r for r in rs if r['spec']['case']==c]) for c in CASES},'branch_complete_acceptance':passed,
   'centres':centres(),'centres_previously_inspected':True,'centres_selected_by_index_not_outcome':True,
   'all32_requested_states_retained':True,'invalid_preparations':sum(not r['preparation_valid'] for r in rs),
   'new_training_steps':0,'state_generation_retries':0,'test_based_branch_or_checkpoint_selection':False,
   'online_write_Jacobians':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,'online_write_backtracking':0,
   'postwrite_response_used_for_proposal':False,'teacher_fallback_used':False,
   'offline_state_preparation_uses_geometry':True,'centre_preparation_Jacobian_files':len(list((out/'centre_geometry').glob('*.pt'))),
   'prepared_distances':{s['id']:read(out/'c35/new_states'/s['id']/'COMPLETE.json')['prepared_distance_from_centre'] for s in specs()},
   'C35_frozen_sha256':sha(out/'c35/COEFFICIENTS_FROZEN.json'),'C37_frozen_sha256':sha(out/'C37_FROZEN.json'),
   'deployed_writer_identity_sha256':sha(out/'DEPLOYED_WRITER.json'),'same_model_chart_access_prompts':True,
   'requested_output_bit_supplied':True,'fresh_evaluation':False,'fresh_state_seeds':True,
   'independent_model_or_chart_replication':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
   'LC1_same_model_centre_transfer_passed':any(passed.values()),'LC1_independently_confirmed':False,
   'full_Line_C_complete':False,'scope':SETTINGS['scope']}

def main(out,backup='',runtime_factory=Runtime,identity_checker=writer_identity):
 out=Path(out);check_sources(ROOT);require(read(out/'identity.json')==identity(ROOT),'C37 identity/source mismatch');data=validate(out,ROOT)
 require(read(out/'NEW_STATE_SPECS.json')==specs(),'Fixed state specifications changed')
 if not (out/'PARENT_AUDIT.json').exists():dump(out/'PARENT_AUDIT.json',audit_parent(out,identity_checker));snapshot(out,backup)
 freeze(out)
 if (out/'SUMMARY.json').exists():
  s=summarize(out,collect(out));require(s==read(out/'SUMMARY.json'),'Completed summary recount differs');print(json.dumps(s,indent=2),flush=True);return s
 rt=runtime_factory(runtime_dir(out),ROOT/'inherited/inherited/inherited/inherited');head=ComposedWriter(rt.writer,records(data),rt.device)
 tensor_identity=deployed_identity(head);require(tensor_identity==read(out/'parent_c36/DEPLOYED_WRITER.json')['tensors'],'Actual deployed C35 writer differs from C36')
 deployed={'C37_frozen_sha256':sha(out/'C37_FROZEN.json'),'tensors':tensor_identity}
 if (out/'DEPLOYED_WRITER.json').exists():require(read(out/'DEPLOYED_WRITER.json')==deployed,'Saved runtime identity differs')
 else:dump(out/'DEPLOYED_WRITER.json',deployed)
 replay_parent(rt,head,out);snapshot(out,backup);rs=[]
 for i,s in enumerate(specs(),1):
  freeze(out);prep,inp,valid=prepare_new(rt,out,s);rs.append(evaluate(rt,head,data,s,prep,inp,valid,'new_local'))
  print('C37 ENDPOINT',i,'/32',{a:rs[-1]['arms'][a]['complete_accepted'] for a in ARMS},flush=True)
  if i%4==0:snapshot(out,backup)
 rt.invariant();require(deployed_identity(head)==tensor_identity,'Actual frozen writer changed');freeze(out);validate(out,ROOT)
 s=summarize(out,rs);dump(out/'SUMMARY.json',s);snapshot(out,backup);print(json.dumps(s,indent=2),flush=True);return s

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--drive-dir',default='');a=p.parse_args()
 lock=open(Path(a.out).parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a.out,a.drive_dir)
