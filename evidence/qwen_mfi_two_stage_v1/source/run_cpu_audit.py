"""CPU, float32, no total runtime cutoff. Two-stage model read then compute; reader always on."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,platform,signal,time,traceback
from pathlib import Path
import torch,transformers,numpy as np
from safetensors.torch import load_file,save_file
from readonly_backend import ReadOnlyChart
from reader_backend import attach_reader
from snapshot_io import verify_files,validate_snapshot,load_snapshot
from generation_audit import locked_generate
from reader_protocol import score as read_score,hard_gate
from two_stage_protocol import cases,read_cases,score,summarize,ARMS,SNAPS,bind
from io_utils import atomic_json,PauseRun
from qwen_l1 import MODEL,REVISION,ANCHORS

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
 if args.threads<1:raise ValueError('threads must be positive')
 root=Path(__file__).parent;out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 start=time.monotonic();rows=[];replay=[];audits={};restore=None;restored=False;integrity=False;frozen=False;stable=False;phase='loading';status='IN_PROGRESS';failed=False
 def report():
  result=summarize(rows,replay);result.update(status=status,complete=status=='COMPLETE' and len(rows)==30 and stable and frozen and restored and integrity,
   source_integrity=integrity,cpu_before_after_stable=stable,frozen_verified=frozen,source_restored=restored,
   replay_rows=len(replay),expected_replay_rows=18,phase=phase,elapsed_seconds=time.monotonic()-start,
   time_limit_seconds=None,device='cpu',dtype='float32',original_gpu_certification_reproduced=False,
   reference_scope='Same original zero-memory / no-reader reference STATE recomputed on CPU before reader attachment; not original GPU numerical reference. Original budgets .02/.01 unchanged, scored separately from ablation completion.')
  constraints=hard_gate(audits.get('after',{}).get('on',{}))
  result['known_two_stage_gate']=bool(result['complete'] and constraints['pass_all'] and result['known_raw_four_pass'])
  result['known_text_control_gate']=bool(result['known_two_stage_gate'] and result['known_single_flip_eight_pass'] and result['known_double_flip_four_pass'])
  atomic_json(out/'summary.json',result)
 def check():pass
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(PauseRun('SIGTERM')))
 try:
  source=verify_files(root);integrity=True
  atomic_json(out/'protocol.json',dict(id='QWEN_SELF_READ_COMPUTE_CPU_V1',model=MODEL,revision=REVISION,source=source,
   cases=cases(),read_cases=read_cases(),reader_arms=list(ARMS),memory_snapshots=list(SNAPS),
   precision='float32, no quantization',time_limit_seconds=None,max_new_tokens=128,
   training=False,output_repair=False,rule_adapter='always active',
   primary_condition='raw model-produced intermediate',control_conditions=['flip0','flip1','flip_both','no_intermediate'],
   prompt_provenance='Stage1 uses fixed canonical read prompt. Stage2 receives the original decoded stage1 text plus newline plus frozen protected task body. Only control arms alter intermediate text.',
   generation_input='Only prompt text enters tokenizer/model; state/expected metadata used solely for scoring after generation.',
   reference='CPU recomputation from original zero-memory no-reader state; never recentered at trained reader.',
   code_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}))
  report();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
  print('Loading pinned Qwen on CPU; first download may take time.',flush=True)
  b=ReadOnlyChart(root,check,json.loads((root/'protection_cases.json').read_text()))
  if b.device!='cpu':raise RuntimeError('CPU required')
  atomic_json(out/'runtime.json',dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
   python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),threads=torch.get_num_threads(),device=b.device))
  gpu_ref=json.loads((root/'source_zero_reference.json').read_text())
  atomic_json(out/'cpu_reference.json',dict(response=b.ref_response.tolist(),anchor_texts=ANCHORS,
   original_gpu_response=gpu_ref['response'],max_response_difference_from_gpu=max(abs(x-y) for x,y in zip(b.ref_response.tolist(),gpu_ref['response'])),
   ref_logp_sha256=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in b.ref_logp],
   note='Full original GPU log probabilities unavailable; CPU KL cannot certify original GPU KL budget.'))
  save_file({f'anchor_{i}':x.cpu().contiguous() for i,x in enumerate(b.ref_logp)},str(out/'cpu_ref_logp.safetensors'))
  data={s:load_file(str(root/f'memory_{s}.safetensors')) for s in SNAPS}
  for s,d in data.items():validate_snapshot(b.modules,d,zero=s=='zero')
  readers,params=attach_reader(b);endpoint=load_file(str(root/'reader_endpoint.safetensors'))
  if set(endpoint)!={n+'.'+k for n in readers for k in ('A','B')}:raise ValueError('Reader keys mismatch')
  # GPU and CPU RNG differ. Load BOTH serialized A and B, never regenerate the reader A.
  with torch.no_grad():
   for n,m in readers.items():
    for k in ('A','B'):
     v=endpoint[n+'.'+k];dest=getattr(m,k)
     if v.shape!=dest.shape or v.dtype!=dest.dtype or not torch.isfinite(v).all():raise ValueError('Bad reader tensor')
     dest.copy_(v)
  b.model.requires_grad_(False);b.model.eval()
  scales={n:m.scale for n,m in readers.items()}
  memory_ids={id(m.B) for m in b.modules.values()}
  fixed=[(t,t._version) for t in list(b.model.parameters())+list(b.model.buffers()) if id(t) not in memory_ids]
  def assert_fixed():
   b.assert_frozen()
   if any(t._version!=v for t,v in fixed) or any(p.requires_grad for p in b.model.parameters()):raise RuntimeError('Frozen weights changed')
   for n,m in readers.items():
    for k in ('A','B'):
     if not torch.equal(getattr(m,k).detach(),endpoint[n+'.'+k]):raise RuntimeError('Serialized reader changed')
  def set_arm(arm):
   if arm not in ARMS:raise ValueError('Bad arm')
   for n,m in readers.items():m.scale=scales[n] if arm=='on' else 0.
   assert_fixed()
  def load(s):load_snapshot(b,data[s],zero=s=='zero')
  def assert_state(s,arm):
   assert_fixed()
   for n,m in b.modules.items():
    for k in ('A','B'):
     if not torch.equal(getattr(m,k).detach(),data[s][n+'.'+k]):raise RuntimeError('Memory changed')
   if any(m.scale!=(scales[n] if arm=='on' else 0.) for n,m in readers.items()):raise RuntimeError('Arm changed')
  def restore_source():
   set_arm('on');load('zero');assert_state('zero','on')
  restore=restore_source
  def generate(c):
   versions=[(t,t._version) for t in list(b.model.parameters())+list(b.model.buffers())];t0=time.monotonic()
   with torch.inference_mode():
    enc=b.tok(c['prompt'],return_tensors='pt',add_special_tokens=False).to('cpu')
    seq,cfg=locked_generate(b.model,b.tok,enc['input_ids'],enc['attention_mask'])
    new=seq[0,enc['input_ids'].shape[1]:];raw=b.tok.decode(new,skip_special_tokens=True)
   if any(t._version!=v for t,v in versions):raise RuntimeError('Generation modified weights')
   return dict(raw=raw,output_ids=new.tolist(),truncated=len(new)>=128 and int(new[-1])!=b.tok.eos_token_id,
               seconds=time.monotonic()-t0,effective_config=cfg)
  def audit_and_read(which):
   aa={a:{} for a in ARMS}
   for s in SNAPS:
    load(s)
    for arm in ARMS:
     set_arm(arm);print('AUDIT',which,s,arm,flush=True)
     with torch.no_grad():
      a=b.audit(0) if s=='zero' else b.state_audit(s)
      if s=='zero':a={k:a[k] for k in ('response','kl')}
      a.update(b.rule_audit())
     rr=[]
     for c in (c for c in read_cases() if c['snapshot']==s):
      g=generate(c);r=dict(**c,arm=arm,phase=which,**g,**read_score(c,g['raw'],g['truncated']))
      replay.append(r);rr.append(r);atomic_json(out/'replay_rows.json',replay);assert_state(s,arm)
      print('READ',which,s,arm,c['id'],repr(g['raw']),flush=True)
     if s!='zero':a['copies']=[{k:r[k] for k in ('id','raw','truncated','output_ids','correct')} for r in rr if r['kind']=='copy']
     aa[arm][s]=a
   return aa
  phase='cpu_baseline';audits['before']=audit_and_read('before');atomic_json(out/'audits.json',audits)
  old=json.loads((root/'endpoint_rows.json').read_text());gpu_compare=[]
  for r in replay:
   if r['arm']=='on':
    q=next(x for x in old if x['phase']=='after' and x['id']==r['id'])
    gpu_compare.append(dict(id=r['id'],exact_output_match=r['output_ids']==q['output_ids'],cpu_raw=r['raw'],gpu_raw=q['raw']))
  atomic_json(out/'gpu_read_comparison.json',gpu_compare)
  stage1={snap:next(r for r in replay if(r['phase'],r['arm'],r['id'])==('before','on',snap+'-joint-0')) for snap in SNAPS}
  atomic_json(out/'stage1_rows.json',stage1)
  phase='two_stage_compute';report();current=None
  for base_case in cases():
   c=bind(base_case,stage1[base_case['snapshot']])
   if c['snapshot']!=current:load(c['snapshot']);current=c['snapshot']
   set_arm('on');skipped=c['prompt'] is None
   if skipped:g=dict(raw='',output_ids=[],truncated=True,seconds=0.,effective_config=None)
   else:g=generate(c)
   r=dict(**c,**g,skipped=skipped,**score(c,g['raw'],g['truncated']));rows.append(r)
   assert_state(c['snapshot'],'on');atomic_json(out/'logic_rows.json',rows);report()
   print(f"COMPUTE {len(rows)}/30 {c['id']} end_to_end={r['end_to_end_success']} follows_text={r['follows_intermediate']} skipped={skipped} raw={g['raw']!r}",flush=True)
  previous=json.loads((root/'parent_implicit_rows.json').read_text());comparison=[]
  for r in rows:
   if r['condition']=='no_intermediate':
    q=next(x for x in previous if(x['snapshot'],x['task'])==(r['snapshot'],r['task']))
    comparison.append(dict(id=r['id'],prompt_match=r['prompt']==q['prompt'],output_match=r['output_ids']==q['output_ids'],previous_raw=q['raw'],current_raw=r['raw']))
  atomic_json(out/'parent_direct_replay.json',comparison)
  phase='final_cpu_replay';audits['after']=audit_and_read('after');atomic_json(out/'audits.json',audits)
  stable=audits['before']==audits['after']
  for r in replay:
   if r['phase']=='after':
    q=next(x for x in replay if x['phase']=='before' and (x['id'],x['arm'])==(r['id'],r['arm']))
    stable=stable and all(r[k]==q[k] for k in ('raw','truncated','output_ids'))
  atomic_json(out/'cpu_constraints.json',{which:{arm:hard_gate(audits[which][arm]) for arm in ARMS} for which in audits})
  if not stable:raise RuntimeError('CPU pre/post repeatability failed')
  frozen=True;status='COMPLETE';phase='finished'
 except (PauseRun,KeyboardInterrupt) as e:
  failed=True;status='INTERRUPTED';atomic_json(out/'interruption.json',dict(reason=str(e),phase=phase,rows=len(rows)))
 except Exception:
  failed=True;status='ERROR';(out/'exception.txt').write_text(traceback.format_exc())
 finally:
  if restore is not None:
   try:restore();restored=True
   except Exception:failed=True;status='ERROR';(out/'restoration_error.txt').write_text(traceback.format_exc())
  report();print((out/'summary.json').read_text(),flush=True)
 return int(failed)
if __name__=='__main__':raise SystemExit(main())
