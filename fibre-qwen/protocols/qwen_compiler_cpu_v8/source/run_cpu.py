#!/usr/bin/env python3
"""CPU-only, independent calls, immutable query schedule and resumable point commits."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
import argparse,datetime,json,sys,time,traceback,zipfile,shutil,platform,importlib.metadata
from pathlib import Path
from protocol import CONFIG,ID,MODEL,REVISION,TOTAL,sha,canonical,make_queries,schedule,summarize,parser_audit
from candidate_backend import score_candidates
from interface_compiler import process_input
from constrained_backend import constrained_evaluate
from numerical_diagnostic import evaluate_numeric
ROOT=Path(__file__).resolve().parent
class Pause(Exception):pass

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));os.replace(tmp,path)

def verify(root):
    m=json.loads((root/'MANIFEST.json').read_text())
    for n,h in m.items():
        if sha((root/n).read_bytes())!=h:raise ValueError('SOURCE_HASH_MISMATCH '+n)
    return sha(canonical(m))

def archive(out):
    payload={str(p.relative_to(out)):p.read_bytes() for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('RESULT_MANIFEST.json','ACTIVE_SESSION.pid') and p.suffix!='.tmp'}
    hashes={n:sha(b) for n,b in payload.items()};write(out/'RESULT_MANIFEST.json',hashes)
    dest=out.parent/(out.name+'_results.zip');tmp=dest.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(out.name+'/'+n,b)
        z.writestr(out.name+'/RESULT_MANIFEST.json',json.dumps(hashes,indent=2))
    os.replace(tmp,dest);print('RESULT_ARCHIVE='+str(dest),flush=True);return dest

def ensure_same(path,data):
    if path.exists() and json.loads(path.read_text())!=data:raise ValueError('FROZEN_STATE_MISMATCH '+path.name)
    write(path,data)

def saved_records(out,plan):
    paths=sorted((out/'records').glob('*.json'));records=[]
    for i,p in enumerate(paths):
        if p.name!=f'{i:03d}.json':raise ValueError('CHECKPOINT_SEQUENCE_GAP')
        r=json.loads(p.read_text())
        if i>=len(plan) or any(r.get(k)!=plan[i][k] for k in ('run_id','repeat','query')):raise ValueError('CHECKPOINT_QUERY_MISMATCH')
        records.append(r)
    return records

def execute(out,plan,baseline,evaluator,environment_match,deadline=float('inf')):
    records=saved_records(out,plan)
    summarize(records,plan,baseline,environment_match)
    for item in plan[len(records):]:
        if time.monotonic()>=deadline:raise Pause('Session budget reached; committed points can resume')
        # evaluator receives source text only; never receives expected fields, truth labels or case metadata.
        observation=evaluator(item['query']['source_text'])
        r=dict(**item,**observation);candidate=records+[r]
        report=summarize(candidate,plan,baseline,environment_match)
        write(out/'records'/f'{len(records):03d}.json',r);records=candidate;write(out/'summary.json',report)
        print('QWEN_COMPILER_V8_POINT '+json.dumps(dict(n=len(records),id=item['run_id'],accepted=observation['accepted'],parsed=observation['parsed'],bounded=observation.get('constrained',{}).get('raw'),compiler_matches=report['positive_compiler_matches']),ensure_ascii=False),flush=True)
        ap=archive(out)
        if len(records)%16==0:
            stage=ap.parent/(out.name+f'_{len(records):03d}_results.zip');shutil.copy2(ap,stage);print('STAGE_ARCHIVE='+str(stage),flush=True)
    return summarize(records,plan,baseline,environment_match)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',default='compiled_interface',choices=['compiled_interface']);p.add_argument('--threads',type=int,choices=[4],default=4);p.add_argument('--minutes',type=float,default=90);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if args.minutes<=0:p.error('minutes must be positive')
    source_hash=verify(ROOT);baseline=json.loads((ROOT/'baseline.json').read_text())
    if args.check_only:print('SOURCE_OK',source_hash,str(TOTAL)+' checkpoints; 128 constrained generations; 24 negative inputs; no model loaded');return 0
    out=args.out.resolve()
    if out==ROOT or ROOT in out.parents:p.error('output must be outside source')
    out.mkdir(parents=True,exist_ok=True)
    lock=dict(protocol_id=ID,source_manifest_sha256=source_hash,config=CONFIG)
    ensure_same(out/'RUN_LOCK.json',lock)
    if not (out/'frozen_source').exists():shutil.copytree(ROOT,out/'frozen_source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if not (out/'freeze.json').exists():write(out/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest_sha256=source_hash,external_preregistration=False))
    mutex=out/'ACTIVE_SESSION.pid'
    if mutex.exists():
        try:os.kill(int(mutex.read_text()),0)
        except ProcessLookupError:mutex.unlink()
        else:raise RuntimeError('CONTROLLER_ALREADY_ACTIVE')
    mutex.write_text(str(os.getpid()));start=time.monotonic();plan=None;report=None;match=False;error=None
    try:
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(84917)
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','tokenizers','huggingface_hub','safetensors')}
        expected={'torch':'2.8.0','transformers':'4.56.2','numpy':'2.3.5','tokenizers':'0.22.2','huggingface_hub':'0.36.2','safetensors':'0.6.2'}
        if any(packages[k].split('+')[0]!=v for k,v in expected.items()) or torch.version.cuda is not None:raise ValueError('PINNED_CPU_DEPENDENCIES_REQUIRED')
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown') if Path('/proc/cpuinfo').exists() else platform.processor()
        env=dict(packages=packages,python=platform.python_version(),cpu=cpu,threads=4,device='cpu',dtype='float32');ensure_same(out/'environment.json',env)
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
        queries=make_queries(tok,baseline)
        if queries!=json.loads((ROOT/'FROZEN_QUERIES.json').read_text()):raise ValueError('PREDECLARED_QUERY_MISMATCH')
        plan=schedule(queries)
        negative=parser_audit();write(out/'PARSER_AUDIT.json',negative)
        if not negative['pass_gate']:raise ValueError('NEGATIVE_PARSER_PREFLIGHT_FAILED')
        if tok.eos_token_id!=baseline['eos_token_id'] or [tok.encode(v,add_special_tokens=False) for v in ('0','1')]!=baseline['candidate_token_ids']:raise ValueError('TOKEN_MAPPING_CHANGED')
        if sha(tok.get_chat_template().encode())!=baseline['chat_template_sha256']:raise ValueError('CHAT_TEMPLATE_CHANGED')
        ensure_same(out/'protocol.json',dict(config=CONFIG,queries=queries,schedule=plan,chat_template_sha256=baseline['chat_template_sha256'],baseline_sha256=sha((ROOT/'baseline.json').read_bytes())))
        print('Loading pinned Qwen3-1.7B: CPU float32; no Adapter or training.',flush=True)
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float32,trust_remote_code=False,attn_implementation='sdpa').eval();model.requires_grad_(False)
        if model.config._commit_hash!=REVISION or model.config.model_type!='qwen3' or model.config._attn_implementation!='sdpa':raise ValueError('MODEL_CONFIGURATION_MISMATCH')
        tensors=list(model.parameters())+list(model.buffers());fixed=[(t,t._version) for t in tensors];topology=[(n,id(m)) for n,m in model.named_modules()]
        if any(t.device.type!='cpu' for t in tensors) or any(t.is_floating_point() and t.dtype!=torch.float32 for t in model.parameters()):raise ValueError('CPU_FP32_REQUIRED')
        def check():
            if any(t._version!=v or t.requires_grad for t,v in fixed) or topology!=[(n,id(m)) for n,m in model.named_modules()]:raise ValueError('MODEL_CHANGED')
        def infer(prompt):
            scores=score_candidates(model,tok,prompt)
            constrained=constrained_evaluate(model,tok,prompt)
            numeric=evaluate_numeric(model,tok,prompt);check()
            return dict(scores=scores,constrained=constrained,numeric=numeric)
        def evaluate(source_text):return process_input(source_text,tok,infer)
        match=env==baseline['environment']
        report=execute(out,plan,baseline,evaluate,match,start+args.minutes*60);check();verify(ROOT)
        report.update(frozen_model_verified=True,observed_parameter_count=sum(t.numel() for t in model.parameters()))
    except (Pause,KeyboardInterrupt):
        error=dict(kind='PAUSED',message='Resume with the same source and latest result archive.');print('PAUSED_RESUMABLE',flush=True)
    except Exception:
        error=dict(kind='ERROR',message=traceback.format_exc());print(error['message'],flush=True)
    finally:
        if plan is not None and report is None:report=summarize(saved_records(out,plan),plan,baseline,match)
        if report is None:report=dict(complete=False,completed=0,expected=TOTAL,decision='INCOMPLETE',overall_pass=False,formal_l6_pass=False)
        if error:report.update(interruption=error,overall_pass=False,confirmation_pass=False,unseen_expression_confirmation=False,semantic_pass=False,native_free_pass=None,compiled_interface_pass=False,compiler_pass=False,constrained_interface_pass=False,integrity_pass=False,constraint_audit_pass=False,deployment_ready=False,decision=error['kind'])
        report['session_seconds']=time.monotonic()-start;write(out/'summary.json',report)
        if mutex.exists():mutex.unlink()
        archive(out);print('SUMMARY '+json.dumps({k:report[k] for k in ('complete','completed','decision','overall_pass','formal_l6_pass')}),flush=True)
    return 0 if error is None or error['kind']=='PAUSED' else 1
if __name__=='__main__':raise SystemExit(main())
