"""Frozen-evaluator replay plus independent raw-field audit; no model inference."""
from pathlib import Path
import hashlib,json,math,re,sys,zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PACKAGE=ROOT/'fibre-qwen/protocols/qwen_chain_cpu_v9'
SOURCE=PACKAGE/'source'
sys.path.insert(0,str(SOURCE))
import protocol as p
from evaluation import summarize

def require(ok,label):
    if not ok:raise ValueError(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def oracle(op,a,b):
    if op=='OR':return int(bool(a or b))
    if op=='AND':return int(bool(a and b))
    if op=='XOR':return int(a!=b)
    if op=='EQUAL':return int(a==b)
    raise ValueError('Unknown operation')
def load_zip(path):
    prefix='cpu_two_step_chain/'
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and all(n.startswith(prefix) for n in names),'Archive layout')
        require(z.testzip() is None,'ZIP CRC')
        data={n[len(prefix):]:z.read(n) for n in names}
    m=json.loads(data['RESULT_MANIFEST.json'])
    require(set(data)==set(m)|{'RESULT_MANIFEST.json'},'Payload inventory')
    for name,digest in m.items():require(sha(data[name])==digest,'Payload checksum: '+name)
    records=sorted(n for n in data if n.startswith('records/'))
    require(records==[f'records/{i:03d}.json' for i in range(len(records))],'Record sequence')
    return data,records,len(m)

def audit():
    manifest=json.loads((SOURCE/'MANIFEST.json').read_text())
    for n,h in manifest.items():require(sha((SOURCE/n).read_bytes())==h,'Delivered source: '+n)
    manifest_hash=sha(p.canonical(manifest))
    require(manifest_hash=='9fdfc82381f988451d3b84695a4f3703191deb44f95d9faadd7834ac5c37a038','Frozen source ID')
    require(sha((PACKAGE/'qwen_chain_cpu_v9.zip').read_bytes())=='d983ecf996f9ef537c47418354758924e891c205f39f742644af9d496e0af8fc','Delivered ZIP')
    with zipfile.ZipFile(PACKAGE/'qwen_chain_cpu_v9.zip') as z:
        for n in [*manifest,'MANIFEST.json']:
            require(z.read('qwen_chain_cpu_v9/'+n)==(SOURCE/n).read_bytes(),'Source extraction')
    plan=p.verify_inventory();baseline=json.loads((SOURCE/'baseline.json').read_text())
    main,names,main_hashes=load_zip(HERE/'cpu_two_step_chain_results.zip')
    require(len(names)==112,'Main incomplete')
    rows=[json.loads(main[n]) for n in names]
    env=json.loads(main['environment.json']);require(env==baseline['environment'],'Baseline environment')
    stages={}
    for path in sorted(HERE.glob('*results.zip')):
        data,record_names,hash_count=load_zip(path)
        lock=json.loads(data['RUN_LOCK.json'])
        require(lock==dict(protocol_id=p.ID,source_manifest_sha256=manifest_hash,config=p.CONFIG),'Run lock')
        require(json.loads(data['protocol.json'])==dict(config=p.CONFIG,schedule=plan),'Protocol schedule')
        require(data['environment.json']==main['environment.json'],'Stage environment')
        for n in [*manifest,'MANIFEST.json']:
            require(data['frozen_source/'+n]==(SOURCE/n).read_bytes(),'Result frozen source')
        for n in record_names:require(data[n]==main[n],'Stage prefix mismatch')
        observed=json.loads(data['summary.json'])
        recomputed=summarize([json.loads(data[n]) for n in record_names],plan,baseline,True)
        extras=set(observed)-set(recomputed)
        if path.name=='cpu_two_step_chain_results.zip':
            require(extras=={'frozen_model_verified','observed_parameter_count','session_seconds'},'Main summary additions')
            require(observed['frozen_model_verified'] is True and observed['observed_parameter_count']==1720574976,'Recorded freeze check')
            require(math.isfinite(observed['session_seconds']) and observed['session_seconds']>0,'Runtime')
        else:require(not extras,'Unexpected stage summary additions')
        require({k:v for k,v in observed.items() if k not in extras}==recomputed,'Frozen summary replay')
        stages[path.name]=dict(records=len(record_names),sha256=sha(path.read_bytes()),payload_hashes=hash_count,summary_reproduced=True)
    # Independent raw-field/dataflow checks, separate from the original evaluator.
    direct={};margin=[];aligned=[];legacy=[];mask_exact=0;wires=0;direct_exact=0
    for i,r in enumerate(rows):
        f=r['fields'];o=r['observation'];item=r['item'];c=o['constrained'];trace=c['trace'];s=o['scores']
        require(item==plan[i],'Independent item ordering')
        require(o['accepted'] is True and o['model_invocations']==1 and o['parse_error'] is None,'Invocation count')
        require(json.loads(r['source_text'])==f==o['parsed'],'Parsed input')
        if item['kind']=='direct':require(f==item['fields'] and r['link'] is None,'Direct input')
        else:
            q=item['program']
            if item['kind']=='step1':require(f==dict(op=q['op1'],a=q['a'],b=q['b']) and r['link'] is None,'First input')
            else:
                first=rows[item['first_index']];bit=int(first['observation']['constrained']['raw'])
                flip=item['kind']=='flip_step2'
                require(f==dict(op=q['op2'],a=1-bit if flip else bit,b=q['c']),'Actual bit transfer')
                require(r['link']==dict(first_index=item['first_index'],first_id=first['item']['id'],observed_bit=bit,intervention='flip' if flip else 'none'),'Link provenance')
                wires+=1
        expected=oracle(f['op'],f['a'],f['b'])
        require(c['raw']==str(expected) and c['output_ids']==[15+expected,151645] and c['truncated'] is False,'Bounded truth')
        gap=s['bit_logp'][1]-s['bit_logp'][0]
        signed=(1 if expected else -1)*gap
        require(math.isfinite(signed) and signed>1e-5,'Primary truth/margin');margin.append(signed)
        require(trace['steps']==[0,1] and trace['finite_bit_count']==2,'Mask structure')
        require(trace['first_bit_logits']==trace['masked_bit_logits'],'Mask logits');mask_exact+=1
        for key in ['last_no_cache','last_cache']:
            diffs=[abs(a-b) for a,b in zip(trace['first_bit_logits'],o['numeric'][key]['bit_logits'])]
            require(len(diffs)==2 and all(math.isfinite(x) and x<=1e-5 for x in diffs),'Aligned logits');aligned.extend(diffs)
        legacy.append(abs((trace['first_bit_logits'][1]-trace['first_bit_logits'][0])-gap))
        key=(f['op'],f['a'],f['b'])
        if item['kind']=='direct':direct[key]=o
        else:
            require(all(o[k]==direct[key][k] for k in ['scores','constrained','numeric']),'Exact fresh direct replay');direct_exact+=1
    changed=0;unchanged=0;endpoints=0;upstream_errors=0
    for i in range(16,112,3):
        a,b,c=rows[i:i+3];q=a['item']['program']
        mid=oracle(q['op1'],q['a'],q['b']);end=oracle(q['op2'],mid,q['c'])
        t=int(a['observation']['constrained']['raw']);y=int(b['observation']['constrained']['raw']);yf=int(c['observation']['constrained']['raw'])
        upstream_errors+=int(t!=mid);require(y==end,'End-to-end truth');endpoints+=1
        sensitive=oracle(q['op2'],t,q['c'])!=oracle(q['op2'],1-t,q['c'])
        require((y!=yf)==sensitive,'Intervention effect')
        changed+=int(y!=yf);unchanged+=int(y==yf)
    logged=[json.loads(s) for s in re.findall(r'QWEN_CHAIN_V9_POINT (\{[^\n]*\})',(HERE/'QWEN_CHAIN_V9_SESSION_OUTPUT.txt').read_text())]
    require(len(logged)==112,'Log count')
    for i,(r,l) in enumerate(zip(rows,logged),1):
        require(l==dict(n=i,id=r['item']['id'],kind=r['item']['kind'],fields=r['fields'],link=r['link'],raw=r['observation']['constrained']['raw'],primary_correct=i,bounded_correct=i),'Log record mismatch')
    summary=json.loads(main['summary.json']);require(summary['overall_pass'] is True,'Formal verdict')
    return dict(status='AUDITED_EXTERNAL_TWO_STEP_INTERFACE_PASS',completed=112,expected=112,
        frozen_source_manifest_sha256=manifest_hash,original_summary_reproduced=True,primary_correct=112,bounded_correct=112,
        minimum_primary_margin_nats=min(margin),actual_dependency_links_verified=wires,end_to_end_correct=endpoints,
        intervention_changed=changed,intervention_unchanged=unchanged,upstream_errors=upstream_errors,
        natural_error_propagation_observed=False,same_forward_masks_exact=mask_exact,
        maximum_aligned_logit_delta=max(aligned),legacy_diagnostic_failures=sum(x>1e-5 for x in legacy),
        maximum_legacy_gap_delta=max(legacy),fresh_direct_replays_exact=direct_exact,
        runtime_seconds=summary['session_seconds'],archives=stages,source_files_verified=len(manifest),
        v8_final_pass_confirmed=False,pretrained_model_rerun_for_audit=False,
        scope='32 authored programs,16 canonical model prompts,one recorded CPU environment; external controller, no native planning or parameter-memory evidence')

if __name__=='__main__':
    result=audit()
    if sys.argv[1:]==['--write']:(HERE/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        require(not sys.argv[1:],'Usage: audit_v9.py [--write]')
        require(result==json.loads((HERE/'AUDIT.json').read_text()),'AUDIT.json changed')
    print(json.dumps({k:v for k,v in result.items() if k!='archives'},indent=2))
