"""Verify compact archived evidence and exported heads without Torch/model downloads."""
import argparse, csv, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
from numpy_writer import load_head,propose

START={'xor_00':[0,0,1,1],'xor_01':[0,1,1,0],'xor_10':[1,0,1,0],'xor_11':[1,1,1,1]}
def read(p):return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gates(v,case):
    s=START[case];target=s[:];target[3]=s[0]^s[1]
    e=v['evaluation'];exact=e['exact_margins'];dev=e['heldout_margins']
    assert len(exact)==4 and len(dev)==12
    assert all(math.isfinite(t) for t in exact+dev+[v['drift'],v['kl']])
    g={'bits_match':v['bits']==target,'exact_read_margin':min(exact)>=.05,
       'development_read_margin':min(dev)>=.05,'response':v['drift']<=.002,
       'KL':v['kl']<=.005,'output_flip':v['bits'][3]!=s[3],
       'unselected_preserved':v['bits'][:3]==s[:3]}
    assert g==v['gates'],('Recorded gate mismatch',case,g,v['gates'])
    assert bool(v['episode_pass'])==all(g.values())
    return all(g.values())

def main():
    if not __debug__:raise RuntimeError('Run verification without Python -O; assertions are required')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--protocol-tests',action='store_true');a=p.parse_args()
    root=a.root.resolve();index=read(root/'evidence/line_c_index.json');result={}
    for version,item in index['versions'].items():
        base=root/item['directory'];m=read(base/'SHA256SUMS.json')
        actual={str(x.relative_to(base)) for x in base.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc'}
        assert actual==set(m)|{'SHA256SUMS.json'},('File coverage',version,actual-set(m)-{'SHA256SUMS.json'})
        for name,h in m.items():
            rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
            assert digest(base/rel)==h,(version,name)
        orig=read(base/'recorded/MANIFEST.json');export=read(base/'EXPORT_PROVENANCE.json')
        for target,member in export['original_members'].items():assert digest(base/target)==orig[member]
        s=read(base/'recorded/SUMMARY.json');assert s['decision']==item['decision'] and s['complete']
        rows=list((base/'recorded/evaluation').glob('*/*/COMPLETE.json'))
        if version=='C7':
            baseline_counts={co:{'n':0,'passed':0} for co in ('training','excluded_development')}
            for f in (base/'recorded/baseline').glob('*/*/endpoint.json'):
                fold,sid=f.parent.parent.name,f.parent.name
                co='excluded_development' if sid.split('__')[0]==fold else 'training'
                baseline_counts[co]['n']+=1
                baseline_counts[co]['passed']+=gates(read(f),sid.split('__')[0])
            assert baseline_counts==s['frozen_baseline']
            rows=list((base/'recorded/evaluation').glob('*/*/COMPLETE.json'))
            counts={co:{arm:0 for arm in ('parameter_mse','functional')} for co in ('training','excluded_development')}
            den={k:0 for k in counts}
            for f in rows:
                r=read(f);den[r['cohort']]+=1
                for arm in counts[r['cohort']]:counts[r['cohort']][arm]+=gates(r[arm],r['state'].split('__')[0])
            assert den=={'training':27,'excluded_development':9}
            for co in counts:
                for arm,n in counts[co].items():assert n==s['paired_cohorts'][co][arm+'_passed']
            result[version]={'frozen_baseline':baseline_counts,'paired_denominators':den,'paired_counts':counts}
        elif version=='C8':
            candidates=list((base/'recorded/candidates').glob('*.json'));assert len(candidates)==120
            accum={}
            for f in candidates:
                r=read(f);key=(r['family'],r['arm'],r['cohort']);x=accum.setdefault(key,{'n':0,'raw_passed':0,'corrected_passed':0});x['n']+=1
                for arm in ('raw','corrected'):x[arm+'_passed']+=gates(r[arm],r['state'].split('__')[0])
            for (family,arm,co),x in accum.items():
                for k,v in x.items():assert s['aggregate'][family][arm][co][k]==v
            result[version]={'candidates':len(candidates),'endpoint_gate_recounts':240}
        else:
            assert len(rows)==20
            arms=('raw','corrected') if version=='C9' else ('frozen_c9','distilled')
            counts={co:{arm:0 for arm in arms} for co in ('training','new')};den={k:0 for k in counts}
            for f in rows:
                r=read(f);co=f.parent.parent.name;den[co]+=1;assert r['state_usable']
                for arm in arms:counts[co][arm]+=gates(r[arm],r['case'])
            assert den=={'training':12,'new':8}
            for co in counts:
                skey=('training' if co=='training' else 'new_state_development')+('' if version=='C9' else '_raw')
                for arm,n in counts[co].items():assert n==s[skey][arm+('_passed' if version=='C9' else '')]
                if version=='C10':
                    items=[read(f)['distilled'] for f in rows if f.parent.parent.name==co]
                    gc={k:sum(v['gates'][k] for v in items) for k in items[0]['gates']}
                    assert gc==s[skey]['distilled_gate_counts']
            with (base/'recorded/endpoint_gate_table.csv').open(newline='') as f:
                table=list(csv.DictReader(f))
            assert len(table)==40
            assert len({(r['cohort'],r['state'],r['arm']) for r in table})==40
            for r in table:
                v=read(base/'recorded/evaluation'/r['cohort']/r['state']/'COMPLETE.json')[r['arm']]
                assert float(r['mean_KL'])==v['kl'] and float(r['response_drift'])==v['drift']
                assert float(r['exact_margin_min'])==min(v['evaluation']['exact_margins'])
                assert float(r['development_margin_min'])==min(v['evaluation']['heldout_margins'])
                assert r['pass']==str(v['episode_pass'])
                for k,b in v['gates'].items():assert r[k]==str(b)
            for f in (base/'weights').glob('*.npz'):load_head(f)
            data=base/'replay/saved_raw_states.npz'
            with np.load(data,allow_pickle=False) as z:
                head,meta=load_head(base/'weights/final_head.npz');candidate=z['prepared']+propose(head,z['features'])
                diff=float(np.max(np.abs(candidate-z['recorded_raw'])))
                assert diff<=1e-4,('NumPy/CUDA proposal replay disagreement',version,diff)
            result[version]={'denominators':den,'counts':counts,'numpy_vs_recorded_raw_max_abs':diff,'head_sha256':meta['head_sha256']}
        if a.protocol_tests:
            subprocess.run([sys.executable,str(base/'source'/('test_'+version.lower()+'.py'))],check=True,cwd=base/'source')
    assert index['versions']['C10']['decision']=='C10_RAW_TRAINING_FUNCTION_FAILURE'
    print(json.dumps({'status':'PASS','recorded_consistency':result,'scope':'Hashes, recorded gates and saved proposal replay only. No independent GPT-2 inference or Line C confirmation.'},indent=2))

if __name__=='__main__':main()
