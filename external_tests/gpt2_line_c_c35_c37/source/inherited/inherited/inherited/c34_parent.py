"""Compact import of actual saved C31 endpoints and original pinned writer."""
from c34_protocol import *

def probe(z,root,full=False):
    i=zread(z,'identity.json');s=zread(z,'SUMMARY.json')
    require(i['protocol']==s['protocol']==PARENT_PROTOCOL and s['complete'] and s['decision']=='C31_RESIDUAL_COVERAGE_COMPARISON_COMPLETE','Completed C31 RESULTS ZIP required')
    pins=read(Path(root)/'C31_SOURCE_PINS.json')
    require(i['source_hashes']==pins and i['settings']==read(Path(root)/'C31_SETTINGS.json'),'Exact C31 source/settings required')
    require(s['C27_writer_unchanged'] and s['all48_training_and16_prospective_states_retained'],'Unexpected C31 scope')
    if full:
        for n,h in pins.items():require(hashlib.sha256(z.read('source/'+n)).hexdigest()==h,'C31 source changed: '+n)
        require(hashlib.sha256(z.read('PREDICTORS_FROZEN.json')).hexdigest()==s['frozen_predictor_sha256'],'C31 predictor freeze digest differs')
        require(zread(z,'PREDICTORS_FROZEN.json')['writer_frozen_sha256']==hashlib.sha256(z.read('WRITER_FROZEN.json')).hexdigest(),'C31 writer freeze binding differs')
    return i,s

def prepare(parent,out,root):
    out=Path(out);copied={}
    with zipfile.ZipFile(parent) as z:
        manifest=verify(z);_,s=probe(z,root,True)
        fixed=zread(z,'FIXED_STATE_SPECS.json')
        from c31_protocol import specs
        require(fixed=={'new_training':specs('new_training'),'prospective':specs('prospective'),'old_inspected16_used':False,'resampling_allowed':False},'C31 exact state specifications differ')
        names=['WRITER_FROZEN.json','FIXED_STATE_SPECS.json']
        names+=['parent/'+n for n in ('MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt')]
        for cohort in ('new_training','prospective'):
            for t in fixed[cohort]:
                rec=zread(z,'states/'+cohort+'/'+t['id']+'/COMPLETE.json')
                require(rec['spec']==t and rec['writer_frozen_sha256']==hashlib.sha256(z.read('WRITER_FROZEN.json')).hexdigest(),'State spec/writer identity differs')
                for n,h in rec['files'].items():require(manifest['states/'+cohort+'/'+t['id']+'/'+n]==h,'C31 tensor binding differs')
                names+=['states/'+cohort+'/'+t['id']+'/'+n for n in ('COMPLETE.json','INPUT.pt','prepared.pt','RAW_WRITTEN.pt')]
        for old,new in [('SUMMARY.json','parent/C31_SUMMARY.json'),('identity.json','parent/C31_IDENTITY.json'),('RUNTIME.json','parent/C31_RUNTIME.json')]+[(n,n) for n in names]:
            p=out/new;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(old) as src,p.open('wb') as target:shutil.copyfileobj(src,target,8<<20)
            require(sha(p)==manifest[old],'Copied parent member differs')
            copied[new]={'member':old,'sha256':manifest[old]}
        dump(out/'parent/C31_MANIFEST.json',manifest)
    dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':sha(parent),'verified_members':len(manifest),'members':copied,
         'manifest_sha256':sha(out/'parent/C31_MANIFEST.json'),'parent_modified':False,
         'original_C31_64_state_tensors_copied':True,'parent_ZIP_not_embedded':True})
    validate(out,root);dump(out/'PREPARED.json',{'complete':True,'n_states':64})
    print('C34 actual C31 states + frozen C25/C27 writer imported; old76 parameter states not needed.',flush=True)

def validate(out,root):
    out=Path(out);prov=read(out/'PARENT_PROVENANCE.json');manifest=read(out/'parent/C31_MANIFEST.json')
    require(sha(out/'parent/C31_MANIFEST.json')==prov['manifest_sha256'],'Stored parent manifest differs')
    for n,r in prov['members'].items():require(sha(out/n)==r['sha256']==manifest[r['member']],'Imported parent bytes changed: '+n)
    i=read(out/'parent/C31_IDENTITY.json');require(i['source_hashes']==read(Path(root)/'C31_SOURCE_PINS.json') and i['settings']==read(Path(root)/'C31_SETTINGS.json'),'Imported C31 identity differs')
    from c31_protocol import specs
    fixed=read(out/'FIXED_STATE_SPECS.json')
    require(fixed['new_training']==specs('new_training') and fixed['prospective']==specs('prospective'),'Fixed seeds/specs changed')
    allspec=fixed['new_training']+fixed['prospective']
    require(len(allspec)==64 and len(set(t['id'] for t in allspec))==64,'Fixed64 denominator differs')
    return allspec

def resume_score(path,root):
    with zipfile.ZipFile(path) as z:
        verify(z,True);require(zread(z,'identity.json')=={'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)},'C34 source/settings differ')
        require(zread(z,'PREPARED.json')['complete'],'Use complete C31 input: import not complete')
        ns=z.namelist()
        return (int('SUMMARY.json' in ns and zread(z,'SUMMARY.json')['complete']),int('BASES_FROZEN.json' in ns),
                sum(n.startswith('basis/') and n.endswith('/COMPLETE.json') for n in ns),
                sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in ns))
