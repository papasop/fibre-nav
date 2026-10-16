"""Read-only compact C34 import; old16 input/target payloads stay opaque until freeze."""
from c35_protocol import *
def probe(z,root,full=False):
    i=zread(z,'identity.json');s=zread(z,'SUMMARY.json')
    require(i['protocol']==s['protocol']==PARENT_PROTOCOL and s['complete'] and s['decision']=='C34_LOW_DIMENSIONAL_ORACLE_COMPARISON_COMPLETE','Completed C34 RESULTS ZIP required')
    require(i['source_hashes']==read(Path(root)/'C34_SOURCE_PINS.json') and i['settings']==read(Path(root)/'C34_SETTINGS.json'),'Patched C34 source/settings required')
    require(s['all64_requested_states_retained'] and s['original_C25_C27_writer_unchanged'],'C34 scope differs')
    fr=zread(z,'BASES_FROZEN.json');require(hashlib.sha256(z.read('BASES_FROZEN.json')).hexdigest()==s['bases_frozen_sha256'],'C34 basis freeze digest differs')
    require(fr['basis_training_ids']==SETTINGS['prior_basis_states'] and fr['frozen_before_nonbasis_endpoint_evaluation'],'C34 fixed prior IDs/order differ')
    require(s['original_writer_sha256']==hashlib.sha256(z.read('WRITER_FROZEN.json')).hexdigest()==fr['writer_frozen_sha256'],'Original frozen writer binding differs')
    if full:
        for n,h in i['source_hashes'].items():require(hashlib.sha256(z.read('source/'+n)).hexdigest()==h,'C34 source member differs: '+n)
        for n,h in fr['files'].items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'Frozen C34 basis/geometry differs: '+n)
    return i,s
def prepare(parent,out,root):
    out=Path(out);copied={}
    with zipfile.ZipFile(parent) as z:
        manifest=verify(z);_,s=probe(z,root,True);fixed=zread(z,'FIXED_STATE_SPECS.json')
        from c31_protocol import specs as old_specs
        require(fixed['new_training']==old_specs('new_training') and fixed['prospective']==old_specs('prospective'),'C31 source state specs differ')
        names=['WRITER_FROZEN.json','FIXED_STATE_SPECS.json']
        names+=['parent/'+n for n in ('MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt','C31_SUMMARY.json','C31_IDENTITY.json','C31_RUNTIME.json')]
        fr=zread(z,'BASES_FROZEN.json');names+=list(fr['files'])
        for cohort in ('new_training','prospective'):
            for t in fixed[cohort]:
                names+=['states/'+cohort+'/'+t['id']+'/'+n for n in ('INPUT.pt','prepared.pt','RAW_WRITTEN.pt','COMPLETE.json')]
                names+=['evaluation/'+t['id']+'/COMPLETE.json']
        rename=[('SUMMARY.json','parent/C34_SUMMARY.json'),('identity.json','parent/C34_IDENTITY.json'),('BASES_FROZEN.json','parent/C34_BASES_FROZEN.json')]
        for old,new in rename+[(n,n) for n in names]:
            p=out/new;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(old) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst,8<<20)
            require(sha(p)==manifest[old],'Compact copy differs');copied[new]={'member':old,'sha256':manifest[old]}
        dump(out/'parent/C34_MANIFEST.json',manifest)
    dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':sha(parent),'verified_members':len(manifest),'members':copied,
       'manifest_sha256':sha(out/'parent/C34_MANIFEST.json'),'parent_modified':False,'parent_ZIP_not_embedded':True,
       'old16_not_used_for_training':True,'old16_payloads_opaque_during_import':True})
    validate(out,root);dump(out/'PREPARED.json',{'complete':True,'training':48,'inspected':16})
    print('C35 compact actual C34 bases/writer/states imported; no old16 target used for fitting.',flush=True)
def validate(out,root):
    out=Path(out);prov=read(out/'PARENT_PROVENANCE.json');manifest=read(out/'parent/C34_MANIFEST.json')
    require(sha(out/'parent/C34_MANIFEST.json')==prov['manifest_sha256'],'Stored parent manifest changed')
    for n,r in prov['members'].items():require(sha(out/n)==r['sha256']==manifest[r['member']],'Imported parent member changed: '+n)
    i=read(out/'parent/C34_IDENTITY.json');require(i['source_hashes']==read(Path(root)/'C34_SOURCE_PINS.json') and i['settings']==read(Path(root)/'C34_SETTINGS.json'),'Stored C34 identity differs')
    fr=read(out/'parent/C34_BASES_FROZEN.json');s=read(out/'parent/C34_SUMMARY.json')
    require(sha(out/'parent/C34_BASES_FROZEN.json')==s['bases_frozen_sha256'],'Stored C34 frozen basis changed')
    for n,h in fr['files'].items():require(sha(out/n)==h,'Pinned C34 basis/geometry changed')
    require(sha(out/'WRITER_FROZEN.json')==s['original_writer_sha256']==fr['writer_frozen_sha256'],'Original writer identity differs')
    from c31_protocol import specs as old_specs
    fixed=read(out/'FIXED_STATE_SPECS.json');require(fixed['new_training']==old_specs('new_training') and fixed['prospective']==old_specs('prospective'),'Imported state specs differ')
    return fixed
def resume_score(path,root):
    with zipfile.ZipFile(path) as z:
        verify(z,True);require(zread(z,'identity.json')=={'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)},'Other C35 source/settings')
        require(zread(z,'PREPARED.json')['complete'],'Import incomplete; use full C34 result')
        ns=z.namelist();return (int('SUMMARY.json' in ns and zread(z,'SUMMARY.json')['complete']),
             int('COEFFICIENTS_FROZEN.json' in ns),sum(n.startswith('fit/') and n.endswith('/COMPLETE.json') for n in ns),
             sum(n.startswith('results/') and n.endswith('/COMPLETE.json') for n in ns))
