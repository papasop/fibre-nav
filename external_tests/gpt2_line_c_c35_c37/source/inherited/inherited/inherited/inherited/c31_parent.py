"""Compact frozen writer import from the same completed C29 used by C30."""
import shutil, zipfile
from pathlib import Path
from c30_data import probe, validate_split
from c31_protocol import *


def prepare_parent(parent, out, root):
    out=Path(out);provenance={}
    with zipfile.ZipFile(parent) as z:
        manifest=verify(z);probe(z,root,True)
        split=zread(z,'parent/SPLIT.json');tr,dv=validate_split(split)
        def cp(old,new):
            p=out/new;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(old) as src,p.open('wb') as dest:shutil.copyfileobj(src,dest,8<<20)
            provenance[new]={'member':old,'sha256':sha(p)}
        for old,new in [('SUMMARY.json','parent/C29_SUMMARY.json'),('identity.json','parent/C29_IDENTITY.json'),
                        ('FROZEN_IDENTITIES.json','parent/C29_FROZEN_IDENTITIES.json')]:cp(old,new)
        for n in ('SPLIT.json','MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt','C27_FROZEN.json'):
            cp('parent/'+n,'parent/'+n)
        for t in tr:
            sid=t['id'];b='data/training/'+sid+'/'
            for n in ('INPUT.pt','REFERENCE.json','C29_INPUT_AUDIT.json'):cp(b+n,b+n)
            cp('diagnosis/training/'+sid+'/COMPLETE.json',b+'C29_COMPLETE.json')
            if sid in [c+'__v%d'%i for c in CASES for i in range(3)]:cp(b+'prepared.pt',b+'prepared.pt')
        dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':sha(parent),'verified_members':len(manifest),'members':provenance,
             'parent_archives_modified':False,'old_inspected16_not_imported':True,'parent_weights_loaded_later_for_identity_checks':True})
    dump(out/'PREPARED.json',{'complete':True,'historical_training_states':76,'original_preparations':12,'old_inspected_states_imported':0})
    print('C31 original C29 writer + historical76 imported; no old16 evaluation data imported.',flush=True)


def resume_score(path,root):
    with zipfile.ZipFile(path) as z:
        verify(z,True);i=zread(z,'identity.json')
        require(i['protocol']==PROTOCOL and i['settings']==SETTINGS and i['source_hashes']==sources(root),'C31 source/settings differ')
        ns=set(z.namelist())
        return (int('SUMMARY.json' in ns),int('PREDICTORS_FROZEN.json' in ns),
                sum(n.startswith('states/prospective/') and n.endswith('/COMPLETE.json') for n in ns),
                sum(n.startswith('states/new_training/') and n.endswith('/COMPLETE.json') for n in ns),
                sum(n.startswith('predictors/') and n.endswith('/COMPLETE.json') for n in ns))
