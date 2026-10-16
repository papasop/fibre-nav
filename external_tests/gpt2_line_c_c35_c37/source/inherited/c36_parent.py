"""Strict C35 archive/source import. Original member bytes never overwritten."""
from c36_protocol import *
from c35_protocol import SETTINGS as C35_SETTINGS

def probe(z,root,full=False):
    require('identity.json' in z.namelist(),'Select C35 RESULTS ZIP, not source ZIP')
    i=zread(z,'identity.json');s=zread(z,'SUMMARY.json')
    require(i=={'protocol':PARENT_PROTOCOL,'settings':C35_SETTINGS,'source_hashes':read(Path(root)/'C35_SOURCE_PINS.json')},'C35 source/settings mismatch')
    require(s['protocol']==PARENT_PROTOCOL and s['complete'] and s['decision']==SETTINGS['parent_expected_decision'],'Completed passing C35 result required')
    fh=hashlib.sha256(z.read('COEFFICIENTS_FROZEN.json')).hexdigest()
    require(fh==s['coefficients_frozen_sha256']==SETTINGS['expected_parent_freeze_sha256'],'Not the recorded C35 frozen writer')
    for k,v in {'old16_used_for_training_or_selection':False,'all_heads_frozen_before_new_state_generation':True,
                'candidate_points_sealed_before_all_endpoint_audits':True,'postwrite_response_used_for_proposal':False,
                'online_write_Jacobians':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,
                'online_write_backtracking':0,'teacher_fallback_used':False}.items():
        require(s[k]==v,'C35 scope/policy mismatch: '+k)
    if full:
        for n,h in i['source_hashes'].items():require(hashlib.sha256(z.read('source/'+n)).hexdigest()==h,'Parent source member changed: '+n)
    return s

def prepare(path,out,root):
    out=Path(out);data=out/'c35';data.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        manifest=verify(z);s=probe(z,root,True);members={}
        for n,h in manifest.items():
            if n.startswith('source/'):continue
            p=data/n;p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(n) as f,p.open('wb') as g:shutil.copyfileobj(f,g,8<<20)
            require(sha(p)==h,'Imported C35 member differs: '+n);members[n]=h
        dump(out/'PARENT_MANIFEST.json',manifest)
    dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':sha(path),'parent_modified':False,'parent_zip_not_embedded':True,
      'members':members,'verified_members':len(manifest),'manifest_sha256':sha(out/'PARENT_MANIFEST.json'),
      'c35_freeze_sha256':sha(data/'COEFFICIENTS_FROZEN.json')})
    validate(out,root)
    dump(out/'PREPARED.json',{'complete':True,'parent_decision':s['decision']})

def validate(out,root):
    out=Path(out);data=out/'c35';p=read(out/'PARENT_PROVENANCE.json');m=read(out/'PARENT_MANIFEST.json')
    require(sha(out/'PARENT_MANIFEST.json')==p['manifest_sha256'],'Parent manifest changed')
    require(p['members']=={n:h for n,h in m.items() if not n.startswith('source/')},'Compact parent member coverage changed')
    for n,h in p['members'].items():require(sha(data/n)==h,'Original C35 member changed: '+n)
    require(read(data/'identity.json')=={'protocol':PARENT_PROTOCOL,'settings':C35_SETTINGS,'source_hashes':read(Path(root)/'C35_SOURCE_PINS.json')},'Imported C35 identity changed')
    require(sha(data/'COEFFICIENTS_FROZEN.json')==p['c35_freeze_sha256']==SETTINGS['expected_parent_freeze_sha256'],'Imported frozen C35 differs')
    return data

def resume_score(path,root):
    with zipfile.ZipFile(path) as z:
        verify(z,True);require(zread(z,'identity.json')==identity(root),'C36 source/settings mismatch')
        require(zread(z,'PREPARED.json')['complete'],'Incomplete C35 import; use C35 input again')
        ns=z.namelist()
        return (int('SUMMARY.json' in ns and zread(z,'SUMMARY.json')['complete']),
                int('C36_FROZEN.json' in ns),int('PROPOSAL_REPRODUCTION.json' in ns),
                sum(n.startswith('replay/') and n.endswith('.json') for n in ns),
                sum(n.startswith('c35/results/') and '__c36_new' in n and n.endswith('/COMPLETE.json') for n in ns))
