"""Frozen C4 paired projection ablation; no newly fitted learner."""
from pathlib import Path,PurePosixPath
import hashlib,io,json,math,zipfile

PROTOCOL='GPT2_C4_PROJECTION_ABLATION_A100_V1'
C3_PROTOCOL='GPT2_C3_TRAINING_GEOMETRY_A100_V1'
C3_SHA='96b4b923670a80cc9b338886878e5ec469a59a1d862b041c49efc63b62858239'
C2_PROTOCOL='GPT2_C2_C1_TRAIN_RECHECK_GEOMETRY_A100_V1'
C2_SHA='25366e8133d8900255dbe6ed930072e6f38faf7da70cea95552c7a596ebd09f5'
C1_PROTOCOL='GPT2_C1_ONE_FORWARD_WRITER_LOO_A100_V1'
C1_SHA='223455a96f1ddbf332f64b2b1e0b39320ca111c77d6d607d2c05b67f6528e52c'
V5_SHA='30ac85d7208ba1525b7cc8934ffc94e4d42bb8b593a17bba715db2f652dd66dd'
REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
CASES=('xor_00','xor_01','xor_10','xor_11')
SETTINGS={'margin_gate':.05,'response_budget':.002,'kl_budget':.005,
          'svd_relative_cutoff':1e-8,'svd_absolute_cutoff':1e-10,
          'correction_ridge':1e-6,'correction_max_steps':8,
          'correction_backtracking':6,'correction_checkpoint_every':2,
          'correction_acceptance':'strict measured infinity-response reduction only; final all gates mandatory',
          'head_training_steps':0,'memory_loss_optimiser_steps':0}

JOBS=tuple(('training',fold,case) for fold in CASES for case in CASES if case!=fold)+tuple(('heldout',case,case) for case in CASES)

def job_id(kind,fold,case):
    if (kind,fold,case) not in JOBS:raise ValueError('Undeclared paired job')
    return kind+'__'+fold+'__'+case

def arm_order(index):
    return ('direct','combined') if index%2==0 else ('combined','direct')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify_open_zip(z,protocol=None):
    names=z.namelist()
    if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
    if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
        raise ValueError('Unsafe ZIP path')
    if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    if 'MANIFEST.json' not in names:raise ValueError('Result ZIP required; MANIFEST.json missing')
    m=json.loads(z.read('MANIFEST.json'))
    if {n for n in names if not n.endswith('/')}!=set(m)|{'MANIFEST.json'}:
        raise ValueError('Manifest coverage mismatch')
    for n,h in m.items():
        if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Hash mismatch: '+n)
    ident=json.loads(z.read('identity.json'))
    if protocol and ident.get('protocol')!=protocol:raise ValueError('Wrong protocol')
    return ident

def verified_zip(path,protocol=None):
    with zipfile.ZipFile(path) as z:return verify_open_zip(z,protocol)


def validate_c3(path):
    if sha(path)!=C3_SHA:raise ValueError('C4 V1 requires the exact inspected complete C3 result ZIP')
    with zipfile.ZipFile(path) as z:
        ident=verify_open_zip(z,C3_PROTOCOL);summary=json.loads(z.read('SUMMARY.json'))
        if not summary.get('complete') or summary.get('n_completed')!=12 or not summary.get('positive_references_all_pass'):
            raise ValueError('Complete C3 with valid references required')
        b=z.read('parent_c2.zip')
        if hashlib.sha256(b).hexdigest()!=C2_SHA:raise ValueError('C2 parent identity changed')
        with zipfile.ZipFile(io.BytesIO(b)) as c2:
            verify_open_zip(c2,C2_PROTOCOL);b=c2.read('parent_c1.zip')
            if hashlib.sha256(b).hexdigest()!=C1_SHA:raise ValueError('C1 parent identity changed')
            with zipfile.ZipFile(io.BytesIO(b)) as c1:
                verify_open_zip(c1,C1_PROTOCOL);b=c1.read('parent_v5.zip')
                if hashlib.sha256(b).hexdigest()!=V5_SHA:raise ValueError('V5 identity changed')
                with zipfile.ZipFile(io.BytesIO(b)) as v5:verify_open_zip(v5,'GPT2_B5_XOR_CONTINUATION_A100_V5')
        for kind,fold,case in JOBS:
            if kind=='training':
                name=fold+'__'+case+'/raw.pt'
                if name not in z.namelist():raise ValueError('Saved C3 training raw candidate missing')
    return ident

def paired_counts(rows):
    a=[r['direct'].get('episode_pass',False) for r in rows]
    b=[r['combined'].get('episode_pass',False) for r in rows]
    return {'n':len(rows),'direct_passed':sum(a),'combined_passed':sum(b),
            'both_passed':sum(x and y for x,y in zip(a,b)),
            'direct_only':sum(x and not y for x,y in zip(a,b)),
            'combined_only':sum(not x and y for x,y in zip(a,b)),
            'both_failed':sum(not x and not y for x,y in zip(a,b))}

def target_from_reads(bits):
    if len(bits)!=4 or any(b not in (0,1) for b in bits):raise ValueError('Expected 4 binary cells')
    return list(bits[:3])+[bits[0]^bits[1]]

def endpoint_gate(v,target):
    if 'evaluation' not in v:return False
    e=v['evaluation'];a=e['exact_margins'];b=e['heldout_margins']
    if len(a)!=4 or len(b)!=12:return False
    vals=a+b+[v['drift'],v['kl']]
    return (all(math.isfinite(x) for x in vals) and v['bits']==target
            and min(a+b)>=SETTINGS['margin_gate'] and v['drift']<=SETTINGS['response_budget']
            and v['kl']<=SETTINGS['kl_budget'])

def episode_gate(start,v):
    target=target_from_reads(start)
    return bool(start[3]!=target[3] and endpoint_gate(v,target)
                and v['bits'][:3]==start[:3] and v['bits'][3]!=start[3])

def training_diagnosis(n_passed,n_total,test_pass):
    if n_passed<n_total:return 'TRAINING_ENDPOINT_FAILURE_PRESENT'
    return 'TRAINING_ENDPOINTS_PASS_HELDOUT_'+('PASS' if test_pass else 'FAIL')
