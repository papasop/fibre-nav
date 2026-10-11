import json, zipfile, shutil
from pathlib import Path
from b5_protocol import verified_zip, sha, PROTOCOL

PREPARATION_TOTAL = 1200
WRITE_TOTAL = 1080
V4_PROTOCOL = 'GPT2_B5_XOR_TRUTH_TABLE_A100_V4'

def validate_parent_summary(s):
    if s.get('protocol') != V4_PROTOCOL or not s.get('complete'):
        raise ValueError('A complete V4 result is required')
    if s.get('n_completed') != 4 or s.get('n_passed') != 2:
        raise ValueError('This continuation requires the recorded 2/4 V4 outcome')
    expected = {'xor_00': True, 'xor_01': True, 'xor_10': False, 'xor_11': False}
    if {k:v['pass'] for k,v in s.get('cases', {}).items()} != expected:
        raise ValueError('Unexpected V4 case outcomes; do not guess recovery')

def initialise(parent, out):
    parent, out = Path(parent), Path(out)
    verified_zip(parent, V4_PROTOCOL)
    with zipfile.ZipFile(parent) as z:
        validate_parent_summary(json.loads(z.read('SUMMARY.json')))
        prep11=json.loads(z.read('xor_11/PREPARATION.json'))
        if not prep11['pass'] or prep11['target_bits'] != [1,1,1,1]:
            raise ValueError('Missing verified xor_11 prepared state')
        if 'xor_10/preparation_optimizer.pt' not in z.namelist():
            raise ValueError('Missing actual xor_10 optimisation checkpoint')
        out.mkdir(parents=True, exist_ok=True)
        fixed=['parent_b4.zip','parent_v3.zip','controller_training.pt',
               'controller_identity.json','controller_progress.json',
               'controller_training_data.json','controller_split.json','controller_readiness.json']
        names=fixed + ['xor_10/preparation_optimizer.pt',
                       'xor_11/prepared.pt','xor_11/PREPARATION.json']
        names += [n for n in z.namelist() if n.startswith(('xor_00/','xor_01/')) and not n.endswith('/')]
        for name in names:
            q=out/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(name))
    shutil.copyfile(parent, out/'parent_v4.zip')
    verified_zip(out/'parent_v3.zip','GPT2_B5_SINGLE_BACKBONE_A100_V3')
    verified_zip(out/'parent_b4.zip','GPT2_B4_MINIMAL_CAUSAL_A100_V1')
    plan={'protocol':PROTOCOL,'parent_v4_sha256':sha(parent),
          'preparation_total':PREPARATION_TOTAL,'write_total':WRITE_TOTAL,
          'thresholds_changed':False,'writer_changed':False,'controller_retrained':False,
          'fresh_evaluation':False,'independent_memory_seed':False,
          'case_origins':{'xor_00':'INHERITED_V4_PASS_NOT_RERUN',
                         'xor_01':'INHERITED_V4_PASS_NOT_RERUN',
                         'xor_10':'RESUME_REAL_V4_PREPARATION_CHECKPOINT_600',
                         'xor_11':'RESTART_WRITE_FROM_V4_PREPARED_STATE; rejected candidate checkpoint absent'},
          'candidate_checkpoint_retention':'All V5 attempts retain optimizer and selected candidate, including rejected writes'}
    (out/'CONTINUATION_PLAN.json').write_text(json.dumps(plan,indent=2))

def resume_score(path):
    verified_zip(path,PROTOCOL)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        summary=json.loads(z.read('SUMMARY.json')) if 'SUMMARY.json' in names else {}
        completed=sum(n.endswith('/COMPLETE.json') for n in names)
        progress=json.loads(z.read('PHASE_PROGRESS.json')) if 'PHASE_PROGRESS.json' in names else {}
        return (int(summary.get('complete',False)),completed,sum(progress.values()))
