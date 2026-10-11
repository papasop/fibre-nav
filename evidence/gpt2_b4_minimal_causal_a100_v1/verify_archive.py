"""Read-only archive/record verification; not a GPT-2 rerun or B4 certification."""
from pathlib import Path
import zipfile,json,hashlib,ast,base64
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def verify():
 provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
 archive=ROOT/'gpt2_b4_minimal_causal_a100_results.zip'
 assert sha(archive.read_bytes())==provenance['original_result_sha256']
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();assert len(names)==len(set(names))
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in names)
  assert z.testzip() is None
  m=json.loads(z.read('MANIFEST.json'));assert set(names)==set(m)|{'MANIFEST.json'}
  for n,h in m.items():assert sha(z.read(n))==h,n
  identity=json.loads(z.read('identity.json'));summary=json.loads(z.read('SUMMARY.json'))
  assert identity['protocol']==summary['protocol']=='GPT2_B4_MINIMAL_CAUSAL_A100_V1'
  assert summary['decision']=='B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS' and summary['complete'] and summary['all_gates_pass']
  assert summary['development_only'] and not summary['full_b4_confirmation'] and not summary['b5_claim']
  assert summary['n_planned']==summary['n_passed']==1 and summary['new_seed_total']==0
  for n,h in identity['engine_sha256'].items():assert sha(z.read('source/'+n))==h
  source_manifest=json.loads(z.read('source/SOURCE_MANIFEST.json'))
  for n,h in source_manifest.items():assert sha(z.read('source/'+n))==h,n
  for n in names:
   if n.startswith('source/'):assert (ROOT/n).read_bytes()==z.read(n)
   elif n.endswith('.json'):assert (ROOT/'recorded'/n).read_bytes()==z.read(n)
  rec=json.loads(z.read('seeds/82931/records.json'))
  expected={'prepare':[0,1,1,0],'normal_writeback':[0,1,1,1],'blocked_writeback':[0,1,1,0],'input_substitution':[1,1,1,1],'substituted_writeback':[1,1,1,0]}
  assert set(rec)=={'01_'+k for k in expected}
  for role,bits in expected.items():
   r=rec['01_'+role];ev=r['evaluation']
   assert r['target_bits']==r['observed_bits']==bits and r['pass']
   assert ev['exact_accuracy']==ev['heldout_accuracy']==1
   assert len(ev['exact_margins'])==4 and len(ev['heldout_margins'])==12
   assert min(ev['exact_margins'])==ev['minimum_exact_margin']>=.05
   assert min(ev['heldout_margins'])==ev['minimum_heldout_margin']>=.05
   assert r['response_drift']<=.002 and r['anchor_kl']<=.005
   assert len(z.read('seeds/82931/01_'+role+'.pt'))>0
  c=json.loads(z.read('seeds/82931/cases.json'));assert set(c)=={'01'};c=c['01'];t=c['traces']
  assert c['pass'] and len(c['checks'])==19 and all(c['checks'].values()) and all(c['reload_checks'].values())
  for role,bits in {'normal':[0,1],'blocked':[1,1],'substituted':[1,1],'reset':[0,0],'restored':[0,1]}.items():
   assert t[role]['read_bits']==bits and t[role]['value']==(bits[0]^bits[1])
  assert t['blocked']['read_blocked'] and t['blocked']['replacement_bits']==[1,1]
  assert all(not t[k]['read_blocked'] for k in ('normal','substituted','reset','restored'))
  assert c['reset_bits']==[0,0,0,0] and t['reset']['value']!=t['normal']['value']==t['restored']['value']
  with zipfile.ZipFile(ROOT/'GPT2_B4_MINIMAL_CAUSAL_A100_V1.zip') as packaged:
   assert packaged.testzip() is None
   assert set(packaged.namelist())=={n[7:] for n in names if n.startswith('source/')}
   for n in packaged.namelist():assert packaged.read(n)==z.read('source/'+n)
  vals={}
  for n in ast.parse((ROOT/'GPT2_B4_MINIMAL_CAUSAL_A100_V1.py').read_text()).body:
   if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
    try:vals[n.targets[0].id]=ast.literal_eval(n.value)
    except (ValueError,TypeError):pass
  payload=base64.b64decode(vals['PAYLOAD']);assert sha(payload)==vals['EXPECTED_SHA256']
  assert payload==(ROOT/'GPT2_B4_MINIMAL_CAUSAL_A100_V1.zip').read_bytes()
 print('PASS: original result CRC,34 member hashes, engine identities, frozen source,19 recorded gates, five endpoint records, launcher payload')
 print('Scope: read-only recorded evidence consistency; model forwards and tensor readouts not independently rerun.')
if __name__=='__main__':verify()
