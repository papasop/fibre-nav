"""Actual source/member hashes and compact path extraction; tensor bytes are fixtures."""
import hashlib,json,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import c29_protocol as p

def main():
 root=Path(__file__).parent;members={};split={'training':[],'prospective_new_states':[]}
 for c in p.CASES:
  for i in range(19):split['training'].append({'id':c+'__train%d'%i,'case':c})
  for i in range(4):split['prospective_new_states'].append({'id':c+'__new%d'%i,'case':c})
 def put(n,v):members[n]=json.dumps(v).encode()
 sources=json.loads((root/'C28_SOURCE_PINS.json').read_text())
 for n in sources:members['source/'+n]=(root/'inherited'/n).read_bytes()
 put('identity.json',{'protocol':p.PARENT_PROTOCOL,'source_hashes':sources})
 put('SUMMARY.json',{'protocol':p.PARENT_PROTOCOL,'complete':True,'parent_C27_matrix_sha256':p.PARENT_MATRIX_SHA,'parent_C27_unchanged':True,'all76_training_passes_preserved':True,'training':{arm:{'passed':76,'valid':76} for arm in p.ARMS}})
 for n in ('C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','MODEL_PINS.json','C27_BRANCH.pt','C27_FEATURE_MAP.pt','C27_FROZEN.json'):members['parent/'+n]=b'fixture'
 put('parent/SPLIT.json',split);put('BASELINE_REPLAY.json',{})
 pins={}
 for arm in p.ARMS:
  b=('frozen checkpoint '+arm).encode();members['arms/'+arm+'/FINAL_CHECKPOINT.pt']=b
  pins[arm]={'checkpoint_sha256':hashlib.sha256(b).hexdigest(),'matrices_sha256':p.PINS[arm]['matrices_sha256']}
  put('arms/'+arm+'/FROZEN.json',pins[arm])
  for case in p.CASES:members['arms/'+arm+'/'+case+'/FEATURE_MAP.pt']=b'featuremap fixture'
 for cohort,ss in [('training',split['training']),('inspected',split['prospective_new_states'])]:
  for t in ss:
   sid=t['id'];prefix='data/'+cohort+'/'+sid+'/'
   for n in ('prepared.pt','INPUT.pt','REFERENCE.json'):members[prefix+n]=(prefix+n).encode()
   if cohort=='training':members['geometry/'+sid+'.pt']=b'original geometry fixture'
   else:put('evaluation/'+sid+'/COMPLETE.json',{'fixture':True})
 put('MANIFEST.json',{n:hashlib.sha256(v).hexdigest() for n,v in members.items()})
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'parent.zip';out=Path(d)/'session/run'
  with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
   for n,b in members.items():z.writestr(n,b)
  before=path.read_bytes()
  with patch.object(p,'PINS',pins):p.prepare_parent(path,out,root)
  assert path.read_bytes()==before
  assert len(list((out/'geometry').glob('*.pt')))==76
  assert len(list((out/'data/training').glob('*/INPUT.pt')))==76
  assert len(list((out/'data/inspected').glob('*/INPUT.pt')))==16
  assert (out/'data/inspected/xor_10__new2/C28_COMPLETE.json').exists()
  prov=json.loads((out/'PARENT_PROVENANCE.json').read_text())
  assert all(p.file_sha(out/n)==r['sha256'] for n,r in prov['members'].items())
  print('C29_COMPACT_PREPARATION_PASS: full source bytes, member hashes, original geometry76 and training76/inspected16 paths; no parent mutation. Fake tensor bytes used only in this extraction fixture.')
if __name__=='__main__':main()
