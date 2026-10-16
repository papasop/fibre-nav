"""Real ZIP/source/provenance validation; small fake tensor bytes, not model evidence."""
import hashlib,json,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import c28_protocol as p

def main():
 root=Path(__file__).parent;members={};split={'training':[],'prospective_new_states':[]}
 for case in p.CASES:
  for i in range(19):split['training'].append({'id':case+'__train%d'%i,'case':case})
  for i in range(4):split['prospective_new_states'].append({'id':case+'__new%d'%i,'case':case})
 pins=json.loads((root/'C27_SOURCE_PINS.json').read_text())
 def put(n,v):members[n]=json.dumps(v).encode()
 for n in pins:members['source/'+n]=(root/'inherited'/n).read_bytes()
 put('identity.json',{'protocol':p.PARENT_PROTOCOL,'source_hashes':pins})
 put('SUMMARY.json',{'protocol':p.PARENT_PROTOCOL,'complete':True,'base_head_sha256':p.HEAD_SHA,'base_head_unchanged':True,'all_initial74_passes_preserved':True,'training':{'state_bypass':{'passed':76,'valid':76}}})
 checkpoint=b'FIXTURE frozen branch, not tensor';featuremap=b'FIXTURE featuremap'
 cpsha=hashlib.sha256(checkpoint).hexdigest();fsha=hashlib.sha256(featuremap).hexdigest()
 put('arms/state_bypass/FROZEN.json',{'accepted_refinement':True,'matrix_sha256':p.PARENT_MATRIX_SHA,'checkpoint_sha256':cpsha,'feature_map_sha256':fsha})
 members['arms/state_bypass/FINAL_CHECKPOINT.pt']=checkpoint;members['arms/state_bypass/FEATURE_MAP.pt']=featuremap
 for n in ('C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','MODEL_PINS.json'):members['parent/'+n]=b'fixture'
 put('parent/SPLIT.json',split)
 for cohort,ts in [('training',split['training']),('inspected',split['prospective_new_states'])]:
  for t in ts:
   b='data/'+cohort+'/'+t['id']+'/'
   for n in ('prepared.pt','INPUT.pt','REFERENCE.json'):members[b+n]=(b+n).encode()
   if cohort=='inspected':put('evaluation/'+t['id']+'/COMPLETE.json',{'fixture':True})
 put('MANIFEST.json',{n:hashlib.sha256(b).hexdigest() for n,b in members.items()})
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'parent.zip';out=Path(d)/'session/run'
  with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
   for n,b in members.items():z.writestr(n,b)
  before=path.read_bytes()
  with patch.object(p,'PARENT_BRANCH_CP_SHA',cpsha):p.prepare_parent(path,out,root)
  assert path.read_bytes()==before
  prov=json.loads((out/'PARENT_PROVENANCE.json').read_text())
  for n,r in prov['members'].items():assert p.file_sha(out/n)==r['sha256']
  assert len(list((out/'data/training').glob('*/INPUT.pt')))==76
  assert len(list((out/'data/inspected').glob('*/INPUT.pt')))==16
  assert (out/'parent/C27_BRANCH.pt').read_bytes()==checkpoint
  print('COMPACT_C27_PREPARATION_PASS: actual source hashes, manifest CRC/member hashes,76+16 path mapping and immutable original bytes; tensor bytes in this fixture are placeholders.')
if __name__=='__main__':main()
