"""Check real nested extraction paths using a tiny synthetic archive.
The external C25 semantic/tensor audit boundary is mocked here; byte verification is real.
"""
import io,zipfile,json,hashlib,tempfile
from pathlib import Path
from unittest.mock import patch
import prepare_c27 as p
from c27_protocol import CASES


def zbytes(data,manifest=False):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for n,v in data.items():z.writestr(n,v)
  if manifest:z.writestr('MANIFEST.json',json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in data.items()}))
 return b.getvalue()


def main():
 train=[{'id':c+'__v%d'%i,'case':c} for c in CASES for i in range(19)]
 original=[t['id'] for t in train if int(t['id'].split('__v')[1])<15]
 new=[{'id':c+'__c25_new%d'%i,'case':c} for c in CASES for i in range(4)]
 orig={'c9_checkpoint.pt':b'original-c9-checkpoint'};promoted={};root={}
 for t in train:
  sid=t['id'];dest=orig if sid in original else promoted;prefix=('training/' if sid in original else 'evaluation_states/')+sid+'/'
  for name in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):dest[prefix+name]=(sid+':'+name).encode()
  root['training_replay/'+sid+'/REFERENCE.json']=b'{}'
  if t['case']=='xor_10':root['calibration/round_000030/states/'+sid+'/JACOBIAN.pt']=(sid+':J').encode()
 chain=zbytes({'training_capsule.zip':zbytes(orig)})
 chain=zbytes({'training_capsule.zip':chain})
 promoted['c23_capsule.zip']=chain
 root['parent_capsule.zip']=zbytes({'c24_capsule.zip':zbytes(promoted)})
 member='heads/calibrated/checkpoint_000030_final.pt';root[member]=b'actual-final-checkpoint'
 for name in ('SUMMARY.json','SPLIT.json','FINAL_TRAINING.json','WRITER_FROZEN.json','identity.json','source/MODEL_PINS.json'):root[name]=b'{}'
 for t in new:
  for name in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):root['states/new/'+t['id']+'/'+name]=(t['id']+':'+name).encode()
  root['evaluation/new/'+t['id']+'/COMPLETE.json']=b'{}'
 blob=zbytes(root,True);sha=hashlib.sha256(blob).hexdigest()
 parent={'checkpoint_member':member,'original':original,'split':{'training':train,'prospective_new_states':new},'progress':{'checkpoint_sha256':hashlib.sha256(root[member]).hexdigest()}}
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);source=d/'C25.zip';source.write_bytes(blob);out=d/'session/run'
  with patch.object(p,'PARENT_SHA',sha),patch.object(p,'PARENT_CP_SHA',parent['progress']['checkpoint_sha256']),patch.object(p,'inspect_parent',lambda z,pins:parent),patch.object(p,'tensor_identity',lambda z,p,torch:None):p.prepare(source,out)
  assert (out/'data/training/xor_10__v0/INPUT.pt').read_bytes()==b'xor_10__v0:INPUT.pt'
  assert (out/'data/training/xor_10__v18/INPUT.pt').read_bytes()==b'xor_10__v18:INPUT.pt'
  assert (out/'data/inspected/xor_11__c25_new3/INPUT.pt').read_bytes()==b'xor_11__c25_new3:INPUT.pt'
  assert (out/'parent/C9_CHECKPOINT.pt').read_bytes()==b'original-c9-checkpoint'
  assert source.read_bytes()==blob
  print('NESTED_PREPARATION_FLOW_PASS: original60,promoted16,inspected16,19 geometry files,normalisation and actual checkpoint paths copied with byte provenance; source unchanged.')


if __name__=='__main__':main()
