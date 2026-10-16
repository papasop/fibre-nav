"""Compact immutable required C25 bytes; do not embed its full history archive."""
import argparse,io,json,zipfile,hashlib,os,shutil
from pathlib import Path
import torch
from archive_c26 import verify_archive,inspect_parent,tensor_identity,require,file_sha,read_json
from c27_protocol import PARENT_SHA,PARENT_CP_SHA,dump
ROOT=Path(__file__).resolve().parent


def prepare(source,out):
 source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True);local=out.parent/'c25_readonly.zip'
 if not local.exists() or file_sha(local)!=PARENT_SHA:
  with source.open('rb') as src,local.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
 require(file_sha(local)==PARENT_SHA,'Exact completed C25 archive SHA256 differs')
 pins=json.loads((ROOT/'C25_SOURCE_PINS.json').read_text());provenance={}
 with zipfile.ZipFile(local) as z:
  manifest=verify_archive(z);p=inspect_parent(z,pins);tensor_identity(z,p,torch)
  require(p['progress']['checkpoint_sha256']==PARENT_CP_SHA,'C25 final checkpoint differs')
  def copy(zz,old,new,origin):
   dest=out/new;dest.parent.mkdir(parents=True,exist_ok=True)
   with zz.open(old) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
   provenance[new]={'origin':origin+'/'+old,'sha256':file_sha(dest)}
  copy(z,p['checkpoint_member'],'parent/C25_FINAL_CHECKPOINT.pt','C25')
  for n in ('SUMMARY.json','SPLIT.json','FINAL_TRAINING.json','WRITER_FROZEN.json','identity.json'):
   copy(z,n,'parent/'+n,'C25')
  copy(z,'source/MODEL_PINS.json','parent/MODEL_PINS.json','C25')
  with zipfile.ZipFile(io.BytesIO(z.read('parent_capsule.zip'))) as z0:
   with zipfile.ZipFile(io.BytesIO(z0.read('c24_capsule.zip'))) as inherited:
    with zipfile.ZipFile(io.BytesIO(inherited.read('c23_capsule.zip'))) as z2:
     with zipfile.ZipFile(io.BytesIO(z2.read('training_capsule.zip'))) as z3:
      with zipfile.ZipFile(io.BytesIO(z3.read('training_capsule.zip'))) as training:
       copy(training,'c9_checkpoint.pt','parent/C9_CHECKPOINT.pt','C25/parent_capsule/c24_capsule/c23_capsule/training_capsule/training_capsule')
       original=set(p['original'])
       for index,t in enumerate(p['split']['training'],1):
        sid=t['id'];zz=training if sid in original else inherited;b=('training/' if sid in original else 'evaluation_states/')+sid+'/'
        for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):copy(zz,b+n,'data/training/'+sid+'/'+n,'C25_nested_training' if sid in original else 'C25_nested_promoted')
        copy(z,'training_replay/'+sid+'/REFERENCE.json','data/training/'+sid+'/C25_REPLAY.json','C25')
        if index%19==0:print('COMPACT TRAINING',index,'/76',flush=True)
  for t in p['split']['prospective_new_states']:
   sid=t['id']
   for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):copy(z,'states/new/'+sid+'/'+n,'data/inspected/'+sid+'/'+n,'C25')
   copy(z,'evaluation/new/'+sid+'/COMPLETE.json','data/inspected/'+sid+'/C25_COMPLETE.json','C25')
  for t in p['split']['training']:
   if t['case']=='xor_10':copy(z,'calibration/round_000030/states/'+t['id']+'/JACOBIAN.pt','geometry/'+t['id']+'.pt','C25')
  dump(out/'PARENT_PROVENANCE.json',{'parent_C25_sha256':PARENT_SHA,'members_verified':len(manifest),'members':provenance,'full_C25_embedded':False})
 dump(out/'PREPARED.json',{'complete':True,'parent_C25_sha256':PARENT_SHA,'n_training':76,'n_inspected':16,'saved_J_states':19,'members':len(provenance)})
 local.unlink();print('C27 compact C25 data prepared; actual final head and saved inputs/geometry retained.',flush=True)


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();prepare(a.parent,a.out)
