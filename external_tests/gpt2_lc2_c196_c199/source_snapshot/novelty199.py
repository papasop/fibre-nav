"""Bound-input novelty inventory only; no claim of global historical non-access."""
from core199 import *
def collect(obj,history,key=''):
 if isinstance(obj,dict):
  for k,v in obj.items():collect(v,history,k)
 elif isinstance(obj,list):
  for v in obj:collect(v,history,key)
 elif key=='state_id' and isinstance(obj,str):history['state_ids'].add(obj)
 elif (key=='seed' or key.endswith('_seed')) and type(obj) is int:history['seeds'].add(obj)
 elif isinstance(obj,str) and re.fullmatch('[0-9a-f]{64}',obj) and any(k in key.lower() for k in ('sha256','hash','digest')):history['hashes'].add(obj)
def inventory(paths,ctx,out,budget=lambda:None):
 h=dict(hashes=set(),state_ids=set(),seeds=set());counts={}
 for k,p in paths.items():
  budget();njson=nline=0
  with contextlib.closing(Archive(p)) as z:
   for n in z.z.namelist():
    if n.startswith('source/') or '/source/' in n:continue
    if n.endswith('.json'):
     collect(z.json(n),h);njson+=1
    elif n.endswith('.jsonl'):
     with z.z.open(n) as f:
      for line in f:
       need(len(line)<=16*1024**2,'Bounded JSONL record');collect(clean_json(line),h);nline+=1
    if (njson+nline)%256==0:budget()
  counts[k]=dict(JSON_records=njson,JSONL_records=nline)
 d=ctx['data']['d']
 for sid,q in d['points'].items():
  h['state_ids'].add(sid)
  for name in ('prepared','raw'):h['hashes'].add(safe.tensor_hash(q[name]))
 for points in (ctx['C196_final_points'],ctx['C198_fresh_points']['0'],ctx['C198_fresh_points']['1']):
  for p in points.values():h['hashes'].add(safe.tensor_hash(p))
 historical=d.get('historical_hashes',[])
 for x in historical:
  if isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x):h['hashes'].add(x)
 dump(Path(out)/'BOUND_HISTORY_NOVELTY_INVENTORY.json',dict(archive_counts=counts,state_ids=sorted(h['state_ids']),seeds=sorted(h['seeds']),hashes=sorted(h['hashes']),source_payloads_scanned=False,inherited_historical_hashes_available='historical_hashes' in d,programme_wide_nonaccess_proved=False,chronology_proved=False));return h
