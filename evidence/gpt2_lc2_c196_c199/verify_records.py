"""Verify delivered source identities and supplied logs, without science ZIP/model claims."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
EXT=ROOT/'external_tests/gpt2_lc2_c196_c199'
def need(v,message):
 if not v:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def extract(text,protocol):
 found=[]
 for match in re.finditer(r'^\{',text,re.M):
  try:s,end=json.JSONDecoder().raw_decode(text[match.start():])
  except ValueError:continue
  if isinstance(s,dict) and s.get('protocol')==protocol and s.get('synthetic_fixture') is False:found.append(s)
 need(len(found)==1,'Exactly one real execution summary required for '+protocol);return found[0]
def main():
 files=json.loads((HERE/'FILES_SHA256.json').read_text())
 for name,h in files.items():
  rel=PurePosixPath(name);need(not rel.is_absolute() and '..' not in rel.parts,'Unsafe package path');need(sha(ROOT/name)==h,'Import identity mismatch: '+name)
 index=json.loads((EXT/'RUN_INDEX.json').read_text());provenance=json.loads((HERE/'PROVENANCE.json').read_text());need(set(index)=={'C196','C197','C198','C199'},'Four registered stages');need(provenance['LC2_closed'] is False and provenance['C199_execution_pending'] is True,'Open boundary')
 summaries={}
 for stage,entry in index.items():
  source=EXT/entry['source_root'];bundle=EXT/entry['bundle'];launcher=EXT/entry['launcher'];need(sha(bundle)==entry['bundle_sha256'],'Original BUNDLE hash '+stage);need(sha(launcher)==entry['launcher_sha256'],'Original launcher hash '+stage);text=launcher.read_text();need(re.search(r"BUNDLE_SHA256='([^']+)'",text)[1]==entry['bundle_sha256'],'Matching launcher/BUNDLE '+stage);need("import_module('launch"+stage[1:]+"')" in text,'Current entrypoint '+stage)
  with zipfile.ZipFile(bundle) as z:
   names=z.namelist();need(len(names)==len(set(names)),'Unique ZIP members');manifest=json.loads(z.read('SOURCE_MANIFEST.json'));need(set(names)==set(manifest)|{'SOURCE_MANIFEST.json'},'Original source inventory '+stage)
   for name in names:
    rel=PurePosixPath(name);need(not rel.is_absolute() and '..' not in rel.parts,'Safe ZIP path');b=z.read(name)
    if name in manifest:need(hashlib.sha256(b).hexdigest()==manifest[name],'Source member hash '+name)
    need((source/name).read_bytes()==b,'Exact canonical source correspondence '+stage+'/'+name)
  need(sha(source/'SOURCE_MANIFEST.json')==entry['source_manifest_sha256'],'Pinned source manifest '+stage);need(entry['scientific_originals_included'] is False,'No absent science-byte claim');record=provenance['runs'][stage]
  need(record['scientific_original_zip_opened_during_this_import'] is False and record['scientific_original_zip_included'] is False and record['model_run_performed_during_this_import'] is False,'Import evidence tier '+stage)
  if stage=='C199':
   need(record['evidence_tier']=='PROTOCOL_ONLY' and record['execution_result_available'] is False,'Pending C199');need(not (HERE/'records/c199').exists(),'Do not invent C199 result');continue
  need(record['evidence_tier']=='LOG_SUPPORTED','Logs remain log-tier');log=HERE/record['log_path'];need(sha(log)==record['log_sha256'],'Log identity');s=json.loads((HERE/record['summary_path']).read_text());need(sha(HERE/record['summary_path'])==record['summary_sha256'],'Summary identity');need(extract(log.read_text(),entry['protocol'])==s,'Exact log extraction');need(s['complete'] is True and s['errors']==[] and s['decision']==record['reported_decision'],'Recorded completion');need(s['LC2_closed'] is False and s['LC2_single_fold_passed'] is False,'LC2 stays open');need(re.findall(r'RESULT_SHA256 ([0-9a-f]{64})',log.read_text())[-1]==record['reported_result_zip_sha256'],'Reported result identity');summaries[stage]=s
 a=summaries['C196'];b=summaries['C197'];c=summaries['C198']
 for j in (0,1):
  need(a['native_final_round'+str(j)]['passed']==84,'C196 complete training reported');need(b['native_summaries']['native_final_round'+str(j)]['passed']==84,'C197 reconstructed training reported');need(c['archived_endpoint_summaries'][str(j)]['passed']==84 and c['fresh_endpoint_summaries'][str(j)]['passed']==84,'C198 both paths reported')
 need(b['parent_decision']==a['decision'],'C197 retains C196 decision');need(c['frozen_training_replay_passed'] is True and c['new_training_steps']==0 and c['new_model_forwards']==1513,'Reported frozen replay');need(c['confirmation_states_evaluated']==0 and c['excluded_input_states_evaluated']==0,'Training-only scope')
 print('VERIFIED: delivered source/BUNDLE identities and exact supplied log summaries.')
 print('NOT PERFORMED: scientific result-ZIP rehash, independent gate/gradient/model replay.')
 print('STATUS: C196-C198 log-supported training milestone; C199 pending; LC2 OPEN.')
 return 0
if __name__=='__main__':raise SystemExit(main())
