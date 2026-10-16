from pathlib import Path
import json,zipfile,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'inherited'),str(ROOT/'inherited/legacy')]
from c28_protocol import RESULT,PROTOCOL,verify_archive,read_json

def main():
 pp=list(Path('/content/gpt2_c28_response_aware_four_route_a100_v1').glob('session_*/'+RESULT+'.zip'))+list(Path('/content/drive/MyDrive/'+PROTOCOL+'_RECOVERY').glob(RESULT+'_*.zip'))
 for path in sorted(pp,key=lambda p:p.stat().st_mtime_ns,reverse=True):
  try:
   with zipfile.ZipFile(path) as z:
    require=read_json(z,'identity.json')['protocol']==PROTOCOL
    if not require:continue
    manifest=verify_archive(z);names=z.namelist();print('文件：',path,'\n大小：',path.stat().st_size,'\nCRC和成员哈希：通过',len(manifest))
    if 'SUMMARY.json' in names:
     s=read_json(z,'SUMMARY.json');keys=('decision','complete','training','inspected_development','by_case','parent_C27_unchanged','all76_training_passes_preserved','offline_geometry','online_correction_steps','online_jacobian_builds','fresh_evaluation','LC1_closed');print(json.dumps({k:s[k] for k in keys},ensure_ascii=False,indent=2))
    else:print('未完成，无最终SUMMARY。已保存几何：',sum(n.startswith('geometry/') and n.endswith('.pt') for n in names),'；已完成候选：',sum('/candidates/' in n and n.endswith('/COMPLETE.json') for n in names))
    print('只读检查结束，未修改存档。');return
  except Exception as e:print('跳过',path.name,type(e).__name__,str(e))
 raise FileNotFoundError('未找到可校验C28结果包；请挂载Drive。')
if __name__=='__main__':main()
