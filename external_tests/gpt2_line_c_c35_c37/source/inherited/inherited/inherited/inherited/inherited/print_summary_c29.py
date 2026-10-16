from pathlib import Path
import sys,json,zipfile
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'inherited'),str(ROOT/'inherited/inherited'),str(ROOT/'inherited/inherited/legacy')]
from c29_protocol import *

def main():
 paths=list(Path('/content/gpt2_c29_frozen_direction_diagnosis_a100_v1').glob('session_*/'+RESULT+'.zip'))+list(Path('/content/drive/MyDrive/'+DRIVE_NAME).glob(RESULT+'_*.zip'))
 for p in sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True):
  try:
   with zipfile.ZipFile(p) as z:
    require(read_json(z,'identity.json')['protocol']==PROTOCOL,'Wrong protocol');m=verify_archive(z);print('文件：',p,'\n大小：',p.stat().st_size,'bytes\nCRC和成员哈希：通过',len(m))
    if 'SUMMARY.json' in z.namelist():
     s=read_json(z,'SUMMARY.json');keys=('decision','complete','endpoint_replay','direction_diagnosis','by_case','all_writers_unchanged','training_Jacobians_reused','committed_diagnostic_work','new_head_training_steps','new_candidates','new_heads_committed','fresh_evaluation','LC1_closed');print(json.dumps({k:s[k] for k in keys},ensure_ascii=False,indent=2))
    else:print('尚未完成，无最终SUMMARY；已完成诊断：',sum(n.startswith('diagnosis/') and n.endswith('/COMPLETE.json') for n in z.namelist()))
    print('只读检查结束；未修改存档。');return
  except Exception as e:print('跳过',p.name,type(e).__name__,str(e))
 raise FileNotFoundError('未找到C29结果包，请挂载Drive。')
if __name__=='__main__':main()
