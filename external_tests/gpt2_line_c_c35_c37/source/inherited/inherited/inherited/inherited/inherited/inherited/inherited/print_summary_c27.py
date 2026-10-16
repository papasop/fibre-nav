"""Read-only C27 result inspection; execute in the run's Colab environment."""
import hashlib,json,zipfile
from pathlib import Path
PROTOCOL='GPT2_C27_STATE_RESIDUAL_COMPARISON_A100_V1'
INPUT_ZIP=''

def inspect(path):
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
  if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe ZIP path')
  ident=json.loads(z.read('identity.json'))
  if ident['protocol']!=PROTOCOL:raise ValueError('Not a C27 result')
  manifest=json.loads(z.read('MANIFEST.json'))
  if set(n for n in names if not n.endswith('/'))!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Manifest coverage differs')
  bad=z.testzip()
  if bad:raise ValueError('CRC failure: '+bad)
  for n,h in manifest.items():
   digest=hashlib.sha256()
   with z.open(n) as f:
    while chunk:=f.read(8<<20):digest.update(chunk)
   if digest.hexdigest()!=h:raise ValueError('Member hash differs: '+n)
  print('文件：',path,'\n大小：',path.stat().st_size,'bytes\nCRC及成员哈希：通过',len(manifest))
  if 'SUMMARY.json' in names:print(json.dumps(json.loads(z.read('SUMMARY.json')),ensure_ascii=False,indent=2))
  else:
   print('尚未完成，无最终 SUMMARY。')
   print(json.dumps({'frozen_arms':[n for n in names if n.startswith('arms/') and n.endswith('/FROZEN.json')],
    'completed_candidates':sum('/candidates/' in n and n.endswith('/COMPLETE.json') for n in names),
    'completed_evaluations':sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names)},ensure_ascii=False,indent=2))
 print('只读检查结束；未修改结果。')

def main():
 if INPUT_ZIP:return inspect(Path(INPUT_ZIP))
 paths=list(Path('/content/gpt2_c27_state_residual_comparison_a100_v1').glob('session_*/gpt2_c27_state_residual_comparison_results.zip'))
 paths+=list(Path('/content/drive/MyDrive/GPT2_C27_STATE_RESIDUAL_COMPARISON_A100_V1_RECOVERY').glob('gpt2_c27_state_residual_comparison_results_*.zip'))
 for p in sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True):
  try:return inspect(p)
  except Exception as e:print('跳过：',p.name,type(e).__name__,str(e))
 raise FileNotFoundError('未找到可完整校验的 C27 结果包；请挂载 Drive 或指定 INPUT_ZIP。')
if __name__=='__main__':main()
