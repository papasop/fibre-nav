"""Read-only summary output, optional explicit RESULTS ZIP path."""
import sys
from pathlib import Path
from c37_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
from c37_protocol import *
if len(sys.argv)>1:p=Path(sys.argv[1])
else:
    ps=list(Path('/content/gpt2_c37_centre_transfer_a100_v1').glob('session_*/'+RESULT+'.zip'))+list(Path('/content/drive/MyDrive',DRIVE_NAME).glob(RESULT+'_*.zip'))
    require(ps,'未找到C37结果包；请挂载Drive或指定结果路径');p=max(ps,key=lambda x:x.stat().st_mtime_ns)
print('文件：',p,'大小：',p.stat().st_size,'ZIP SHA256：',sha(p))
with zipfile.ZipFile(p) as z:
    m=verify(z);print('成员哈希/CRC读取通过：',len(m))
    if 'SUMMARY.json' in z.namelist():print(json.dumps(zread(z,'SUMMARY.json'),indent=2,ensure_ascii=False))
    else:print('尚未完成；新32态已完成',sum(n.startswith('c35/results/') and '__c37_new' in n and n.endswith('/COMPLETE.json') for n in z.namelist()),'/32')
print('只读检查结束，未修改结果。')
