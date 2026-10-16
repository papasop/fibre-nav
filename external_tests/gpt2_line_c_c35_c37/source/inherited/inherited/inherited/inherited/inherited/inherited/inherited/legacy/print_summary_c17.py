"""Read-only result summary, CRC and member digests; no model or tensor execution."""
import argparse,hashlib,json,zipfile
from pathlib import Path

def inspect(path):
 p=Path(path);print('文件：',p);print('大小：',p.stat().st_size,'bytes')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 print('ZIP SHA256：',h.hexdigest())
 with zipfile.ZipFile(p) as z:
  names=z.namelist()
  if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n for n in names):raise ValueError('Unsafe/duplicate archive members')
  if z.testzip():raise ValueError('ZIP CRC failure')
  manifest=json.loads(z.read('MANIFEST.json'))
  if set(n for n in names if not n.endswith('/'))!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Manifest coverage mismatch')
  for n,expected in manifest.items():
   hh=hashlib.sha256()
   with z.open(n) as f:
    for b in iter(lambda:f.read(8<<20),b''):hh.update(b)
   if hh.hexdigest()!=expected:raise ValueError('Member digest mismatch: '+n)
  print('ZIP CRC及',len(manifest),'个成员哈希：通过')
  for n in ['SUMMARY.json','WRITER_FROZEN.json','heads/guarded/PROGRESS.json']:
   if n in names:print(n+'\n'+json.dumps(json.loads(z.read(n)),indent=2,ensure_ascii=False))
  if 'SUMMARY.json' not in names:print('尚未完成；没有最终 SUMMARY。')
 print('只读检查结束；未修改存档。')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('zip');inspect(ap.parse_args().zip)
