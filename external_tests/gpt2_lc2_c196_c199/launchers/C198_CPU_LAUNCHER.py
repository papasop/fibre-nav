"""C198 CPU launcher: one matching BUNDLE upload; no embedded scientific source."""
from pathlib import Path,PurePosixPath
import hashlib,importlib,json,stat,sys,tempfile,zipfile
USE_GOOGLE_DRIVE=True
AUTO_DOWNLOAD=True
BUNDLE_ZIP=''
# Optional original scientific ZIP paths; otherwise use the pinned Drive paths.
INPUT_PATHS={'C'+str(i):'' for i in range(155,198)}
BUNDLE_NAME='C198_CPU_LAUNCHER_BUNDLE.zip'
BUNDLE_SHA256='3a684e93121462c61cfc011e295e52a627b6b95363c2ae0e8be691ce3bb50e6e'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def choose_bundle():
 if BUNDLE_ZIP:
  p=Path(BUNDLE_ZIP).expanduser().resolve()
  if p.is_file() and digest(p)==BUNDLE_SHA256:return p
  raise ValueError('BUNDLE_ZIP不是本启动器匹配的C198配套BUNDLE。')
 candidates=[Path('/content')/BUNDLE_NAME,Path.cwd()/BUNDLE_NAME]
 if Path('/content').is_dir():candidates+=sorted(Path('/content').glob('C198*.zip'))
 for p in candidates:
  if p.is_file() and digest(p)==BUNDLE_SHA256:return p
 try:from google.colab import files
 except ImportError as e:raise FileNotFoundError('请填写BUNDLE_ZIP本地绝对路径。') from e
 print('请一次上传 C198_CPU_LAUNCHER_BUNDLE.zip；不要上传科学结果ZIP，无需解压。',flush=True)
 uploaded=files.upload();matched=[Path(n).resolve() for n in uploaded if Path(n).is_file() and digest(n)==BUNDLE_SHA256]
 if len(uploaded)!=1 or len(matched)!=1:raise ValueError('需要且仅需要一个哈希匹配的C198 BUNDLE；科研结果随后从Drive读取。')
 return matched[0]
def extract_bundle(path):
 if digest(path)!=BUNDLE_SHA256:raise ValueError('配套BUNDLE哈希不匹配。')
 root=Path(tempfile.mkdtemp(prefix='c198_trusted_source_'))
 with zipfile.ZipFile(path) as z:
  items=z.infolist();names=[i.filename for i in items]
  if len(names)!=len(set(names)) or len(items)>1700 or sum(i.file_size for i in items)>40*1024**2:raise ValueError('BUNDLE目录或大小不合法。')
  for i in items:
   p=PurePosixPath(i.filename)
   if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or p.as_posix()!=i.filename or i.is_dir() or stat.S_ISLNK(i.external_attr>>16):raise ValueError('BUNDLE不安全路径。')
  m=json.loads(z.read('SOURCE_MANIFEST.json'))
  if set(names)!=set(m)|{'SOURCE_MANIFEST.json'}:raise ValueError('BUNDLE清单不一致。')
  for n,h in m.items():
   b=z.read(n)
   if hashlib.sha256(b).hexdigest()!=h:raise ValueError('源码文件哈希不匹配：'+n)
   p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  (root/'SOURCE_MANIFEST.json').write_bytes(z.read('SOURCE_MANIFEST.json'))
 return root,m
def start():
 root,manifest=extract_bundle(choose_bundle())
 print('C198 CPU：配套BUNDLE已校验；固定C196最终头和43原包；84训练态两轮重新采集特征/路由/提案，并分开重放原端点与新提案。1513次CPU前向，无反传/训练/修复/选头；工作上限240分钟，自测另计；不计LC2通过。',flush=True)
 oldpath=sys.path[:];names={PurePosixPath(n).stem for n in manifest if n.endswith('.py') and PurePosixPath(n).stem!='__init__'};saved={k:sys.modules.pop(k) for k in names if k in sys.modules}
 try:
  sys.path.insert(0,str(root));launch=importlib.import_module('launch198')
  return launch.start(USE_GOOGLE_DRIVE,dict(INPUT_PATHS),AUTO_DOWNLOAD)
 finally:
  sys.path[:]=oldpath
  for name in names:sys.modules.pop(name,None)
  sys.modules.update(saved)
if __name__=='__main__':C198_RESULT_ZIP=start()
