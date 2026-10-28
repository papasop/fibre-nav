"""C199 CPU launcher: one matching BUNDLE upload; no embedded scientific source."""
from pathlib import Path,PurePosixPath
import hashlib,importlib,json,stat,sys,tempfile,zipfile
USE_GOOGLE_DRIVE=True
AUTO_DOWNLOAD=True
BUNDLE_ZIP=''
# Optional original scientific ZIP paths; otherwise use the pinned Drive paths.
INPUT_PATHS={'C'+str(i):'' for i in range(155,199)}
BUNDLE_NAME='C199_CPU_LAUNCHER_BUNDLE.zip'
BUNDLE_SHA256='e6904011989c7b482802962fbd67560f576c17a229d59d01afd175bc3252ab9b'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def choose_bundle():
 if BUNDLE_ZIP:
  p=Path(BUNDLE_ZIP).expanduser().resolve()
  if p.is_file() and digest(p)==BUNDLE_SHA256:return p
  raise ValueError('BUNDLE_ZIP不是本启动器匹配的C199配套BUNDLE。')
 candidates=[Path('/content')/BUNDLE_NAME,Path.cwd()/BUNDLE_NAME]
 if Path('/content').is_dir():candidates+=sorted(Path('/content').glob('C199*.zip'))
 for p in candidates:
  if p.is_file() and digest(p)==BUNDLE_SHA256:return p
 try:from google.colab import files
 except ImportError as e:raise FileNotFoundError('请填写BUNDLE_ZIP本地绝对路径。') from e
 print('请一次上传 C199_CPU_LAUNCHER_BUNDLE.zip；不要上传科学结果ZIP，无需解压。',flush=True)
 uploaded=files.upload();matched=[Path(n).resolve() for n in uploaded if Path(n).is_file() and digest(n)==BUNDLE_SHA256]
 if len(uploaded)!=1 or len(matched)!=1:raise ValueError('需要且仅需要一个哈希匹配的C199 BUNDLE；科研结果随后从Drive读取。')
 return matched[0]
def extract_bundle(path):
 if digest(path)!=BUNDLE_SHA256:raise ValueError('配套BUNDLE哈希不匹配。')
 root=Path(tempfile.mkdtemp(prefix='c199_trusted_source_'))
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
 print('C199 CPU：配套BUNDLE已校验；固定44原包和C196最终头；先84态已知控制，再测已注册48个新方向态，三个已训练类各16态、完整两轮。最多1429次CPU前向，无训练/修复/选头；工作上限240分钟，自测另计；不是排除输入测试，不计LC2闭合。',flush=True)
 oldpath=sys.path[:];names={PurePosixPath(n).stem for n in manifest if n.endswith('.py') and PurePosixPath(n).stem!='__init__'};saved={k:sys.modules.pop(k) for k in names if k in sys.modules}
 try:
  sys.path.insert(0,str(root));launch=importlib.import_module('launch199')
  return launch.start(USE_GOOGLE_DRIVE,dict(INPUT_PATHS),AUTO_DOWNLOAD)
 finally:
  sys.path[:]=oldpath
  for name in names:sys.modules.pop(name,None)
  sys.modules.update(saved)
if __name__=='__main__':C199_RESULT_ZIP=start()
