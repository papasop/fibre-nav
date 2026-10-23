# C154 CPU启动器：科学源码放在配套BUNDLE，不嵌入本文件。
# 全文粘贴到Colab；提示时只上传一份BUNDLE.zip。
SOURCE_ZIP = ""
USE_GOOGLE_DRIVE = True
AUTO_DOWNLOAD = True
C143_ZIP = ""
C145_ZIP = ""
C146_ZIP = ""
C147_ZIP = ""
C148_ZIP = ""
C149_ZIP = ""
C150_ZIP = ""
C152_ZIP = ""
C153_ZIP = ""

from pathlib import Path, PurePosixPath
import hashlib,importlib,json,sys,tempfile,zipfile
SOURCE_NAME = "C154_CPU_LAUNCHER_BUNDLE.zip"
SOURCE_SHA256 = "8de8d31436da644e5c739785377cd00cb38d7f32212e6573bb424e7953e81189"

def c154_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def c154_choose():
    if SOURCE_ZIP:
        p=Path(SOURCE_ZIP).expanduser().resolve()
        if not p.is_file() or c154_sha(p)!=SOURCE_SHA256:
            raise ValueError('SOURCE_ZIP必须是配套C154 BUNDLE.zip，不能选择旧源码、bridge或科学结果ZIP。')
        return p
    candidates=[]
    for folder in [Path('/content'),Path.cwd()]:
        if folder.is_dir():candidates+=sorted(folder.glob('C154_CPU_LAUNCHER_BUNDLE*.zip'))
    for p in candidates:
        if p.is_file() and c154_sha(p)==SOURCE_SHA256:return p.resolve()
    try:from google.colab import files
    except ImportError as e:raise FileNotFoundError('请填写SOURCE_ZIP为配套BUNDLE.zip的绝对路径。') from e
    print('请上传一份 '+SOURCE_NAME+'；无需解压，不要选择原科学结果ZIP。',flush=True)
    uploaded=files.upload()
    matched=[Path(n).resolve() for n in uploaded if Path(n).is_file() and c154_sha(n)==SOURCE_SHA256]
    if len(matched)!=1:raise ValueError('需要且只需要一个哈希匹配的C154 BUNDLE.zip。')
    return matched[0]

def start():
    source=c154_choose();root=Path(tempfile.mkdtemp(prefix='c154_cpu_source_'))
    with zipfile.ZipFile(source) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('BUNDLE有重复成员')
        for n in names:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n:raise ValueError('非法BUNDLE路径')
        if sum(i.file_size for i in z.infolist())>10*1024**2:raise ValueError('BUNDLE展开体积异常')
        manifest=json.loads(z.read('SOURCE_MANIFEST.json'))
        if set(names)!=set(manifest)|{'SOURCE_MANIFEST.json'}:raise ValueError('BUNDLE目录与清单不一致')
        for n,h in manifest.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('源码成员哈希不匹配：'+n)
        z.extractall(root)
    print('C154 CPU：配套BUNDLE已校验；真实GPT-2固定方向诊断，全部84态、1512次指标测量。无反传或训练；工作上限120分钟，自测另计；不计LC2通过。',flush=True)
    oldpath=sys.path[:];names=sorted({PurePosixPath(n).stem for n in manifest if n.endswith('.py') and PurePosixPath(n).stem!='__init__'});saved={k:sys.modules.pop(k) for k in names if k in sys.modules}
    try:
        sys.path.insert(0,str(root));launch=importlib.import_module('launch154')
        return launch.start(USE_GOOGLE_DRIVE,dict(C153=C153_ZIP,C152=C152_ZIP,C150=C150_ZIP,C149=C149_ZIP,C148=C148_ZIP,C147=C147_ZIP,C146=C146_ZIP,C145=C145_ZIP,C143=C143_ZIP),AUTO_DOWNLOAD)
    finally:
        sys.path[:]=oldpath
        for k in names:sys.modules.pop(k,None)
        sys.modules.update(saved)

if __name__=='__main__':C154_RESULT_ZIP=start()
