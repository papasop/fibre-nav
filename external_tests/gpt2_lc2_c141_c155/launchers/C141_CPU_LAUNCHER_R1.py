# C141 R1 CPU启动器：科学源码放在配套BUNDLE，不嵌入本文件。
# 全文粘贴到Colab；提示时只上传一份BUNDLE.zip。
SOURCE_ZIP = ""
USE_GOOGLE_DRIVE = True
AUTO_DOWNLOAD = True
C132_ZIP = ""
C133_ZIP = ""
C134_ZIP = ""
C135_ZIP = ""
C136_ZIP = ""
C137_ZIP = ""
C138_ZIP = ""
C139_ZIP = ""
C140_ZIP = ""

from pathlib import Path, PurePosixPath
import hashlib,importlib,json,sys,tempfile,zipfile
SOURCE_NAME = "C141_CPU_LAUNCHER_R1_BUNDLE.zip"
SOURCE_SHA256 = "a5270110ea6fd1b58ed8bff7af046d2a805783077b282f5357bd40ab96421957"

def c141_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def c141_choose():
    if SOURCE_ZIP:
        p=Path(SOURCE_ZIP).expanduser().resolve()
        if not p.is_file() or c141_sha(p)!=SOURCE_SHA256:
            raise ValueError('SOURCE_ZIP必须是配套C141 BUNDLE.zip，不能选择旧源码、bridge或科学结果ZIP。')
        return p
    candidates=[]
    for folder in [Path('/content'),Path.cwd()]:
        if folder.is_dir():candidates+=sorted(folder.glob('C141_CPU_LAUNCHER_R1_BUNDLE*.zip'))
    for p in candidates:
        if p.is_file() and c141_sha(p)==SOURCE_SHA256:return p.resolve()
    try:from google.colab import files
    except ImportError as e:raise FileNotFoundError('请填写SOURCE_ZIP为配套BUNDLE.zip的绝对路径。') from e
    print('请上传一份 '+SOURCE_NAME+'；无需解压，不要选择原科学结果ZIP。',flush=True)
    uploaded=files.upload()
    matched=[Path(n).resolve() for n in uploaded if Path(n).is_file() and c141_sha(n)==SOURCE_SHA256]
    if len(matched)!=1:raise ValueError('需要且只需要一个哈希匹配的C141 BUNDLE.zip。')
    return matched[0]

def start():
    source=c141_choose();root=Path(tempfile.mkdtemp(prefix='c141_cpu_source_'))
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
    print('C141 R1 CPU：配套BUNDLE已校验；完整合成测试上限40分钟、模型工作进程仍为50分钟；真实GPT2 CPU；冻结C140第12步；48个新共享核局部态确认；无训练；无需GPU。',flush=True)
    oldpath=sys.path[:];names=('audit124', 'core133', 'core134', 'core135', 'core136', 'core137', 'core138', 'core139', 'core140', 'core141', 'fixture133', 'fixture134', 'fixture135', 'fixture136', 'fixture137', 'fixture138', 'fixture139', 'fixture140', 'fixture141', 'functional136', 'functional137', 'functional140', 'inputs134', 'inputs135', 'inputs136', 'inputs137', 'inputs138', 'inputs139', 'inputs140', 'inputs141', 'launch133', 'launch134', 'launch135', 'launch136', 'launch137', 'launch138', 'launch139', 'launch140', 'launch141', 'lc2_math', 'lc2_protocol', 'reader133', 'run_l5_audit', 'runtime134', 'runtime138', 'runtime139', 'runtime140', 'runtime141', 'runtime_base123', 'test133', 'test134', 'test135', 'test136', 'test137', 'test138', 'test139', 'test140', 'test141', 'torch_smoke136', 'torch_smoke137', 'torch_smoke140', 'worker133', 'worker134', 'worker135', 'worker136', 'worker137', 'worker138', 'worker139', 'worker140', 'worker141', 'writer_core');saved={k:sys.modules.pop(k) for k in names if k in sys.modules}
    try:
        sys.path.insert(0,str(root));launch=importlib.import_module('launch141')
        return launch.start(USE_GOOGLE_DRIVE,dict(C132=C132_ZIP,C133=C133_ZIP,C134=C134_ZIP,C135=C135_ZIP,C136=C136_ZIP,C137=C137_ZIP,C138=C138_ZIP,C139=C139_ZIP,C140=C140_ZIP),AUTO_DOWNLOAD)
    finally:
        sys.path[:]=oldpath
        for k in names:sys.modules.pop(k,None)
        sys.modules.update(saved)

if __name__=='__main__':C141_RESULT_ZIP=start()
