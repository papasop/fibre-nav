# C148 CPU启动器：科学源码放在配套BUNDLE，不嵌入本文件。
# 全文粘贴到Colab；提示时只上传一份BUNDLE.zip。
SOURCE_ZIP = ""
USE_GOOGLE_DRIVE = True
AUTO_DOWNLOAD = True
C143_ZIP = ""
C145_ZIP = ""
C146_ZIP = ""
C147_ZIP = ""

from pathlib import Path, PurePosixPath
import hashlib,importlib,json,sys,tempfile,zipfile
SOURCE_NAME = "C148_CPU_LAUNCHER_BUNDLE.zip"
SOURCE_SHA256 = "c6218bb6a3567ad6b160108651a406e08514b5f6f4ced9b820ace0900814d50f"

def c148_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def c148_choose():
    if SOURCE_ZIP:
        p=Path(SOURCE_ZIP).expanduser().resolve()
        if not p.is_file() or c148_sha(p)!=SOURCE_SHA256:
            raise ValueError('SOURCE_ZIP必须是配套C148 BUNDLE.zip，不能选择旧源码、bridge或科学结果ZIP。')
        return p
    candidates=[]
    for folder in [Path('/content'),Path.cwd()]:
        if folder.is_dir():candidates+=sorted(folder.glob('C148_CPU_LAUNCHER_BUNDLE*.zip'))
    for p in candidates:
        if p.is_file() and c148_sha(p)==SOURCE_SHA256:return p.resolve()
    try:from google.colab import files
    except ImportError as e:raise FileNotFoundError('请填写SOURCE_ZIP为配套BUNDLE.zip的绝对路径。') from e
    print('请上传一份 '+SOURCE_NAME+'；无需解压，不要选择原科学结果ZIP。',flush=True)
    uploaded=files.upload()
    matched=[Path(n).resolve() for n in uploaded if Path(n).is_file() and c148_sha(n)==SOURCE_SHA256]
    if len(matched)!=1:raise ValueError('需要且只需要一个哈希匹配的C148 BUNDLE.zip。')
    return matched[0]

def start():
    source=c148_choose();root=Path(tempfile.mkdtemp(prefix='c148_cpu_source_'))
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
    print('C148 CPU：配套BUNDLE已校验；原C146头初始化，全局共享与3局部头软路由，各固定12步，全84训练态。真实GPT2 CPU；模型工作上限240分钟，自测另计。',flush=True)
    oldpath=sys.path[:];names=('audit124', 'audit142', 'audit144', 'audit147', 'core133', 'core134', 'core135', 'core136', 'core137', 'core138', 'core139', 'core140', 'core141', 'core142', 'core143', 'core144', 'core145', 'core146', 'core147', 'core148', 'fixture133', 'fixture134', 'fixture135', 'fixture136', 'fixture137', 'fixture138', 'fixture139', 'fixture140', 'fixture141', 'fixture142', 'fixture143', 'fixture144', 'fixture145', 'fixture146', 'fixture147', 'fixture148', 'functional136', 'functional137', 'functional140', 'functional143', 'functional145', 'functional146', 'functional148', 'inputs134', 'inputs135', 'inputs136', 'inputs137', 'inputs138', 'inputs139', 'inputs140', 'inputs141', 'inputs143', 'inputs145', 'inputs146', 'inputs148', 'launch133', 'launch134', 'launch135', 'launch136', 'launch137', 'launch138', 'launch139', 'launch140', 'launch141', 'launch142', 'launch143', 'launch144', 'launch145', 'launch146', 'launch147', 'launch148', 'lc2_math', 'lc2_protocol', 'reader133', 'run_l5_audit', 'runtime134', 'runtime138', 'runtime139', 'runtime140', 'runtime141', 'runtime143', 'runtime145', 'runtime146', 'runtime148', 'runtime_base123', 'test133', 'test134', 'test135', 'test136', 'test137', 'test138', 'test139', 'test140', 'test141', 'test142', 'test143', 'test144', 'test145', 'test146', 'test147', 'test148', 'test_bootstrap145', 'test_bootstrap146', 'test_bootstrap148', 'test_inputs148', 'torch_smoke136', 'torch_smoke137', 'torch_smoke140', 'torch_smoke143', 'torch_smoke145', 'torch_smoke146', 'torch_smoke148', 'worker133', 'worker134', 'worker135', 'worker136', 'worker137', 'worker138', 'worker139', 'worker140', 'worker141', 'worker142', 'worker143', 'worker144', 'worker145', 'worker146', 'worker147', 'worker148', 'writer_core');saved={k:sys.modules.pop(k) for k in names if k in sys.modules}
    try:
        sys.path.insert(0,str(root));launch=importlib.import_module('launch148')
        return launch.start(USE_GOOGLE_DRIVE,dict(C147=C147_ZIP,C146=C146_ZIP,C145=C145_ZIP,C143=C143_ZIP),AUTO_DOWNLOAD)
    finally:
        sys.path[:]=oldpath
        for k in names:sys.modules.pop(k,None)
        sys.modules.update(saved)

if __name__=='__main__':C148_RESULT_ZIP=start()
