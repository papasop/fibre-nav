"""A100 Colab launcher with root-only imports, strict parents and progress-aware resume."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from c36_paths import configure
SEARCH=configure(ROOT)
from c36_protocol import *
from c36_parent import probe,prepare,resume_score
from process_io import visible

def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    check_sources(ROOT)
    boot='import sys,runpy;n=int(sys.argv.pop(1));sys.path[:0]=[sys.argv.pop(1) for _ in range(n)];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    def prefix(extra=()):
        ps=list(map(str,SEARCH))+list(map(str,extra));return [sys.executable,'-u','-I','-c',boot,str(len(ps)),*ps]
    visible(prefix()+[str(ROOT/'test_c36_protocol.py')],'C36 source/archive/prospective-split tests',timeout=180)
    visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available(),'A100 GPU required';n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print('A100_READY',n,torch.__version__,flush=True)"],'A100 device',timeout=60)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        require(not any(Path(x).name.startswith('worker_') and Path(x).suffix=='.py' and ('/gpt2_' in ' '.join(argv) or '/qwen_' in ' '.join(argv)) for x in argv),'Another model worker is active: '+p.name)
    backup=''
    if use_drive:
        from google.colab import drive
        if not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
        backup='/content/drive/MyDrive/'+DRIVE_NAME
    base=Path('/content/gpt2_c36_frozen_local_replication_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip:
        paths=list(base.glob('session_*/'+RESULT+'.zip'))
        if backup:paths+=list(Path(backup).glob(RESULT+'_*.zip'))
        good=[]
        for p in paths:
            try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
            except Exception as e:print('跳过不兼容C36恢复包',p.name,str(e),flush=True)
        if good:input_zip=max(good)[2]
    if not input_zip:
        paths=list(Path('/content/gpt2_c35_prewrite_coefficient_a100_v1').glob('session_*/gpt2_c35_prewrite_coefficient_results.zip'))
        if use_drive:paths+=list(Path('/content/drive/MyDrive/GPT2_C35_PREWRITE_COEFFICIENT_A100_V1_RECOVERY').glob('gpt2_c35_prewrite_coefficient_results*.zip'))
        for p in sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:probe(z,ROOT)
                input_zip=str(p);break
            except Exception as e:print('跳过非本次完整C35',p.name,str(e),flush=True)
    if not input_zip:
        from google.colab import files
        print('上传已完成C35结果ZIP，或本版C36续跑结果ZIP；不要上传源码ZIP。',flush=True)
        u=files.upload();require(len(u)==1,'Upload exactly one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
    parent=Path(input_zip)
    with zipfile.ZipFile(parent) as z:
        require('identity.json' in z.namelist(),'需要结果ZIP，不是源码ZIP');kind=zread(z,'identity.json')['protocol']
    require(kind in (PROTOCOL,PARENT_PROTOCOL),'Only C35 complete results or matching C36 accepted')
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True);result=session/(RESULT+'.zip')
    try:
        if kind==PROTOCOL:
            score=resume_score(parent,ROOT)
            with zipfile.ZipFile(parent) as z:z.extractall(out)
            print('恢复C36真实冻结权重、已封存提案与验收记录：',score,flush=True)
        else:
            dump(out/'identity.json',identity(ROOT))
            local=session/'c35_readonly.zip';shutil.copyfile(parent,local)
            try:require(sha(local)==sha(parent),'Read-only C35 copy differs');prepare(local,out,ROOT)
            finally:
                if local.exists():local.unlink()
        if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        if (out/'NEW_STATE_SPECS.json').exists():require(read(out/'NEW_STATE_SPECS.json')==specs(),'C36 prospectively declared seeds differ')
        else:dump(out/'NEW_STATE_SPECS.json',specs())
        print('明确输入：',parent,flush=True)
        print('C36：不训练、不重选头；核对C35全部80态存档、复现80个写前提案、实际重放8态端点。',flush=True)
        print('固定几何/学习校准平行报告；冻结后按固定种子生成新32态，每类8态；不重抽、不删失败态。',flush=True)
        print('每臂写前一次组合提案并封存，外部提交/前向验收；在线无Jacobian/SVD/校正/回溯/教师。',flush=True)
        print('离线几何、准备态计算、外部运行时仍存在；同模型/图的额外局部复验，不是独立模型确认或原生更新。',flush=True)
        print('阈值0.002/0.005/0.05保持；无总会话时限；每4新态备份，仅保留本C36最近3份已校验备份。',flush=True)
        deps=base/'deps';deps.mkdir(parents=True,exist_ok=True)
        check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
        for d in (Path('/content/gpt2_c35_prewrite_coefficient_a100_v1/deps'),Path('/content/gpt2_c34_low_dimensional_control_a100_v1/deps')):
            if d.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(d)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=d;break
        if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
            visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
            visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后依赖检查',timeout=90)
        print('C36 FIX1：合成流水线自测显示分阶段进度，不设300秒超时；实际模型权重、种子和验收规则不变。',flush=True)
        visible(prefix([deps])+[str(ROOT/'test_c36_torch.py')],'C36 archive/tensor/seal/resume tests',timeout=None)
        snapshot(out,backup)
        visible(prefix([deps])+[str(ROOT/'worker_c36.py'),'--out',str(out),'--drive-dir',backup],'C36 A100 frozen local replication',log_path=session/'console.log')
    finally:
        if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
        result=snapshot(out,backup);print('唯一结果总包：',result,flush=True)
    return result
if __name__=='__main__':start()
