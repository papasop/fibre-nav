"""A100 Colab launcher with root-only imports, strict parents and progress-aware resume."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from c37_paths import configure
SEARCH=configure(ROOT)
from c37_protocol import *
from c37_parent import probe,prepare,resume_score
from process_io import visible

def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    check_sources(ROOT)
    boot='import sys,runpy;n=int(sys.argv.pop(1));sys.path[:0]=[sys.argv.pop(1) for _ in range(n)];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    def prefix(extra=()):
        ps=list(map(str,SEARCH))+list(map(str,extra));return [sys.executable,'-u','-I','-c',boot,str(len(ps)),*ps]
    visible(prefix()+[str(ROOT/'test_c37_protocol.py')],'C37 source/archive/prospective-split tests',timeout=180)
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
    base=Path('/content/gpt2_c37_centre_transfer_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip:
        paths=list(base.glob('session_*/'+RESULT+'.zip'))
        if backup:paths+=list(Path(backup).glob(RESULT+'_*.zip'))
        good=[]
        for p in paths:
            try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
            except Exception as e:print('跳过不兼容C37恢复包',p.name,str(e),flush=True)
        if good:input_zip=max(good)[2]
    if not input_zip:
        paths=list(Path('/content/gpt2_c36_frozen_local_replication_a100_v1').glob('session_*/gpt2_c36_frozen_local_replication_results.zip'))
        if use_drive:paths+=list(Path('/content/drive/MyDrive/GPT2_C36_FROZEN_LOCAL_REPLICATION_A100_V1_RECOVERY').glob('gpt2_c36_frozen_local_replication_results*.zip'))
        for p in sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:probe(z,ROOT)
                input_zip=str(p);break
            except Exception as e:print('跳过非本次完整C36 FIX1',p.name,str(e),flush=True)
    if not input_zip:
        from google.colab import files
        print('上传已完成C36 FIX1结果ZIP，或本版C37续跑结果ZIP；不要上传源码ZIP。',flush=True)
        u=files.upload();require(len(u)==1,'Upload exactly one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
    parent=Path(input_zip)
    with zipfile.ZipFile(parent) as z:
        require('identity.json' in z.namelist(),'需要结果ZIP，不是源码ZIP');kind=zread(z,'identity.json')['protocol']
    require(kind in (PROTOCOL,PARENT_PROTOCOL),'Only complete C36 FIX1 or matching C37 accepted')
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True);result=session/(RESULT+'.zip')
    try:
        if kind==PROTOCOL:
            score=resume_score(parent,ROOT)
            with zipfile.ZipFile(parent) as z:z.extractall(out)
            print('恢复C37真实冻结权重、已封存提案与验收记录：',score,flush=True)
        else:
            dump(out/'identity.json',identity(ROOT))
            local=session/'c36_readonly.zip';shutil.copyfile(parent,local)
            try:require(sha(local)==sha(parent),'Read-only C36 copy differs');prepare(local,out,ROOT)
            finally:
                if local.exists():local.unlink()
        if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        if (out/'NEW_STATE_SPECS.json').exists():require(read(out/'NEW_STATE_SPECS.json')==specs(),'C37 prospectively declared seeds differ')
        else:dump(out/'NEW_STATE_SPECS.json',specs())
        print('明确输入：',parent,flush=True)
        print('C37：核对已完成C36归档、实际权重与32态记录；不训练、不重选头。',flush=True)
        print('每类固定C36 new00..03准备态作中心；16个已检查中心，每个生成2个新扰动态，共32态。',flush=True)
        print('中心曾检查但未用于写入器训练；不是未见输入类或独立模型测试。新种子固定，不重抽。',flush=True)
        print('准备态允许外部Jacobian/SVD和纠正；在线写入每臂一个冻结组合提案，无求导/SVD/校正/回溯/教师。',flush=True)
        print('预算0.002/0.005/0.05不变；无总会话时限；每4态备份，仅保留本C37最近3份已校验备份。',flush=True)
        deps=base/'deps';deps.mkdir(parents=True,exist_ok=True)
        check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
        for d in (Path('/content/gpt2_c36_frozen_local_replication_a100_v1/deps'),Path('/content/gpt2_c35_prewrite_coefficient_a100_v1/deps')):
            if d.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(d)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=d;break
        if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
            visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
            visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后依赖检查',timeout=90)
        print('C37 合成自测有阶段/逐项日志，不设300秒限时。',flush=True)
        visible(prefix([deps])+[str(ROOT/'test_c37_torch.py')],'C37 archive/tensor/seal/resume tests',timeout=None)
        snapshot(out,backup)
        visible(prefix([deps])+[str(ROOT/'worker_c37.py'),'--out',str(out),'--drive-dir',backup],'C37 A100 fixed-centre transfer',log_path=session/'console.log')
    finally:
        if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
        result=snapshot(out,backup);print('唯一结果总包：',result,flush=True)
    return result
if __name__=='__main__':start()
