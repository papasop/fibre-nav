"""A100 launcher retaining Colab CUDA Torch and exact original C31 modules."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SEARCH=[ROOT,ROOT/'inherited',ROOT/'inherited/inherited',ROOT/'inherited/inherited/inherited',ROOT/'inherited/inherited/inherited/inherited',ROOT/'inherited/inherited/inherited/inherited/legacy']
sys.path[:0]=list(map(str,SEARCH))
from c34_protocol import *
from c34_parent import probe,prepare,resume_score
from process_io import visible

def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    check_sources(ROOT)
    boot='import sys,runpy;n=int(sys.argv.pop(1));sys.path[:0]=[sys.argv.pop(1) for _ in range(n)];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    def prefix(extra=()):
        paths=list(map(str,SEARCH))+list(map(str,extra))
        return [sys.executable,'-u','-I','-c',boot,str(len(paths)),*paths]
    visible(prefix()+[str(ROOT/'test_c34_math.py')],'C34 fixed-basis/math/archive self-tests',timeout=180)
    visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print('A100_READY',n,torch.__version__,flush=True)"],'A100 device',timeout=60)
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
    base=Path('/content/gpt2_c34_low_dimensional_control_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip:
        paths=list(base.glob('session_*/'+RESULT+'.zip'))
        if backup:paths+=list(Path(backup).glob(RESULT+'_*.zip'))
        good=[]
        for p in paths:
            try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
            except Exception as e:print('跳过不兼容C34恢复包',p.name,str(e),flush=True)
        if good:input_zip=max(good)[2]
    if not input_zip:
        ps=list(Path('/content/gpt2_c31_residual_coverage_a100_v1').glob('session_*/gpt2_c31_residual_coverage_results.zip'))
        if use_drive:ps+=list(Path('/content/drive/MyDrive/GPT2_C31_RESIDUAL_COVERAGE_A100_V1_RECOVERY').glob('gpt2_c31_residual_coverage_results*.zip'))
        for p in sorted(ps,key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:probe(z,ROOT)
                input_zip=str(p);break
            except Exception as e:print('跳过非完整C31',p.name,str(e),flush=True)
    if not input_zip:
        from google.colab import files
        print('请上传完整C31结果ZIP或本版C34恢复包；C32/C33精简包不含模型状态与写入器权重，不能作为输入。',flush=True)
        u=files.upload();require(len(u)==1,'Upload exactly one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
    parent=Path(input_zip)
    with zipfile.ZipFile(parent) as z:
        require('identity.json' in z.namelist(),'请选择结果ZIP，不是源码ZIP');kind=zread(z,'identity.json')['protocol']
    require(kind in (PROTOCOL,PARENT_PROTOCOL),'C34 accepts completed C31 or matching C34 only')
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    result=session/(RESULT+'.zip')
    try:
        if kind==PROTOCOL:
            score=resume_score(parent,ROOT)
            with zipfile.ZipFile(parent) as z:z.extractall(out)
            print('恢复C34实际几何、补偿基和端点记录：',score,flush=True)
        else:
            dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT)})
            local=session/'c31_readonly.zip';shutil.copyfile(parent,local)
            try:require(sha(local)==sha(parent),'Read-only parent copy differs');prepare(local,out,ROOT)
            finally:
                if local.exists():local.unlink()
        if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        print('明确输入：',parent,flush=True)
        print('C34：12个固定C31训练态构造每路由固定1维/6维补偿基；另36训练态+已检查旧16态作机制比较。',flush=True)
        print('离线Jacobian/SVD；系数使用实测原始残差；每臂仅一次补偿、范数上限0.05，无回溯/迭代修正。',flush=True)
        print('这是带真实残差的oracle补偿实验，不是无在线校正写入；不训练系数模型，不闭合LC1。',flush=True)
        print('阈值0.002/0.005/0.05不变；固定64态不重抽，所有失败保留；A100 float32关闭TF32。',flush=True)
        print('无总会话时限；按几何点/4个评估态备份，只清理本C34最近3份以外的已校验备份。',flush=True)
        check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
        deps=base/'deps'
        for d in (Path('/content/gpt2_c31_residual_coverage_a100_v1/deps'),Path('/content/gpt2_c29_frozen_direction_diagnosis_a100_v1/deps')):
            if d.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(d)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=d;break
        deps.mkdir(parents=True,exist_ok=True)
        if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
            visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
            visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后依赖检查',timeout=90)
        visible(prefix([deps])+[str(ROOT/'test_c34_torch.py')],'C34 actual tensor/cache/finite-gate pipeline self-tests',timeout=240)
        snapshot(out,backup)
        visible(prefix([deps])+[str(ROOT/'worker_c34.py'),'--out',str(out),'--drive-dir',backup],'C34 A100 fixed-basis oracle comparison',log_path=session/'console.log')
    finally:
        if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
        result=snapshot(out,backup);print('唯一结果总包：',result,flush=True)
    return result

if __name__=='__main__':start()
