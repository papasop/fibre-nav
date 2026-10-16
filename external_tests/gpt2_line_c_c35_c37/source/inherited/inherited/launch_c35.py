"""Bootstrap inherited paths before all protocol imports."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from c35_paths import configure
SEARCH=configure(ROOT)
from c35_protocol import *
from c35_parent import probe,prepare,resume_score
from process_io import visible
def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    check_sources(ROOT)
    boot='import sys,runpy;n=int(sys.argv.pop(1));sys.path[:0]=[sys.argv.pop(1) for _ in range(n)];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    def prefix(extra=()):
        paths=list(map(str,SEARCH))+list(map(str,extra));return [sys.executable,'-u','-I','-c',boot,str(len(paths)),*paths]
    visible(prefix()+[str(ROOT/'test_c35_math.py')],'C35 calibration/split/bootstrap self-tests',timeout=180)
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
    base=Path('/content/gpt2_c35_prewrite_coefficient_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip:
        paths=list(base.glob('session_*/'+RESULT+'.zip'))
        if backup:paths+=list(Path(backup).glob(RESULT+'_*.zip'))
        good=[]
        for p in paths:
            try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
            except Exception as e:print('跳过不兼容C35恢复包',p.name,str(e),flush=True)
        if good:input_zip=max(good)[2]
    if not input_zip:
        ps=list(Path('/content/gpt2_c34_low_dimensional_control_a100_v1').glob('session_*/gpt2_c34_low_dimensional_control_results.zip'))
        if use_drive:ps+=list(Path('/content/drive/MyDrive/GPT2_C34_LOW_DIMENSIONAL_CONTROL_A100_V1_RECOVERY').glob('gpt2_c34_low_dimensional_control_results*.zip'))
        for p in sorted(ps,key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:probe(z,ROOT)
                input_zip=str(p);break
            except Exception as e:print('跳过非完整C34',p.name,str(e),flush=True)
    if not input_zip:
        from google.colab import files
        print('请上传完整C34结果ZIP或本版C35恢复包；不要上传源码ZIP，也不能用C32/C33精简包。',flush=True)
        u=files.upload();require(len(u)==1,'Upload exactly one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
    parent=Path(input_zip)
    with zipfile.ZipFile(parent) as z:
        require('identity.json' in z.namelist(),'请选择结果ZIP，不是源码ZIP');kind=zread(z,'identity.json')['protocol']
    require(kind in (PROTOCOL,PARENT_PROTOCOL),'C35 requires completed patched C34 or matching C35')
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True);result=session/(RESULT+'.zip')
    try:
        if kind==PROTOCOL:
            score=resume_score(parent,ROOT)
            with zipfile.ZipFile(parent) as z:z.extractall(out)
            print('恢复实际C35系数头、准备态与已封存单次提案：',score,flush=True)
        else:
            dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT)})
            local=session/'c34_readonly.zip';shutil.copyfile(parent,local)
            try:require(sha(local)==sha(parent),'Read-only copy differs');prepare(local,out,ROOT)
            finally:
                if local.exists():local.unlink()
        if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        if (out/'NEW_STATE_SPECS.json').exists():require(read(out/'NEW_STATE_SPECS.json')==specs(),'Declared new seeds changed')
        else:dump(out/'NEW_STATE_SPECS.json',specs())
        print('明确输入：',parent,flush=True)
        print('C35：冻结C34基；12态离线几何先验+36态校准；原写入/固定几何预测/学习系数三组，写前生成所有候选。',flush=True)
        print('运行时不测中间写后响应，无Jacobian/SVD/校正/回溯/教师兜底；每臂一次组合提案+外部提交/前向验收。',flush=True)
        print('离线几何与校准已编译为冻结仿射权重，不在线求导。冻结后生成新16态，种子/扰动幅度固定，不重抽。',flush=True)
        print('旧16态仅开发诊断；新态仍同模型/参数图；准备态允许外部几何。不测试内部原生更新或B5控制器。',flush=True)
        print('阈值0.002/0.005/0.05不变；无总会话时限；每路由/4态备份，仅清理本C35最近3份外已校验备份。',flush=True)
        check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
        deps=base/'deps'
        for d in (Path('/content/gpt2_c34_low_dimensional_control_a100_v1/deps'),Path('/content/gpt2_c31_residual_coverage_a100_v1/deps')):
            if d.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(d)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=d;break
        deps.mkdir(parents=True,exist_ok=True)
        if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
            visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
            visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后依赖检查',timeout=90)
        visible(prefix([deps])+[str(ROOT/'test_c35_torch.py')],'C35 frozen-forward/no-leak/resume self-tests',timeout=240)
        snapshot(out,backup)
        visible(prefix([deps])+[str(ROOT/'worker_c35.py'),'--out',str(out),'--drive-dir',backup],'C35 A100 frozen prewrite coefficient experiment',log_path=session/'console.log')
    finally:
        if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
        result=snapshot(out,backup);print('唯一结果总包：',result,flush=True)
    return result
if __name__=='__main__':start()
