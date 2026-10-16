import os, sys, time, shutil, zipfile
from pathlib import Path
from c31_protocol import *
from c31_parent import prepare_parent, resume_score
from c30_data import probe
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'inherited'),str(ROOT/'inherited/inherited'),str(ROOT/'inherited/inherited/inherited'),str(ROOT/'inherited/inherited/inherited/legacy')]


def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    check_sources(ROOT)
    from process_io import visible
    paths=[str(ROOT/'inherited/inherited/inherited/legacy'),str(ROOT/'inherited/inherited/inherited'),str(ROOT/'inherited/inherited'),str(ROOT/'inherited'),str(ROOT)]
    boot='import sys,runpy;n=int(sys.argv.pop(1));paths=[sys.argv.pop(1) for _ in range(n)];sys.path[:0]=list(reversed(paths));p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',boot,str(len(paths))]+paths
    visible(prefix+[str(ROOT/'test_c31_protocol.py')],'C31 fixed split/archive/source self-tests',timeout=120)
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
    base=Path('/content/gpt2_c31_residual_coverage_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip:
        paths0=list(base.glob('session_*/'+RESULT+'.zip'))
        if backup:paths0+=list(Path(backup).glob(RESULT+'_*.zip'))
        good=[]
        for p in paths0:
            try:good.append((resume_score(p,ROOT),p.stat().st_mtime_ns,str(p)))
            except Exception as e:print('跳过不兼容C31恢复包',p.name,str(e),flush=True)
        if good:input_zip=max(good)[2]
    if not input_zip:
        ps=list(Path('/content/gpt2_c29_frozen_direction_diagnosis_a100_v1').glob('session_*/gpt2_c29_frozen_direction_diagnosis_results.zip'))
        if use_drive:ps+=list(Path('/content/drive/MyDrive/GPT2_C29_FROZEN_DIRECTION_DIAGNOSIS_A100_V1_RECOVERY').glob('gpt2_c29_frozen_direction_diagnosis_results*.zip'))
        for p in sorted(ps,key=lambda p:p.stat().st_mtime_ns,reverse=True):
            try:
                with zipfile.ZipFile(p) as z:probe(z,ROOT)
                input_zip=str(p);break
            except Exception as e:print('跳过非完整C29',p.name,str(e),flush=True)
    if not input_zip:
        from google.colab import files
        print('请上传已完成C29结果ZIP，或本版C31恢复包。C30精简包不含真实写入器权重，不能作为输入。',flush=True)
        u=files.upload();require(len(u)==1,'Upload exactly one result ZIP');input_zip=str(Path(next(iter(u))).resolve())
    parent=Path(input_zip)
    with zipfile.ZipFile(parent) as z:
        require('identity.json' in z.namelist(),'请选择结果ZIP，不是源码ZIP')
        kind=zread(z,'identity.json')['protocol']
    require(kind in (PROTOCOL,PARENT_PROTOCOL),'C31 requires completed C29 or matching C31 recovery')
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    if kind==PROTOCOL:
        score=resume_score(parent,ROOT)
        with zipfile.ZipFile(parent) as z:z.extractall(out)
        print('恢复C31真实状态/端点/预测器记录：',score,flush=True)
    else:
        dump(out/'identity.json',{'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(ROOT)})
        local=session/'parent_c29_readonly.zip';shutil.copyfile(parent,local)
        try:
            require(sha(local)==sha(parent),'Read-only parent copy differs');prepare_parent(local,out,ROOT)
        finally:
            if local.exists():local.unlink()
    if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    fixed={'new_training':specs('new_training'),'prospective':specs('prospective'),'old_inspected16_used':False,'resampling_allowed':False}
    if (out/'FIXED_STATE_SPECS.json').exists():require(read(out/'FIXED_STATE_SPECS.json')==fixed,'Predeclared specs changed')
    else:dump(out/'FIXED_STATE_SPECS.json',fixed)
    print('明确输入：',parent,flush=True)
    print('C31：冻结真实C27；原76对照 vs 原76+48个固定新训练态；另16个固定新种子开发态。',flush=True)
    print('新训练48态全部收集，不按写后残差筛选、不逐态校准写入器；准备失败不替换。',flush=True)
    print('两组残差预测器冻结后才生成新16态。此轮预测诊断不构造写回补偿，LC1仍另行验收。',flush=True)
    print('离线状态准备允许Jacobian/SVD/有限响应修正；写后端点不修正、不回溯、不兜底。',flush=True)
    print('先重放12个历史缓存输入与C27端点；无总会话时限；只清理本C31超过最近3份的已校验备份。',flush=True)
    check="import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)"
    deps=base/'deps'
    for d in (Path('/content/gpt2_c29_frozen_direction_diagnosis_a100_v1/deps'),Path('/content/gpt2_c28_response_aware_four_route_a100_v1/deps')):
        if d.is_dir() and visible([sys.executable,'-u','-I','-c',check,str(d)],'复用隔离HF依赖',timeout=90,check=False)==0:deps=d;break
    deps.mkdir(parents=True,exist_ok=True)
    if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
        visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
        visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后依赖检查',timeout=90)
    gpu_paths=[str(deps)]+paths
    prefix=[sys.executable,'-u','-I','-c',boot,str(len(gpu_paths))]+gpu_paths
    # root and current source take precedence over inherited libraries; deps over global HF.
    visible(prefix+[str(ROOT/'test_c31_torch.py')],'C31 sketch/predictor/freeze/resume self-tests',timeout=180)
    result=session/(RESULT+'.zip')
    try:
        snapshot(out,backup)
        visible(prefix+[str(ROOT/'worker_c31.py'),'--out',str(out),'--drive-dir',backup],'C31 A100 residual coverage comparison',log_path=session/'console.log')
    finally:
        if (session/'console.log').exists():shutil.copyfile(session/'console.log',out/'console.log')
        result=snapshot(out,backup);print('唯一结果总包：',result,flush=True)
    return result


if __name__=='__main__':start()
