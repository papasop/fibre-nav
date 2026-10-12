from pathlib import Path
import os,shutil,sys,time,zipfile
from c7_protocol import *
from process_io import visible

def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
    root=Path(__file__).resolve().parent
    visible([sys.executable,'-u',str(root/'test_c7.py')],'C7 functional training-gate protocol/SVD self-tests',timeout=60)
    visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],
            'A100 device',timeout=60)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(x).name in ('worker_c1.py','worker_c2.py','worker_c3.py','worker_c4.py','worker_c5.py','worker_c6.py','worker_c7.py','worker_b5.py','worker_b4.py') for x in argv):
            raise RuntimeError('Another C1/C2/C3/C4/C5/C6/C7/B4/B5 worker is active; stop it before launching C7')
    backup=''
    if use_drive:
        from google.colab import drive
        drive.mount('/content/drive');backup='/content/drive/MyDrive/GPT2_C7_FUNCTIONAL_GATE_A100_V1_RECOVERY'
    base=Path('/content/gpt2_c7_functional_gate_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip and backup:
        valid=[]
        for p in Path(backup).glob('gpt2_c7_functional_gate_results_*.zip'):
            try:valid.append((resume_score(p),p.stat().st_mtime,str(p)))
            except Exception as e:print('跳过不兼容C7备份：',p,str(e),flush=True)
        if valid:input_zip=max(valid)[2]
    if not input_zip:
        candidates=list(Path('/content').glob('gpt2_c6_staged_writer_results*.zip'))
        if use_drive:
            candidates+=list(Path('/content/drive/MyDrive/GPT2_C6_STAGED_WRITER_A100_V1_RECOVERY').glob('gpt2_c6_staged_writer_results_*.zip'))
        for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
            try:validate_c6(p);input_zip=str(p);break
            except Exception as e:print('跳过非指定C6输入：',p,str(e),flush=True)
    if not input_zip:
        print('请上传刚分析的 gpt2_c6_staged_writer_results.zip，或本版C7结果续跑。不要上传C1/C2/C3/C4/C5/B5/源码/启动器ZIP。',flush=True)
        from google.colab import files
        uploaded=files.upload()
        if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
        input_zip=str(Path('/content')/next(iter(uploaded)))
    path=Path(input_zip);ident=verified_zip(path)
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
    if ident['protocol']==PROTOCOL:
        progress=resume_score(path)
        with zipfile.ZipFile(path) as z:z.extractall(out)
        print('恢复C7真实进度：',progress,flush=True)
    elif ident['protocol']==C6_PROTOCOL:
        validate_c6(path);shutil.copyfile(path,out/'parent_c6.zip')
        print('C6结果完整校验通过；复用四个真实1200步头，不重训warm阶段，C6结论不改判。',flush=True)
    else:raise ValueError('C7 accepts the inspected C6 result or matching C7 results only')
    print('明确输入：',path,flush=True)
    print('C7：先测四头冻结基线；MSE仅诊断，实际训练9/9通过才进入两臂各200步对照；排除输入不用于资格选择。',flush=True)
    print('响应校正最多8步，每步最多6次预声明回溯；响应/KL/读取阈值保持0.002/0.005/0.05。',flush=True)
    print('两臂从相同C6头开始，学习率0.0001；温和功能损失与外部直接校正保持C6设定；不是纯前向内化。',flush=True)
    print('无限会话时长；Drive只保留本C7目录最近3个已校验备份；旧存档不修改；每25步本地保存，每200步/基线检查/评估检查点Drive备份。',flush=True)
    deps=base/'deps';deps.mkdir(parents=True,exist_ok=True)
    check=("import sys;sys.path.insert(0,sys.argv[1]);import torch,numpy,transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;"
           "assert transformers.__version__=='4.56.2';assert peft.__version__=='0.17.1';print('DEPENDENCIES_OK',flush=True)")
    if visible([sys.executable,'-u','-I','-c',check,str(deps)],'HF依赖检查',timeout=90,check=False):
        visible([sys.executable,'-u','-m','pip','--disable-pip-version-check','install','--upgrade','--no-deps',
                 '--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),
                 'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4',
                 'accelerate==1.10.1','safetensors==0.6.2'],'隔离HF依赖；保留CUDA PyTorch',timeout=600)
        visible([sys.executable,'-u','-I','-c',check,str(deps)],'安装后检查',timeout=90)
    bootstrap='import sys,runpy;sys.path.insert(0,sys.argv.pop(1));sys.path.insert(0,sys.argv.pop(1));s=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(s,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',bootstrap,str(deps),str(root)]
    visible(prefix+[str(root/'test_c7_torch.py')],'C7 Torch autograd/isolation self-tests',timeout=90)
    result=session/'gpt2_c7_functional_gate_results.zip'
    try:
        visible(prefix+[str(root/'worker_c7.py'),'--out',str(out),'--drive-dir',backup],
                'C7 A100 functional-gate comparison',log_path=session/'console.log')
    except KeyboardInterrupt:
        print('已中断；请用本C7最新校验结果续跑，已完成配对/臂不会重复执行。',flush=True)
    finally:
        try:
            saved=snapshot(out,backup)
            if saved:result=saved
        except Exception as e:print('最终快照未完成；保留已有已校验ZIP：',type(e).__name__,str(e),flush=True)
        if result.exists():
            print('唯一结果总包：',result,flush=True)
            try:
                from google.colab import files
                files.download(str(result))
            except Exception as e:print('请手动保存结果ZIP：',result,type(e).__name__,flush=True)

if __name__=='__main__':start()
