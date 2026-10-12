from pathlib import Path
import os,shutil,sys,time,zipfile
from c9_protocol import *
from process_io import visible

def start(use_drive=True,input_zip=''):
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')
    root=Path(__file__).resolve().parent
    visible([sys.executable,'-u',str(root/'test_c9.py')],'C9 split/freeze/gate protocol self-tests',timeout=60)
    visible([sys.executable,'-I','-c',"import torch;assert torch.cuda.is_available();n=torch.cuda.get_device_name(0);assert 'A100' in n,n;print(n,torch.__version__,flush=True)"],
            'A100 device',timeout=60)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(x).name in ('worker_c1.py','worker_c2.py','worker_c3.py','worker_c4.py','worker_c5.py','worker_c6.py','worker_c7.py','worker_c8.py','worker_c9.py','worker_b5.py','worker_b4.py') for x in argv):
            raise RuntimeError('Another C1/C2/C3/C4/C5/C6/C9/B4/B5 worker is active; stop it before launching C9')
    backup=''
    if use_drive:
        from google.colab import drive
        drive.mount('/content/drive');backup='/content/drive/MyDrive/GPT2_C9_STATE_GENERALISATION_A100_V1_RECOVERY'
    base=Path('/content/gpt2_c9_state_generalisation_a100_v1');base.mkdir(parents=True,exist_ok=True)
    if not input_zip and backup:
        valid=[]
        for p in Path(backup).glob('gpt2_c9_state_generalisation_results_*.zip'):
            try:valid.append((resume_score(p),p.stat().st_mtime,str(p)))
            except Exception as e:print('跳过不兼容C9备份：',p,str(e),flush=True)
        if valid:input_zip=max(valid)[2]
    if not input_zip:
        candidates=list(Path('/content').glob('gpt2_c8_extrapolation_diagnosis_results*.zip'))
        if use_drive:
            candidates+=list(Path('/content/drive/MyDrive/GPT2_C8_EXTRAPOLATION_DIAGNOSIS_A100_V1_RECOVERY').glob('gpt2_c8_extrapolation_diagnosis_results_*.zip'))
        for p in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
            try:validate_c8(p);input_zip=str(p);break
            except Exception as e:print('跳过非指定C8输入：',p,str(e),flush=True)
    if not input_zip:
        print('请上传刚分析的完整 gpt2_c8_extrapolation_diagnosis_results*.zip，或本版C9结果续跑。不要上传C1/C2/C3/C4/C5/B5/源码/启动器ZIP。',flush=True)
        from google.colab import files
        uploaded=files.upload()
        if len(uploaded)!=1:raise ValueError('Upload exactly one result ZIP')
        input_zip=str(Path('/content')/next(iter(uploaded)))
    path=Path(input_zip);ident=verified_zip(path)
    session=base/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True,exist_ok=True)
    if ident['protocol']==PROTOCOL:
        progress=resume_score(path)
        with zipfile.ZipFile(path) as z:z.extractall(out)
        validate_c8(out/'parent_c8.zip');print('恢复C9实际训练/评估进度：',progress,flush=True)
    elif ident['protocol']==PARENT_PROTOCOL:
        validate_c8(path);shutil.copyfile(path,out/'parent_c8.zip')
        print('指定C8结果完整校验通过；原始父包/源码/缓存身份保持。',flush=True)
    else:raise ValueError('C9 accepts the exact inspected C8 result or matching C9 results only')
    print('明确输入：',path,flush=True)
    print('C9：四类XOR全部进入训练；12个已有状态；新写入器MSE 2000步+功能细化240步；固定最终头。',flush=True)
    print('冻结写入器后生成8个固定种子新准备态；范数0.375/0.75；不按成绩重采样或选择检查点。',flush=True)
    print('读取/响应/KL阈值保持0.05/0.002/0.005；新状态教师仅作正对照，不替换预测；原始/校正后分别报告。',flush=True)
    print('这测试已见输入类附近的局部状态泛化，仍是开发；不是未见XOR组合、独立模型确认或原生参数更新。',flush=True)
    print('Drive只保留本C9目录最近3个已校验备份；父包不修改；无限会话时长，真实检查点续跑。',flush=True)
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
    visible(prefix+[str(root/'test_c9_torch.py')],'C9 Torch writer gradient/isolation self-tests',timeout=90)
    result=session/'gpt2_c9_state_generalisation_results.zip'
    try:
        visible(prefix+[str(root/'worker_c9.py'),'--out',str(out),'--drive-dir',backup],
                'C9 A100 four-input training and frozen state generalisation',log_path=session/'console.log')
    except KeyboardInterrupt:
        print('已中断；请用本C9最新校验结果续跑，不重复已完成训练阶段或评估候选。',flush=True)
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
