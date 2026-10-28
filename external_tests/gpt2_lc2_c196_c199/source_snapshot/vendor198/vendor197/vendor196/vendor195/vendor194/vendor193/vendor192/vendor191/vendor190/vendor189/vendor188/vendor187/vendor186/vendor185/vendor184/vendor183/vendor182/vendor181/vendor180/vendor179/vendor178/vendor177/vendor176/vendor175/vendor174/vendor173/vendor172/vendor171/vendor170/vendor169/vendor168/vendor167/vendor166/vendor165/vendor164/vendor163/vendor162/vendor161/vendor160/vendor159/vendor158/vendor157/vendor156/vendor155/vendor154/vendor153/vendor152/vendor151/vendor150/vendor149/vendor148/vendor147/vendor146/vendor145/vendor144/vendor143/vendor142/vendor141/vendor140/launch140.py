"""CPU launcher with bounded child processes and one success/failure capsule."""
from pathlib import Path
import os,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
from core140 import *

def child_environment():return dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')

def dependencies(deps,env,log):
    visible([sys.executable,'-I','-c',"import torch,numpy;v=torch.ones(2,device='cpu');assert v.device.type=='cpu';print('CPU_TORCH_READY',torch.__version__,numpy.__version__,flush=True)"],'C140 CPU PyTorch',90,env,log)
    script="import sys;sys.path.insert(0,sys.argv[1]);import transformers,peft,tokenizers,huggingface_hub,accelerate,safetensors;v=[transformers.__version__,peft.__version__,tokenizers.__version__,huggingface_hub.__version__,accelerate.__version__,safetensors.__version__];assert v==['4.56.2','0.17.1','0.22.0','0.34.4','1.10.1','0.6.2'],v;print('PINNED_HF_DEPENDENCIES_OK',v,flush=True)"
    check=[sys.executable,'-I','-c',script,str(deps)];p=subprocess.run(check,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=90)
    if p.returncode:
        print('C140：在隔离目录安装固定HF依赖；保留系统PyTorch与NumPy。',flush=True)
        visible([sys.executable,'-I','-m','pip','install','--upgrade','--no-deps','--only-binary=:all:','--progress-bar','off','--timeout','30','--retries','2','--target',str(deps),'transformers==4.56.2','peft==0.17.1','tokenizers==0.22.0','huggingface_hub==0.34.4','accelerate==1.10.1','safetensors==0.6.2'],'C140 isolated pinned HF dependencies',600,env,log)
    visible(check,'C140 pinned HF check',90,env,log)

def visible(cmd,label,timeout,env,log):
    print('START',label,flush=True);started=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
    def drain():
        for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    reader=threading.Thread(target=drain,daemon=True);reader.start()
    try:
        while True:
            try:code=p.wait(timeout=min(30,max(.1,timeout-(time.monotonic()-started))));break
            except subprocess.TimeoutExpired:
                if time.monotonic()-started>=timeout:raise TimeoutError(label+' exceeded fixed budget')
                print(label,'运行中',int(time.monotonic()-started),'秒',flush=True)
        reader.join(timeout=5);need(code==0,label+' exit='+str(code));print('END',label,'exit=',code,flush=True)
    except BaseException:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait();reader.join(timeout=5);raise

def choose(kind,explicit='',drive=Path('/content/drive/MyDrive')):
    r=clean_json((ROOT/'INPUT_PINS.json').read_bytes())[kind]
    if explicit:
        p=Path(explicit).expanduser().resolve();need(p.is_file() and sha(p)==r['sha256'],'Wrong fixed original '+kind);return p
    preferred=drive/Path(r['path']).relative_to('/content/drive/MyDrive')
    for p in [preferred,Path('/content')/preferred.name]:
        if p.is_file() and sha(p)==r['sha256']:return p
    raise FileNotFoundError('未找到固定'+kind+'科学结果ZIP。请挂载原Drive或填写'+kind+'_ZIP绝对路径；内容仍须匹配原哈希。')

def start(use_drive=True,inputs=None,auto_download=True,base=None):
    manifest=clean_json((ROOT/'SOURCE_MANIFEST.json').read_bytes());actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual==set(manifest)|{'SOURCE_MANIFEST.json'},'C140 source inventory changed')
    for n,h in manifest.items():need(sha(ROOT/n)==h,'C140 delivered source changed: '+n)
    base=Path(base or ('/content' if Path('/content').is_dir() else Path.cwd()));session=base/'gpt2_lc2_c140_expanded_margin_training_cpu_v1'/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='launcher_started'))
    originals={};failure=None;drive=Path('/content/drive/MyDrive');pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes());result=session/('c140_expanded_margin_training_'+str(time.time_ns())+'_results.zip')
    deps=session/'deps';deps.mkdir()
    boot='import sys,runpy;d=sys.argv.pop(1);r=sys.argv.pop(1);sys.path[:0]=[d,r];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',boot,str(deps),str(ROOT)];env=child_environment()
    try:
        if use_drive and not drive.is_dir():
            from google.colab import drive as gd
            gd.mount('/content/drive')
        for k in reversed(INPUT_KEYS):
            originals[k]=choose(k,(inputs or {}).get(k,''),drive);print('C140 原始只读输入',k,str(originals[k]),flush=True)
        dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in originals.items()})
        with (out/'console.log').open('w',encoding='utf-8') as log:
            visible(prefix+[str(ROOT/'test140.py')],'C140 archive/full36/Adam/schedule/zero-warm-parity/final-only tests',900,env,log)
            dependencies(deps,env,log)
            visible(prefix+[str(ROOT/'torch_smoke140.py')],'C140 actual PyTorch CPU functional-gradient smoke',180,env,log)
            print('C140：真实GPT2 CPU；原12中心+C139全部24态均为训练数据，一个共享36维向量；从C137固定第24步初始化，重置Adam矩。',flush=True)
            print('固定12步全36态加权Adam，共432次逐态反传；训练响应安全线0.00125，验收门槛仍0.002。只验收第12步。',flush=True)
            print('只读C132-C139八原包并重算C139归档；108已知点重放和全36态零/初始功能前向一致后才反传。最终raw/C137/C140三臂两轮216条记录，预计1657次模型前向；工作进程上限50分钟。',flush=True)
            print('主干/A/原writer/V/归一化固定；无新Jacobian/SVD/求解/教师/回溯/在线修复；不选最佳或恢复头；全部失败保留；不做独立确认或排除输入，不闭合LC2。',flush=True)
            args=[]
            for k,p in originals.items():args+=['--'+k.lower(),str(p)]
            visible(prefix+[str(ROOT/'worker140.py')]+args+['--out',str(out)],'C140 fixed12 expanded-margin full36 training',TIMEOUT,env,log)
        s=clean_json((out/'SUMMARY.json').read_bytes());need(s.get('complete') is True,'C140 incomplete');dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=True,stage='fixed12_training_and_qualification_complete_no_LC2_claim'))
    except BaseException as e:
        failure=e;dump(out/'LAUNCHER_ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));print('C140 stopped:',type(e).__name__,str(e),flush=True)
    finally:
        checks={}
        for k,p in originals.items():
            try:checks[k]=sha(p)==pins[k]['sha256']
            except Exception:checks[k]=False
        dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',checks)
        if not all(checks.values()):failure=failure or ValueError('Original input changed')
        if failure:
            s=clean_json((out/'SUMMARY.json').read_bytes()) if (out/'SUMMARY.json').is_file() else dict(protocol=PROTOCOL,completed_training_steps=0,final_endpoint_records=0,new_backward_calls=0,new_model_forwards=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False,parent_decision_reclassified=False)
            s.update(complete=False,decision='C140_INCOMPLETE_FIXED_FINAL_NOT_QUALIFIED',launcher_failure=str(failure));dump(out/'SUMMARY.json',s);dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='stopped_and_packaged'))
        for n in ('TRAINING_PROGRESS.json','ZERO_FUNCTIONAL_PROGRESS.json','WARM_FUNCTIONAL_PROGRESS.json','FINAL_ENDPOINT_PROGRESS.json','PREFLIGHT_PROGRESS.json'):
            if not (out/n).is_file():dump(out/n,[])
        current=clean_json((out/'SUMMARY.json').read_bytes())
        dump(out/'REQUESTED_WORK.json',dict(training_states_requested=36,training_steps_requested=12,scalar_backward_calls_requested=432,native_preflight_points_requested=108,functional_zero_parity_states_requested=36,functional_warm_parity_states_requested=36,final_endpoint_measurement_requests=216,uncompleted_training_steps=max(0,12-current.get('completed_training_steps',0)),uncompleted_scalar_backward_calls=max(0,432-current.get('new_backward_calls',0)),unrecorded_preflight_requests=max(0,108-current.get('preflight_records',0)),unrecorded_final_endpoint_requests=max(0,216-current.get('final_endpoint_records',0))))
        result_manifest={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.json'};dump(out/'RESULT_MANIFEST.json',result_manifest)
        with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(out).as_posix())
        print('唯一结果总包：',result,flush=True);print('RESULT_SHA256',sha(result),flush=True)
        if use_drive and drive.is_dir():
            try:
                target=drive/(PROTOCOL+'_RECOVERY')/result.name;target.parent.mkdir(exist_ok=True);shutil.copy2(result,target);need(sha(target)==sha(result),'Drive backup hash differs');print('DRIVE_BACKUP_OK',target,flush=True)
            except Exception as e:print('Drive备份失败；本地总包保留：',repr(e),flush=True)
        if auto_download:
            try:
                from google.colab import files
                files.download(str(result))
            except Exception as e:print('请从上方路径下载总包：',repr(e),flush=True)
    if failure:raise RuntimeError('C140未完成；已封装失败结果：'+str(result)) from failure
    return result
