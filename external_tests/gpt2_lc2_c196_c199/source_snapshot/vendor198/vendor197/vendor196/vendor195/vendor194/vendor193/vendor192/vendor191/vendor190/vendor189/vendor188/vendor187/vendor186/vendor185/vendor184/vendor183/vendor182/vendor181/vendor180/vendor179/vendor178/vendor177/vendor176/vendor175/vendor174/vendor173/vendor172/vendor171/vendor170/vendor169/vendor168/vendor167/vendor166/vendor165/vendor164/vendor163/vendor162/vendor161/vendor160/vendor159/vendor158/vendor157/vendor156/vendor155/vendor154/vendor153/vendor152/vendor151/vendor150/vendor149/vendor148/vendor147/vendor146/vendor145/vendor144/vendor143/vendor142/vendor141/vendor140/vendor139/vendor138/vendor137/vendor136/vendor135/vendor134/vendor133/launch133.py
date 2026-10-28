"""CPU launcher with bounded child processes and one success/failure capsule."""
from pathlib import Path
import os,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
from core133 import *

def child_environment():return dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0')

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
    need(actual==set(manifest)|{'SOURCE_MANIFEST.json'},'C133 source inventory changed')
    for n,h in manifest.items():need(sha(ROOT/n)==h,'C133 delivered source changed: '+n)
    base=Path(base or ('/content' if Path('/content').is_dir() else Path.cwd()));session=base/'gpt2_lc2_c133_finite_step_decomposition_cpu_v1'/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='launcher_started'))
    originals={};failure=None;drive=Path('/content/drive/MyDrive');pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes());result=session/('c133_finite_step_decomposition_'+str(time.time_ns())+'_results.zip')
    deps=session/'deps';deps.mkdir()
    boot='import sys,runpy;d=sys.argv.pop(1);r=sys.argv.pop(1);sys.path[:0]=[d,r];p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',boot,str(deps),str(ROOT)];env=child_environment()
    try:
        if use_drive and not drive.is_dir():
            from google.colab import drive as gd
            gd.mount('/content/drive')
        for k in reversed(INPUT_KEYS):
            originals[k]=choose(k,(inputs or {}).get(k,''),drive);print('C133 原始只读输入',k,str(originals[k]),flush=True)
        dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in originals.items()})
        with (out/'console.log').open('w',encoding='utf-8') as log:
            visible(prefix+[str(ROOT/'test133.py')],'C133 archive/local-binding/finite-decomposition/no-model tests',600,env,log)
            visible([sys.executable,'-I','-c',"import numpy,sys;assert 'torch' not in sys.modules;print('CPU_NUMPY_READY',numpy.__version__,flush=True)"],'C133 CPU NumPy',90,env,log)
            print('C133：CPU纯NumPy；只读C132原包，核对全部72条端点；仅对6个具有局部Jacobian的构造中心做36条有限步长分解、24条相对raw对照。',flush=True)
            print('独立重建裁剪前/后与float32实际提交增量；余项不等于纯曲率。无模型/前向/反传/拟合/求解/新Jacobian或SVD，不生成写入器、不测试排除输入；上限20分钟。',flush=True)
            args=[]
            for k,p in originals.items():args+=['--'+k.lower(),str(p)]
            visible(prefix+[str(ROOT/'worker133.py')]+args+['--out',str(out)],'C133 read-only finite-step decomposition',TIMEOUT,env,log)
        s=clean_json((out/'SUMMARY.json').read_bytes());need(s.get('complete') is True,'C133 incomplete');dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=True,stage='decomposition_complete_parent_decision_retained'))
    except BaseException as e:
        failure=e;dump(out/'LAUNCHER_ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));print('C133 stopped:',type(e).__name__,str(e),flush=True)
    finally:
        checks={}
        for k,p in originals.items():
            try:checks[k]=sha(p)==pins[k]['sha256']
            except Exception:checks[k]=False
        dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',checks)
        if not all(checks.values()):failure=failure or ValueError('Original input changed')
        if failure:
            s=clean_json((out/'SUMMARY.json').read_bytes()) if (out/'SUMMARY.json').is_file() else dict(protocol=PROTOCOL,local_decomposition_records_completed=0,arm_vs_raw_contrasts_completed=0,new_backward_calls=0,new_model_forwards=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False,parent_decision_reclassified=False)
            s.update(complete=False,decision='C133_INCOMPLETE_NO_LC2_CLAIM',launcher_failure=str(failure));dump(out/'SUMMARY.json',s);dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='stopped_and_packaged'))
        for n in ('DECOMPOSITION_PROGRESS.json','CONTRAST_PROGRESS.json'):
            if not (out/n).is_file():dump(out/n,[])
        current=clean_json((out/'SUMMARY.json').read_bytes())
        dump(out/'REQUESTED_WORK.json',dict(endpoint_records_to_audit=72,local_decomposition_requests=36,contrast_requests=24,unrecorded_local_decomposition_requests=max(0,36-current.get('local_decomposition_records_completed',0)),unrecorded_contrast_requests=max(0,24-current.get('arm_vs_raw_contrasts_completed',0))))
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
    if failure:raise RuntimeError('C133未完成；已封装失败结果：'+str(result)) from failure
    return result
