"""Pure NumPy read-only CPU audit launcher with separate bounded child processes."""
from pathlib import Path
import os,shutil,signal,subprocess,sys,threading,time,traceback,zipfile
from core185 import *

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
        reader.join(timeout=5);p.stdout.close();need(code==0,label+' exit='+str(code));print('END',label,'exit=',code,flush=True)
    except BaseException:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait();reader.join(timeout=5);p.stdout.close();raise

def choose(kind,explicit='',drive=Path('/content/drive/MyDrive')):
    r=clean_json((ROOT/'INPUT_PINS.json').read_bytes())[kind]
    if explicit:
        p=Path(explicit).expanduser().resolve();need(p.is_file() and sha(p)==r['sha256'],'Wrong fixed original '+kind);return p
    preferred=drive/Path(r['path']).relative_to('/content/drive/MyDrive')
    for p in [preferred,Path(r.get('local_path',str(preferred))),Path('/content')/preferred.name]:
        if p.is_file() and sha(p)==r['sha256']:return p
    raise FileNotFoundError('未找到固定'+kind+'科学结果ZIP。请挂载原Drive或填写'+kind+'_ZIP绝对路径；内容仍须匹配原哈希。')

def start(use_drive=True,inputs=None,auto_download=True,base=None):
    manifest=clean_json((ROOT/'SOURCE_MANIFEST.json').read_bytes());actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual==set(manifest)|{'SOURCE_MANIFEST.json'},'C185 source inventory changed')
    for n,h in manifest.items():need(sha(ROOT/n)==h,'C185 delivered source changed: '+n)
    base=Path(base or ('/content' if Path('/content').is_dir() else Path.cwd()));session=base/'gpt2_lc2_c185_readonly_continuation_train_audit_cpu_v1'/('session_'+str(time.time_ns()));out=session/'run';out.mkdir(parents=True)
    shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='launcher_started'))
    originals={};failure=None;drive=Path('/content/drive/MyDrive');pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes());result=session/('c185_readonly_continuation_train_audit_'+str(time.time_ns())+'_results.zip')
    boot='import sys,runpy;r=sys.argv.pop(1);sys.path.insert(0,r);p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name="__main__")'
    prefix=[sys.executable,'-u','-I','-c',boot,str(ROOT)];env=child_environment()
    try:
        if use_drive and not drive.is_dir():
            from google.colab import drive as gd
            gd.mount('/content/drive')
        for k in INPUT_KEYS:
            originals[k]=choose(k,(inputs or {}).get(k,''),drive);print('C185 原始只读输入',k,str(originals[k]),flush=True)
        dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in originals.items()})
        with (out/'console.log').open('w',encoding='utf-8') as log:
            visible(prefix+[str(ROOT/'test_math185.py')],'C185 independent LP/sign/units/binary64 endpoint checks',120,env,log)
            visible(prefix+[str(ROOT/'test_diagnostics185.py')],'C185 observed-cap/guard-decomposition controls',120,env,log)
            visible(prefix+[str(ROOT/'test_outcomes185.py')],'C185 full-gate rescue/destruction/repeat controls',120,env,log)
            visible(prefix+[str(ROOT/'test_precision185.py')],'C185 independent precision-margin and gate-separation tests',120,env,log)
            visible(prefix+[str(ROOT/'test_focus185.py')],'C185 dynamic rescue/failure/precision/crossing controls',120,env,log)
            visible(prefix+[str(ROOT/'test_activity185.py')],'C185 activity exit/re-entry/safety-vs-acceptance controls',120,env,log)
            visible(prefix+[str(ROOT/'test185.py')],'C185 archive/fixed8/point-seal/exact-affine/gate tests',1200,env,log)
            visible(prefix+[str(ROOT/'test_bootstrap185.py')],'C185 isolated current-entrypoint checks',90,env,log)
            visible(prefix+[str(ROOT/'test_launcher185.py')],'C185 failure-packaging checks (synthetic only)',120,env,log)
            print('C185：纯NumPy；固定C155–C184三十个原包；全84态、8步、1176记录、840实际FP64点、336原生门槛。重算救回/破坏与剩余失败轨迹，不预设81/84。',flush=True)
            print('重算全84态活动集退出/进入/再进入，区分0.0018训练安全尺度与0.002原生预算；重算每步1008有符号约束、全部16份证书、固定更新、4032条逐坐标记录、逐步损失和guard分项。独立重建336条精度裕度JSON/CSV。不加载模型或Torch/HF/SciPy；无前向/反传/训练/求解，不选步长或头；上限60分钟。',flush=True)
            args=[]
            for k,p in originals.items():args+=['--'+k.lower(),str(p)]
            visible(prefix+[str(ROOT/'worker185.py')]+args+['--out',str(out)],'C185 read-only all-coordinate training audit',TIMEOUT,env,log)
        s=clean_json((out/'SUMMARY.json').read_bytes());need(s.get('complete') is True,'C185 incomplete');dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=True,stage='readonly_all_coordinate_train_audit_complete'))
    except BaseException as e:
        failure=e;dump(out/'LAUNCHER_ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));print('C185 stopped:',type(e).__name__,str(e),flush=True)
    finally:
        checks={}
        for k,p in originals.items():
            try:checks[k]=sha(p)==pins[k]['sha256']
            except Exception:checks[k]=False
        dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',checks)
        if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):failure=failure or ValueError('Original input missing or changed')
        if failure:
            s=clean_json((out/'SUMMARY.json').read_bytes()) if (out/'SUMMARY.json').is_file() else dict(protocol=PROTOCOL,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False,parent_decision_reclassified=False)
            s.update(complete=False,decision=INCOMPLETE,runtime_counts_finalized=False,launcher_failure=str(failure));dump(out/'SUMMARY.json',s);dump(out/'STATUS.json',dict(protocol=PROTOCOL,complete=False,stage='stopped_and_packaged'))
        dump(out/'REQUESTED_WORK.json',dict(training_states_requested=84,new_model_forwards_requested=0,scalar_backward_calls_requested=0,new_training_steps_requested=0))
        result_manifest={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'RESULT_MANIFEST.json'};dump(out/'RESULT_MANIFEST.json',result_manifest)
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
    if failure:raise RuntimeError('C185未完成；已封装失败结果：'+str(result)) from failure
    return result
