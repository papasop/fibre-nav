#!/usr/bin/env python3
"""V9 CPU Colab uploader / local launcher. Paste into a cell or run as a .py."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

PROFILE = 'two_step_chain'
MINUTES = 90
THREADS = 4
USE_GOOGLE_DRIVE = False
AUTO_DOWNLOAD_STAGES = True
AUTO_DOWNLOAD_FINAL = True
SOURCE_MANIFEST_SHA256 = '9fdfc82381f988451d3b84695a4f3703191deb44f95d9faadd7834ac5c37a038'
PROTOCOL_ID = 'QWEN_CHAIN_CPU_V9'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive member: '+name)
    return p


def inspect_archive(path):
    """Identify by protocol + manifest contents, never by upload filename."""
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>512*1024**2:
            raise ValueError('Duplicate members or oversized ZIP')
        for n in names:safe_name(n)
        manifests=[n for n in names if n.endswith('/RESULT_MANIFEST.json') or n=='RESULT_MANIFEST.json']
        if manifests:
            if len(manifests)!=1:raise ValueError('Expected one result root')
            prefix=manifests[0][:-len('RESULT_MANIFEST.json')]
            m=json.loads(z.read(manifests[0]))
            payload={n[len(prefix):] for n in names if n.startswith(prefix) and not n.endswith('/')}
            if payload!=set(m)|{'RESULT_MANIFEST.json'}:raise ValueError('Result inventory mismatch')
            for n,h in m.items():
                safe_name(n)
                if sha(z.read(prefix+n))!=h:raise ValueError('Result checksum mismatch: '+n)
            lock=json.loads(z.read(prefix+'RUN_LOCK.json'))
            if lock.get('protocol_id')!=PROTOCOL_ID or lock.get('source_manifest_sha256')!=SOURCE_MANIFEST_SHA256:
                raise ValueError('只能续跑本次 V9 的完整或阶段 results.zip；旧版本结果不能混入。')
            return 'resume',prefix,m
        candidates=[]
        for n in names:
            if n=='MANIFEST.json' or n.endswith('/MANIFEST.json'):
                prefix=n[:-len('MANIFEST.json')]
                if prefix+'protocol.json' in names and prefix+'run_cpu.py' in names:candidates.append((n,prefix))
        if len(candidates)!=1:raise ValueError('未找到唯一实验源；请上传 qwen_chain_cpu_v9.zip。')
        manifest,prefix=candidates[0];m=json.loads(z.read(manifest))
        if sha(canonical(m))!=SOURCE_MANIFEST_SHA256:raise ValueError('V9 源清单不匹配；启动器与 ZIP 必须来自同一次交付。')
        for n,h in m.items():
            safe_name(n)
            if sha(z.read(prefix+n))!=h:raise ValueError('Source checksum mismatch: '+n)
        if json.loads(z.read(prefix+'protocol.json'))['protocol_id']!=PROTOCOL_ID:raise ValueError('Protocol mismatch')
        return 'source',prefix,m


def materialize(path,description,destination):
    kind,prefix,manifest=description
    destination=Path(destination)
    members=list(manifest)+(['MANIFEST.json'] if kind=='source' else ['RESULT_MANIFEST.json'])
    with zipfile.ZipFile(path) as z:
        # Check all conflicts before any writes.
        for name in members:
            safe_name(name);target=destination/name
            if target.exists() and target.read_bytes()!=z.read(prefix+name):
                raise RuntimeError('目标已有不同记录，未覆盖：'+str(target)+'；请用新的 --workdir 续跑。')
        for name in members:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(prefix+name))


def bootstrap_code():
    # -S keeps unrelated notebook packages (torchvision/librosa/etc.) out of the worker.
    return "import sys,runpy; deps=sys.argv.pop(1); script=sys.argv.pop(1); sys.argv[0]=script; sys.path.insert(0,deps); sys.path.insert(0,__import__('os').path.dirname(script)); runpy.run_path(script,run_name='__main__')"


def prepare_dependencies(root,source,skip_install=False):
    deps=root/'cpu_dependencies'
    choices=[Path('/content/qwen_compiler_cpu_v8/cpu_dependencies'),Path('/content/qwen_compiler_cpu_v8_resume/cpu_dependencies'),Path('/content/qwen_scope_cpu_v2/cpu_dependencies'),root.parent/'cpu_dependencies']
    deps=next((p for p in choices if p.is_dir()),root.parent/'cpu_dependencies')
    deps.mkdir(parents=True,exist_ok=True)
    probe="import sys;sys.path.insert(0,sys.argv[1]);import torch,transformers,numpy,tokenizers,huggingface_hub,safetensors;from transformers import Qwen3ForCausalLM;assert torch.__version__.split('+')[0]=='2.8.0';assert torch.version.cuda is None;assert transformers.__version__=='4.56.2';assert numpy.__version__=='2.3.5';assert tokenizers.__version__=='0.22.2';assert huggingface_hub.__version__=='0.36.2';assert safetensors.__version__=='0.6.2';print('CPU dependency check OK')"
    check=[sys.executable,'-I','-S','-c',probe,str(deps)]
    if subprocess.run(check,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        if skip_install:raise RuntimeError('Dependencies missing; remove --skip-install')
        print('安装隔离的 CPU 依赖（首次需要联网）；无需 venv 或 ensurepip。',flush=True)
        base=[sys.executable,'-m','pip','install','--disable-pip-version-check','--only-binary=:all:','--upgrade','--target',str(deps)]
        subprocess.run(base+['torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu'],check=True)
        subprocess.run(base+['-r',str(source/'requirements_cpu.txt')],check=True)
    subprocess.run(check,check=True)
    return deps


def launch(argv=None):
    try:
        from google.colab import files
        colab=True
    except ImportError:
        files=None;colab=False
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--workdir',type=Path,default=Path('/content/qwen_chain_cpu_v9') if colab else Path.cwd()/'qwen_chain_cpu_v9_work')
    parser.add_argument('--minutes',type=float,default=MINUTES);parser.add_argument('--threads',type=int,choices=[4],default=THREADS)
    parser.add_argument('--no-download',action='store_true');parser.add_argument('--skip-install',action='store_true')
    # A notebook's -f kernel.json must never become experiment arguments.
    a=parser.parse_args([] if colab and argv is None else argv)
    if a.minutes<=0 or a.threads<1:parser.error('minutes and threads must be positive')
    os.environ.update(CUDA_VISIBLE_DEVICES='',USE_TORCH='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_XET='1')
    base=a.workdir.resolve();base.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='session_',dir=base))
    print('启动器：QWEN_V9_TWO_STEP；本次目录：'+str(root),flush=True)
    print('CPU 模式，GPU 不参与计算。Profile: '+PROFILE,flush=True)
    print('Qwen V9：实际中间输出传递＋翻转对照；16直接调用＋32两步程序，共112检查点；CPU，无训练。',flush=True)
    candidates=[p for p in (a.source,a.resume) if p]
    source=root/'source';out=root/'run'/'cpu_two_step_chain'
    if not candidates and colab:
        print('请上传 qwen_chain_cpu_v9.zip；续跑可同时上传一份 V9 *_results.zip。',flush=True)
        uploaded=files.upload();uploads=root/'uploads';uploads.mkdir(exist_ok=True)
        for name,data in uploaded.items():
            path=uploads/Path(name).name;path.write_bytes(data);candidates.append(path)
    descriptions=[(p,inspect_archive(p)) for p in candidates]
    sources=[x for x in descriptions if x[1][0]=='source'];resumes=[x for x in descriptions if x[1][0]=='resume']
    if len(sources)>1 or len(resumes)>1:raise RuntimeError('每次最多一个源 ZIP 和一个续跑 ZIP。')
    if sources:materialize(*sources[0],source)
    if not (source/'MANIFEST.json').is_file():raise RuntimeError('请提供 V9 源 ZIP。它已包含历史对照，不需要之前版本的结果。')
    m=json.loads((source/'MANIFEST.json').read_text())
    if sha(canonical(m))!=SOURCE_MANIFEST_SHA256 or any(sha((source/n).read_bytes())!=h for n,h in m.items()):raise RuntimeError('Existing source integrity failure')
    if resumes:materialize(*resumes[0],out)
    drive_target=None
    if USE_GOOGLE_DRIVE and colab:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            drive_target=Path('/content/drive/MyDrive/qwen_chain_cpu_v9');drive_target.mkdir(parents=True,exist_ok=True)
        except Exception as e:print('Drive 不可用，继续本地保存与下载：'+str(e),flush=True)
    def deliver(path,download):
        path=Path(path)
        if not path.is_file():return
        print('已保存：'+str(path),flush=True)
        if drive_target:
            try:shutil.copy2(path,drive_target/path.name)
            except Exception as e:print('Drive 备份失败，本地包仍在：'+str(e),flush=True)
        if download and colab and not a.no_download:
            try:files.download(str(path))
            except Exception as e:print('自动下载未完成，请用上面的实际路径下载：'+str(e),flush=True)
    deps=prepare_dependencies(root,source,a.skip_install)
    os.environ['V9_CPU_DEPS']=str(deps.resolve())
    command=[sys.executable,'-I','-S','-c',bootstrap_code(),str(deps),str(source/'run_cpu.py'),'--profile',PROFILE,'--out',str(out),'--minutes',str(a.minutes),'--threads',str(a.threads)]
    log=root/'QWEN_CHAIN_V9_SESSION_OUTPUT.txt';child=None
    try:
        with log.open('a',encoding='utf-8') as f:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
                if line.startswith('STAGE_ARCHIVE='):deliver(line.strip().split('=',1)[1],AUTO_DOWNLOAD_STAGES)
            code=child.wait();child=None
            if code:print('实验进程退出码：'+str(code)+'；请检查日志及已保存结果。',flush=True)
    except KeyboardInterrupt:
        if child is not None:
            import signal
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.terminate()
        print('已请求停止；已提交的阶段结果可续跑。',flush=True)
    finally:
        deliver(out.parent/(out.name+'_results.zip'),AUTO_DOWNLOAD_FINAL)
        deliver(out.parent/(out.name+'_diagnostics_only.zip'),False)
        deliver(log,AUTO_DOWNLOAD_FINAL)
        print('请确认浏览器下载完成。完整原始数据以 results.zip 为准。',flush=True)


if __name__=='__main__':
    launch()
