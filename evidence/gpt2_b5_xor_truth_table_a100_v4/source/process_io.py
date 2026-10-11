import subprocess,sys,time,threading,queue,os,signal

def visible(cmd,label,timeout=None,heartbeat=20,log_path=None,check=True):
    print('START',label,flush=True)
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
    lines=queue.Queue()
    def read():
        try:
            for line in proc.stdout:lines.put(line)
        finally:lines.put(None)
    threading.Thread(target=read,daemon=True).start()
    start=last=time.monotonic();done=False
    log=open(log_path,'a',encoding='utf-8') if log_path else None
    def stop():
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    try:
        while not done or proc.poll() is None:
            now=time.monotonic()
            if timeout is not None and now-start>timeout:
                stop();raise TimeoutError(label+' exceeded '+str(timeout)+' seconds; subprocess stopped')
            try:
                line=lines.get(timeout=.2)
                if line is None:done=True
                else:
                    print(line,end='',flush=True)
                    if log:log.write(line);log.flush()
                    last=time.monotonic()
            except queue.Empty:pass
            if time.monotonic()-last>=heartbeat:
                print(label+'：子进程尚未退出，等待输出；已用 '+str(int(time.monotonic()-start))+' 秒。',flush=True)
                last=time.monotonic()
        code=proc.wait()
        print('END',label,'exit=',code,flush=True)
        if check and code:raise subprocess.CalledProcessError(code,cmd)
        return code
    except BaseException:
        stop();raise
    finally:
        if log:log.close()
        proc.stdout.close()

def refuse_existing_worker(base):
    from pathlib import Path
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(a).name=='worker_b4.py' for a in args) and any(a.startswith(str(base)+'/session_') for a in args):
            raise RuntimeError('Another B4 worker is already running: PID '+p.name+'. Do not start duplicate training.')
