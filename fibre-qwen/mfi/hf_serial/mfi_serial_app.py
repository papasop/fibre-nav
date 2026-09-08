"""Manual HF CPU evaluation console. No model load or run at startup."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
import zipfile

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'mfi_serial_config.json').read_text())
ARCHIVE = HERE/'mfi_serial_frozen.zip'
if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != CONFIG['archive_sha256']:
    raise RuntimeError('Frozen archive checksum mismatch')
ROOT = Path(tempfile.mkdtemp(prefix='mfi-task-'))
with zipfile.ZipFile(ARCHIVE) as z:
    for name in z.namelist():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0] != 'mfi':
            raise RuntimeError('Invalid archive path')
    z.extractall(ROOT)
SOURCE = ROOT/'mfi'
REFERENCE = None
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
STATE = {'status':'idle', 'origin':'连续改写验证；本版本尚未运行云端测试',
         'report':REFERENCE, 'stages':[], 'log':'', 'download':False}
BUNDLE = ROOT/'latest_run.zip'


def collect_stages(out):
    return [json.loads(p.read_text()) for p in sorted(out.glob('84*.json'))]


def run_job(new_seeds=False):
    out = ROOT/f'run_{time.time_ns()}'
    logpath = ROOT/'latest.log'
    try:
        with logpath.open('w') as log:
            completed = subprocess.run([sys.executable, str(SOURCE/'evaluate_serial.py'),
                '--out', str(out)] + (['--new-seeds'] if new_seeds else []), cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_DISABLE_TELEMETRY='1'),
                timeout=CONFIG['timeout_seconds'])
        rc = completed.returncode
        status = 'completed' if rc in (0,2) and (out/'report.json').exists() else 'error'
    except subprocess.TimeoutExpired:
        rc, status = 124, 'timeout'
    except Exception as exc:
        rc, status = -1, 'error'
        logpath.write_text(type(exc).__name__ + ': ' + str(exc))
    try:
        report = json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else None
        stages = collect_stages(out) if out.exists() else []
        logtext = logpath.read_text(errors='replace')[-20000:]
        with zipfile.ZipFile(BUNDLE,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')) if out.exists() else []:
                if p.is_file(): z.write(p, 'run/'+p.relative_to(out).as_posix())
            z.write(logpath,'run/console.log')
            z.writestr('deployment_config.json',json.dumps(CONFIG,indent=2))
            z.writestr('execution_status.json',json.dumps({'status':status,'returncode':rc}))
        with LOCK:
            STATE.update(status=status, returncode=rc, report=report, stages=stages,
                         log=logtext, download=True)
    except Exception as exc:
        with LOCK:
            STATE.update(status='error', log='Result collection failed: '+str(exc), download=False)


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MFI · 独立任务评估</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f8;color:#152438;margin:0}main{max-width:1000px;margin:40px auto;padding:0 22px}h1{font-size:30px;margin-bottom:4px}.sub{color:#536477}.panel{background:white;border:1px solid #dde4ec;border-radius:14px;padding:22px;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:#edf3fa;padding:18px;border-radius:10px}.value{font-size:28px;font-weight:700}button,a.download{background:#194f9e;color:white;padding:12px 18px;border:0;border-radius:8px;font:inherit;cursor:pointer;text-decoration:none}button:disabled{opacity:.5;cursor:wait}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #e5eaf0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;max-height:450px;overflow:auto}.pill{display:inline-block;padding:3px 10px;background:#e6edf7;border-radius:15px}#download{margin-left:12px}@media(max-width:650px){.cards{grid-template-columns:1fr}main{margin:20px auto}}
</style><main><span class="pill">研究测试 · L1</span><h1>MFI · 连续 20 次改写</h1><p class="sub">代理控制器 → MFI 接口 → 受限 LoRA 状态 → 响应/KL 验证 → 任务评分</p>
<section class="panel"><strong id="origin"></strong><p id="status"></p><button id="run">运行种子 84503</button><button id="run-new">运行种子 84521</button><a id="download" class="download" href="/download" hidden>下载本次记录</a><p class="sub">点击后才加载 Qwen3-0.6B。每组最长 30 分钟，含首次模型下载；超时不算科学结论。刷新页面不会重新开始。</p></section>
<div class="cards" id="cards"></div><section class="panel"><h2>操作与任务门槛</h2><p id="operations"></p><div id="gates"></div><p class="sub">响应偏移 ≤ 0.02 · KL ≤ 0.01 · margin ≥ 1.0。失败与跳过项计入完整流程分母。</p></section>
<section class="panel"><h2>操作记录</h2><table><thead><tr><th>种子</th><th>序号</th><th>操作</th><th>结果</th><th>回滚</th></tr></thead><tbody id="stages"></tbody></table><details><summary>原始指标与预测</summary><pre id="raw"></pre></details><details><summary>运行日志</summary><pre id="log"></pre></details></section>
<p class="sub">任务仅测试“简短/详细”格式选择，使用二 token 读出；不代表开放式回答质量或自主策略。四个新请求及标签互换是相关探针；不是开放式回答测试。20 次操作始终使用最初响应/KL 基准；失败回滚并跳过后续操作。外部记忆为理想正确记忆基线。记录保存在临时磁盘，请在重启前下载。</p></main><script>
const endpoint=path=>path+window.location.search; document.querySelector('#download').href=endpoint('/download'); const token='__TOKEN__'; const names={mfi:'MFI 参数记忆',external:'外部记忆',none:'无记忆'};const gateNames={all_operations_pass:'全部操作通过',task_beats_no_memory:'任务优于无记忆',task_matches_external:'任务不低于外部记忆'};
async function refresh(){const res=await fetch(endpoint('/status'),{cache:'no-store'});if(!res.ok)throw Error('无法读取状态');const s=await res.json();document.querySelector('#origin').textContent=s.origin;document.querySelector('#status').textContent='状态：'+s.status+(s.returncode===undefined?'':' · 退出码 '+s.returncode);document.querySelector('#run').disabled=s.status==='running';document.querySelector('#run-new').disabled=s.status==='running';document.querySelector('#download').hidden=!s.download;
const cards=document.querySelector('#cards');cards.replaceChildren();for(const [arm,label] of Object.entries(names)){const c=document.createElement('div');c.className='card';const t=document.createElement('div');t.textContent=label;const v=document.createElement('div');v.className='value';const score=s.report?.scores?.[arm];v.textContent=score?`${score.correct}/${score.total} · ${(100*score.accuracy).toFixed(1)}%`:'等待结果';c.append(t,v);cards.append(c)}
document.querySelector('#operations').textContent=s.report?`通过 ${s.report.operation_pass_count}/${s.report.operation_total} 个计划阶段`:'完整报告尚未生成';const g=document.querySelector('#gates');g.replaceChildren();for(const [k,v] of Object.entries(s.report?.gates||{})){const p=document.createElement('p');p.textContent=(v?'✓ ':'✗ ')+(gateNames[k]||k);g.append(p)}const body=document.querySelector('#stages');body.replaceChildren();for(const r of s.stages){const tr=document.createElement('tr');for(const value of [r.seed,r.index,r.operation,r.result?(r.operation_pass?'通过':'未通过'):(r.error?'异常':'跳过'),r.result?String(r.result.rolled_back):'—']){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}document.querySelector('#raw').textContent=JSON.stringify(s.stages,null,2);document.querySelector('#log').textContent=s.log||'暂无新运行日志';}
async function startRun(path){document.querySelector('#run').disabled=true;document.querySelector('#run-new').disabled=true;try{const r=await fetch(endpoint(path),{method:'POST',headers:{'X-MFI-Token':token}});if(!r.ok)throw Error(await r.text());await refresh()}catch(e){document.querySelector('#status').textContent=String(e);document.querySelector('#run').disabled=false;document.querySelector('#run-new').disabled=false}}document.querySelector('#run').onclick=()=>startRun('/run');document.querySelector('#run-new').onclick=()=>startRun('/run-heldout');refresh().catch(e=>document.querySelector('#status').textContent=String(e));setInterval(()=>refresh().catch(()=>{}),3000);
</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # HF private iframe requests may carry signed credentials in the query.
        pass
    def send(self, code, body, content_type='application/json'):
        self.send_response(code); self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/': return self.send(200,PAGE.replace('__TOKEN__',TOKEN).encode(),'text/html; charset=utf-8')
        if path=='/status':
            with LOCK: payload=json.dumps(STATE).encode()
            return self.send(200,payload)
        if path=='/download':
            with LOCK:
                if not STATE['download'] or STATE['status']=='running':return self.send(409,b'No completed archive')
                content=BUNDLE.read_bytes()
            return self.send(200,content,'application/zip')
        return self.send(404,b'Not found')
    def do_POST(self):
        if urlsplit(self.path).path not in ('/run','/run-heldout'):return self.send(404,b'Not found')
        if not secrets.compare_digest(self.headers.get('X-MFI-Token',''),TOKEN):return self.send(403,b'Invalid request token')
        with LOCK:
            if STATE['status']=='running':return self.send(409,b'An evaluation is already running')
            STATE.clear();STATE.update(status='running',origin=('连续 20 次改写：种子 84521' if urlsplit(self.path).path=='/run-heldout' else '连续 20 次改写：种子 84503'),report=None,stages=[],log='',download=False)
            threading.Thread(target=run_job,args=(urlsplit(self.path).path=='/run-heldout',),daemon=True).start()
        return self.send(202,b'{"status":"running"}')

if __name__=='__main__':
    STATE['stages']=collect_stages(SOURCE/'reference')
    ThreadingHTTPServer(('0.0.0.0',int(os.environ.get('PORT','7860'))),Handler).serve_forever()
