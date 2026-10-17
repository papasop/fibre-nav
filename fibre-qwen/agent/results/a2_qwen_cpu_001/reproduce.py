import sys,json,time,hashlib,importlib.metadata
from pathlib import Path
agent_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(agent_root))
import evaluate
import torch
torch.set_num_threads(4)
root=agent_root
out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path('a2-reproduction').resolve()
if out.exists(): raise SystemExit('Choose a new output directory.')
out.mkdir(parents=True,exist_ok=True)
model=evaluate.LocalQwen('Qwen/Qwen3-0.6B','c1899de289a04d12100db370d81485cdf75e47ca','cpu')
class Logged:
 def reply(self,messages):
  t=time.monotonic();answer=model.reply(messages)
  with (out/'model_calls.jsonl').open('a') as f:
   f.write(json.dumps({'messages':messages,'reply':answer,'seconds':time.monotonic()-t})+'\n')
  print('model call completed',round(time.monotonic()-t,2),flush=True)
  return answer
report=evaluate.evaluate(Logged())
report['model']={'backend':'qwen','id':'Qwen/Qwen3-0.6B','revision':model.revision,'device':'cpu','torch_threads':4}
report['versions']={n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors','huggingface-hub']}
report['code_sha256']={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['evaluate.py','assistant.py','research_agent.py']}
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['summary']),flush=True)
