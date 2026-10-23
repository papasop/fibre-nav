"""Run one unchanged scientific launcher in an isolated child; default is information only."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['C141','C142','C148','C150','C154','C155']);p.add_argument('--inputs-json');p.add_argument('--out');a=p.parse_args();row=json.loads((ROOT/'RUN_INDEX.json').read_text())[a.stage];src=(ROOT/row['source_root']).resolve();pins=json.loads((src/'INPUT_PINS.json').read_text());print(json.dumps(dict(stage=a.stage,source_root=row['source_root'],required_inputs=pins,real_model=a.stage!='C142',scope='Original protocol; CPU model runs can be long. Info-only unless --inputs-json and --out are provided.'),indent=2))
    if not a.inputs_json:return
    if not a.out:raise ValueError('--out is required to execute, not the repository source tree')
    inputs=json.loads(Path(a.inputs_json).read_text());need=set(inputs)==set(pins)
    if not need:raise ValueError('Exact predecessor key inventory required')
    for key,path in inputs.items():
        f=Path(path)
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=pins[key]['sha256']:raise ValueError('Missing/wrong original '+key)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2');code='import sys,importlib,json;sys.path.insert(0,sys.argv[1]);m=importlib.import_module(sys.argv[2]);m.start(False,json.loads(sys.argv[3]),False,sys.argv[4])';subprocess.run([sys.executable,'-I','-u','-c',code,str(src),row['launch_module'],json.dumps(inputs),str(Path(a.out).resolve())],check=True,env=env)
if __name__=='__main__':main()
