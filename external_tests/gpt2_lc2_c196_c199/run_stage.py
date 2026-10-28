"""Inspect stages by default; explicitly supplied inputs/base run original launcher."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['C196','C197','C198','C199'],required=True);p.add_argument('--inputs-json',type=Path);p.add_argument('--base',type=Path);a=p.parse_args();index=json.loads((ROOT/'RUN_INDEX.json').read_text());entry=index[a.stage];source=ROOT/entry['source_root'];pins=json.loads((source/'INPUT_PINS.json').read_text())
 if bool(a.inputs_json)!=bool(a.base):p.error('--inputs-json and --base are required together')
 if a.inputs_json is None:
  print(json.dumps(dict(stage=a.stage,**entry,required_inputs=pins,will_execute_model=False),indent=2));return 0
 paths=json.loads(a.inputs_json.read_text());
 if set(paths)!=set(pins):p.error('Every original input key is required, without extra keys')
 for key,path in paths.items():
  f=Path(path).expanduser().resolve()
  if not f.is_file():p.error('Missing original '+key)
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
  if h.hexdigest()!=pins[key]['sha256']:p.error('Wrong original hash '+key)
  paths[key]=str(f)
 a.base.mkdir(parents=True,exist_ok=True);config=a.base/'INPUT_PATHS_FOR_ORIGINAL_LAUNCHER.json';config.write_text(json.dumps(paths,indent=2)+'\n')
 boot='import sys,json,importlib;sys.path.insert(0,sys.argv[1]);m=importlib.import_module(sys.argv[2]);m.start(False,json.load(open(sys.argv[3])),False,sys.argv[4])'
 return subprocess.run([sys.executable,'-I','-c',boot,str(source),'launch'+a.stage[1:],str(config.resolve()),str(a.base.resolve())],check=False).returncode
if __name__=='__main__':raise SystemExit(main())
