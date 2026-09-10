import json,os
from pathlib import Path
class PauseRun(Exception):pass

def atomic_json(path,data):
 path=Path(path);tmp=path.with_suffix('.tmp')
 with tmp.open('w') as f:
  json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(path)
