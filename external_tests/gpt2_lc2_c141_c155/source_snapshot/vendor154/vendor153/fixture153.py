"""Analytic synthetic C152 archive only; never real GPT2 evidence."""
import copy,functools,shutil,tempfile,zipfile,contextlib,io
from core153 import *
from fixture152 import dummy_inputs,loader,Native,Analytic
from worker152 import run as synthetic_train
_CACHE=[]

def pack(folder,path):
    dump(folder/'RESULT_MANIFEST.json',{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'RESULT_MANIFEST.json'})
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder).as_posix())

@functools.lru_cache(None)
def base_fixture(native_mode='normal'):
    tmp=tempfile.TemporaryDirectory();_CACHE.append(tmp);root=Path(tmp.name);paths,pc=dummy_inputs(root/'inputs');out=root/'C152'
    with contextlib.redirect_stdout(io.StringIO()):s=synthetic_train(paths,out,pc,lambda o,d,p,cs:Native(o,d,p,cs,native_mode),Analytic,loader)
    need(s['complete'],'Synthetic C152 workflow failed: '+repr(s['errors']));shutil.copytree(ROOT/'vendor152',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',{k:True for k in parent.INPUT_KEYS});archive=root/'C152.zip';pack(out,archive);paths=dict(paths,C152=archive);c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},binding152=pc);return root,paths,c

def mutate(root,member,fn,rebind_seals=False):
    _,paths,c=base_fixture();paths=dict(paths);c=copy.deepcopy(c);target=Path(root)/'changed_C152.zip'
    with zipfile.ZipFile(paths['C152']) as z:payload={n:z.read(n) for n in z.namelist()}
    payload[member]=fn(payload[member])
    if member.endswith('/OBSERVATIONS.json'):payload[member.replace('/OBSERVATIONS.json','/GRADIENT_PROGRESS.json')]=payload[member]
    if member=='ALL672_FINAL_TRAINING_ENDPOINTS.json':payload['FINAL_ENDPOINT_PROGRESS.json']=payload[member]
    if rebind_seals:
        seal=clean_json(payload['BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json'])
        if member in seal['artifact_seals']:seal['artifact_seals'][member]=hashlib.sha256(payload[member]).hexdigest()
        payload['BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json']=(json.dumps(seal,ensure_ascii=False,indent=2)+'\n').encode()
    manifest={n:hashlib.sha256(b).hexdigest() for n,b in payload.items() if n!='RESULT_MANIFEST.json'};payload['RESULT_MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in payload.items():z.writestr(n,b)
    paths['C152']=target;c['pins']['C152']['sha256']=sha(target);return paths,c

def json_change(fn):
    def transform(b):
        obj=clean_json(b);fn(obj);return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    return transform

def npz_change(fn):
    def transform(b):
        with np.load(io.BytesIO(b),allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
        fn(arrays);out=io.BytesIO();np.savez_compressed(out,**arrays);return out.getvalue()
    return transform
