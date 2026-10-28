"""Full-size analytic C155 archive for audit tests only, never real model evidence."""
import copy,functools,io,shutil,tempfile,zipfile
from core156 import *
from fixture155 import inputs_fixture as old_inputs,bound,Native,Pair
from worker155 import run as parent_run
from inputs152 import emit
_TEMP=[]
def pack(folder,path):
    dump(folder/'RESULT_MANIFEST.json',{p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=folder/'RESULT_MANIFEST.json'})
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder).as_posix())
@functools.lru_cache(None)
def fixture():
    temp=tempfile.TemporaryDirectory();_TEMP.append(temp);root=Path(temp.name);paths,pc=old_inputs();output=root/'parent'
    def loader(pp,out,cc):
        d,h,p,r,co=copy.deepcopy(bound());d['modelpins']=dict(d['modelpins'],response_ids=list(range(6)));out=Path(out);parent_out=out/'PARENT_BINDING';parent_out.mkdir(parents=True);emit(parent_out/'RECOMPUTED_C153/BOUND_C152_INPUTS',d,pp,{'analytic_fixture':True});np.savez_compressed(parent_out/'ARCHIVED_ANCHORS_AND_DIRECTIONS.npz',**{a+'__'+k:v[k] for a,v in h.items() for k in ('before','after','gradient')});dump(parent_out/'ARCHIVED_ANCHOR_RECORDS.json',{a:{k:v[k] for k in ('step','aggregate','rows','plus_references','zero_native_references')} for a,v in h.items()});(out/'RECONSTRUCTED_C154').mkdir();np.savez_compressed(out/'RECONSTRUCTED_C154/STENCIL_PARAMETERS.npz',**co);return d,h,p,r,co
    def native(out,d,phase,calls):
        rt=Native(out,d,phase,calls);rt.out=Path(out);rt.out.mkdir(parents=True,exist_ok=True);dump(rt.out/'MODEL_PINS.json',d['modelpins']);return rt
    def pair(rt,calls):
        p=Pair(rt,calls);offset=np.arange(6)*1e-7;a=np.zeros((12,16),np.float64);b=a.copy();b[:,:6]=offset;np.savez_compressed(rt.out/'FP64_BASELINE_LOGITS.npz',FP32_logits_cast64=a,FP64_logits=b);dump(rt.out/'FP64_BASELINE_PROVENANCE.json',dict(float32_parameter_values_cast_exactly_to_float64=True,FP64_chart_source='original all-zero float32 chart cast to float64',FP32_response_reference=[0.]*6,FP64_response_reference=offset.tolist(),response_reference_offset=offset.tolist(),FP32_logits_recomputed_in_FP64_for_cross_KL=True,FP64_is_not_ground_truth=True,precision_diagnostic_only=True));return p
    with contextlib.redirect_stdout(io.StringIO()):s=parent_run(paths,output,pc,native,pair,loader)
    need(s['complete'],'Synthetic C155 producer must complete: '+str(s['errors']));dump(output/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});shutil.copytree(ROOT/'vendor155',output/'source',ignore=shutil.ignore_patterns('__pycache__'));archive=root/'C155.zip';pack(output,archive);c=contract();c.update(production=False,pins={'C155':dict(path=str(archive),sha256=sha(archive))});return archive,c
def changed_archive(name,change,rebind_seals=False):
    archive,c=fixture();tmp=tempfile.TemporaryDirectory();_TEMP.append(tmp);path=Path(tmp.name)/'changed.zip'
    with zipfile.ZipFile(archive) as z:members={n:z.read(n) for n in z.namelist()}
    members[name]=change(members[name])
    if rebind_seals:
        seal=clean_json(members['SEALED_BEFORE_MODEL.json'])
        if name in seal['artifact_seals']:seal['artifact_seals'][name]=hashlib.sha256(members[name]).hexdigest();members['SEALED_BEFORE_MODEL.json']=(json.dumps(seal)+'\n').encode()
    m={n:hashlib.sha256(v).hexdigest() for n,v in members.items() if n!='RESULT_MANIFEST.json'};members['RESULT_MANIFEST.json']=(json.dumps(m)+'\n').encode()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in members.items():z.writestr(n,v)
    c=copy.deepcopy(c);c['pins']['C155']['sha256']=sha(path);return path,c
def json_change(fn):
    def mutate(b):o=clean_json(b);fn(o);return (json.dumps(o,allow_nan=False)+'\n').encode()
    return mutate
def array_change(fn):
    def mutate(b):
        with np.load(io.BytesIO(b),allow_pickle=False) as f:o={k:f[k] for k in f.files}
        fn(o);s=io.BytesIO();np.savez_compressed(s,**o);return s.getvalue()
    return mutate
