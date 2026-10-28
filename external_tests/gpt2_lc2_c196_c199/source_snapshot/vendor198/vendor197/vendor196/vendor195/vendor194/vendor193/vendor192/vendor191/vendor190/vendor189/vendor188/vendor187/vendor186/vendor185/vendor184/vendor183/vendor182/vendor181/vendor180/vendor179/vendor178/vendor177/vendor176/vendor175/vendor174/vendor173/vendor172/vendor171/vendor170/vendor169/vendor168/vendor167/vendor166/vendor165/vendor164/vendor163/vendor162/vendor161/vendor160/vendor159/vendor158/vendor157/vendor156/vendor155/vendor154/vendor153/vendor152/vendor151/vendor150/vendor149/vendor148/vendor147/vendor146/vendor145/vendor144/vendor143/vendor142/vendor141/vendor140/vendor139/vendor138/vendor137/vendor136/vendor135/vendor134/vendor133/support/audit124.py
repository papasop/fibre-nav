"""Independent C123 archival arithmetic. NumPy only; no fitting or archived-code execution."""
from pathlib import Path,PurePosixPath
from collections import Counter
from contextlib import ExitStack,contextmanager
import hashlib,io,json,math,time,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
PROTOCOL='GPT2_LC2_C124_READONLY_C123_AUDIT_CPU_V1'
PARENT='GPT2_LC2_C123_CENTER_COVERAGE_CPU_V1'
PARENT_DECISION='C123_FIXED_COVERAGE_DEVELOPMENT_COMPLETE_NO_CANDIDATE'
ARMS=('physical11','physical11_full_coordinates')
CASES=('xor_00','xor_01','xor_11')
RIDGE=.1;FLOOR=1e-6;BUDGET=.002;STRICT=.0018;TIMEOUT=1200

def need(ok,msg):
    if not ok:raise ValueError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)

def clean_json(raw):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'Duplicate JSON key');d[k]=v
        return d
    def bad(s):raise ValueError('Nonfinite JSON constant: '+s)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)

def trusted(name):return clean_json((ROOT/'trusted'/name).read_bytes())

def farray(v,shape):
    a=np.asarray(v);need(a.shape==shape and a.dtype.kind=='f' and a.dtype.itemsize in (4,8),'Floating shape/dtype differs: '+str(a.shape))
    need(np.isfinite(a).all(),'Nonfinite tensor');return a

def binary(v,n):
    a=np.asarray(v);need(a.shape==(n,) and a.dtype.kind in 'bifu','Preparation flag encoding differs')
    need(np.isfinite(a).all() and np.isin(a,[0,1]).all(),'Nonbinary preparation flag');return a.astype(bool)

def exact(x,y,msg):
    x,y=np.asarray(x),np.asarray(y)
    need(x.shape==y.shape and x.dtype==y.dtype and np.array_equal(x,y),msg)

def close(x,y,msg,tol=1e-10):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    need(x.shape==y.shape and x.size>0 and np.isfinite(x).all() and np.isfinite(y).all(),msg+' shape/nonfinite')
    e=float(np.max(abs(x-y)));need(e<=tol,msg+': '+str(e));return e

def compare(actual,expected,path='record',atol=1e-12,rtol=1e-10):
    """Counts, booleans, identities and dictionary coverage stay exact."""
    if isinstance(expected,dict):
        need(isinstance(actual,dict) and set(actual)==set(expected),path+' keys differ')
        for k in expected:compare(actual[k],expected[k],path+'/'+k,atol,rtol)
    elif isinstance(expected,list):
        need(isinstance(actual,list) and len(actual)==len(expected),path+' length differs')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'/'+str(i),atol,rtol)
    elif isinstance(expected,bool):need(type(actual) is bool and actual==expected,path+' boolean differs')
    elif isinstance(expected,int):need(type(actual) is int and actual==expected,path+' integer differs')
    elif isinstance(expected,float):
        need(type(actual) in (int,float) and np.isfinite(actual) and np.isfinite(expected) and math.isclose(actual,expected,abs_tol=atol,rel_tol=rtol),path+' scalar differs')
    else:need(actual==expected,path+' identity differs')

def tensor_hash(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def row_digest(v):
    a=np.ascontiguousarray(v);header=json.dumps(dict(dtype=a.dtype.str,shape=list(a.shape)),sort_keys=True,separators=(',',':')).encode('ascii')
    return hashlib.sha256(header+b'\0'+a.tobytes()).hexdigest()

def head_digest(d):
    h=hashlib.sha256()
    for k in sorted(d):
        a=np.ascontiguousarray(d[k]);need(a.dtype.kind in 'fiu' and np.isfinite(a).all(),'Invalid head tensor')
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def check_directory(z):
    items=z.infolist();ns=[i.filename for i in items]
    need(len(ns)==len(set(ns)) and len(ns)<100000,'Duplicate/oversized archive directory')
    for n in ns:
        p=PurePosixPath(n);need(not p.is_absolute() and '..' not in p.parts and '\\' not in n,'Unsafe ZIP path')
    need(sum(i.file_size for i in items)<3*1024**3,'Archive uncompressed-size budget exceeded')

class Archive:
    def __init__(self,path):
        self.z=zipfile.ZipFile(path);check_directory(self.z)
    def close(self):self.z.close()
    def raw(self,name):
        need(self.z.getinfo(name).file_size<=512*1024**2,'Archive member too large');return self.z.read(name)
    def json(self,name):return clean_json(self.raw(name))
    def arrays(self,name):
        raw=self.raw(name)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:check_directory(z)
        with np.load(io.BytesIO(raw),allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        need(all(v.dtype.kind in 'fbiu' and np.isfinite(v).all() for v in d.values()),'NPZ object/string/nonfinite tensor')
        return d
    def verify_manifest(self):
        manifest=self.json('RESULT_MANIFEST.json');need(set(self.z.namelist())==set(manifest)|{'RESULT_MANIFEST.json'},'Result manifest inventory differs')
        for n,h in manifest.items():need(hashlib.sha256(self.raw(n)).hexdigest()==h,'Result member hash differs: '+n)
        return len(manifest)

@contextmanager
def no_solvers():
    names=('solve','inv','pinv','lstsq','svd','eig','eigh','cholesky');original={n:getattr(np.linalg,n) for n in names}
    def blocked(*args,**kwargs):raise RuntimeError('C124 forbids fitting, solving and decompositions')
    try:
        for n in names:setattr(np.linalg,n,blocked)
        yield
    finally:
        for n,v in original.items():setattr(np.linalg,n,v)

def center(sid):return sid.split('__n',1)[0]

def new_specs():
    out=[]
    for ci,c in enumerate(CASES):
        for role,count,offset in [('training',6,0),('development',2,500000)]:
            for j in range(count):
                sid=f'{c}__c123_{role}_center{j:02d}';seed=323123001+ci*10000+offset+j*1000
                base=dict(case=c,cohort='c123_cpu_'+role,dataset_role=role,group_id=sid)
                out.append(dict(base,id=sid,role=role+'_center',parent_id=c+'__original_training_center',norm=(.75,1.25)[j%2],seed=seed))
                for k,r in enumerate((.125,.375)):
                    for d in range(2):out.append(dict(base,id=f'{sid}__n{k:02d}_d{d:02d}',role=role+'_child',parent_id=sid,norm=r,seed=seed+100+10*k+d))
    return out

def fold_registry(old_ids):
    need(len(old_ids)==len(set(old_ids))==120,'Canonical120 IDs required');groups=[center(sid) for sid in old_ids]
    cs=sorted(set(groups));need(len(cs)==24 and set(Counter(groups).values())=={5},'Whole-center grouping differs')
    out=[dict(name='full210',held_center=None,train_indices=list(range(210)),validation_indices=list(range(210)))]
    for i,c in enumerate(cs):
        v=[j for j in range(120) if groups[j]==c];t=[j for j in range(210) if j>=120 or groups[j]!=c]
        need(len(t)==205 and len(v)==5 and not set(t)&set(v),'Fold train/validation overlap')
        out.append(dict(name=f'outer{i:02d}',held_center=c,train_indices=t,validation_indices=v))
    return out

def target_bits(case,write):
    need(case in CASES and type(write) is bool,'Excluded case or invalid write flag')
    x,y=map(int,case[-2:]);return [x,y,1,(x^y) if write else 1-(x^y)]

def metric_gates(m,case,theta,write,preparation_valid=True):
    o=farray(np.asarray(m['observed'],float),(22,));kl=m['KL']
    need(type(kl) in (int,float) and np.isfinite(kl) and kl>=0,'Invalid archived KL scalar')
    need(m['theta_sha256']==tensor_hash(np.asarray(theta,np.float32)),'Metric coordinate identity differs')
    truth=target_bits(case,write);bits=[int(x) for x in o[6:10]>=0];margins=(2*np.array(truth*4)-1)*o[6:]
    drift=float(np.max(abs(o[:6])));compare(m['bits'],bits);close(m['exact_margins'],margins[:4],'Exact margins',1e-10);close(m['development_margins'],margins[4:],'Development margins',1e-10)
    need(abs(m['response_drift']-drift)<=1e-12,'Response scalar differs')
    gates=dict(bits_match=bits==truth,response=drift<=BUDGET,KL=kl<=.005,exact_read_margin=bool(min(margins[:4])>=.05),development_read_margin=bool(min(margins[4:])>=.05))
    if write:gates.update(output_flip=bits[3]!=target_bits(case,False)[3],unselected_preserved=bits[:3]==target_bits(case,False)[:3])
    compare(m['gates'],gates);compare(m['pass'],all(gates.values()))
    if not write:return bool(all(gates.values()))
    order=('bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin')
    return dict(complete_pass=bool(preparation_valid and all(gates.values())),failed_gates=([] if preparation_valid else ['preparation'])+[k for k in order if not gates[k]],response_drift=drift,response_residual=o[:6].tolist(),KL=kl,minimum_exact_margin=float(min(margins[:4])),minimum_development_margin=float(min(margins[4:])))

def repeat_comparison(m,first):
    e=float(np.max(abs(np.asarray(m['observed'])-np.asarray(first['observed']))));k=abs(m['KL']-first['KL']);same=m['gates']==first['gates']
    return dict(observed_error_Linf=e,KL_absolute_error=k,gate_identity=same,same_point_identity=m['theta_sha256']==first['theta_sha256'],within_replay_tolerance=e<=.0002 and k<=5e-6 and same)

def reconstruct_raw(writer,sketch,theta,capture):
    farray(theta,(36864,));farray(capture,(3147,));farray(sketch,(36864,64))
    zz=(capture-writer['mean'])/writer['scale']
    delta=writer['delta_mean']+(zz@writer['training_z'].T/3147)@writer['coefficients']
    delta*=min(1.,20./max(float(np.linalg.norm(delta)),1e-30))
    raw=(theta.astype(float)+delta).astype(np.float32);full=np.r_[capture,(raw.astype(float)-theta.astype(float))@sketch]
    need(np.isfinite(raw).all() and np.isfinite(full).all(),'Nonfinite raw reconstruction');return raw,full

def verify_capture(cap,zero,full):
    need(set(cap)=={'features','observed','anchor_logits32','anchor_mean32','response32'},'3147 capture inventory differs')
    x=farray(cap['features'],(3147,));o=farray(cap['observed'],(22,));close(x,full[:3147],'Captured3147 prefix',0)
    for ob in (cap,zero):
        farray(ob['anchor_logits32'],(12,6));farray(ob['anchor_mean32'],(6,))
        need(ob['anchor_logits32'].dtype==ob['anchor_mean32'].dtype==np.float32,'Expected captured float32 logits/means')
    farray(cap['response32'],(6,));need(cap['response32'].dtype==np.float32,'Expected float32 response')
    close(o[:6],cap['response32'].astype(float),'Response observable identity',0)
    close(np.subtract(cap['anchor_mean32'],zero['anchor_mean32'],dtype=np.float32),cap['response32'],'Original float32 mean subtraction',0)
    close(x[3077:3083],x[3077:3083].astype(np.float32).astype(float),'Original normalized response storage',0)

def statistics(r,p):
    need(r.shape==p.shape and r.ndim==2 and r.shape[1]==6 and len(r)>0 and np.isfinite(r).all() and np.isfinite(p).all(),'Prediction/label geometry')
    error=r-p;maximum=np.max(abs(error),axis=1);inside=maximum<=BUDGET;raw=np.max(abs(r),axis=1)<=BUDGET
    return dict(n=len(r),prediction_within_budget=int(sum(inside)),prediction_within_strict_band=int(sum(maximum<=STRICT)),maximum_prediction_error_Linf=float(maximum.max()),global_RMSE=float(np.sqrt(np.mean(error*error))),coordinate_mean_signed_error=error.mean(0).tolist(),rescued_vs_raw_prediction_proxy=int(sum(~raw&inside)),destroyed_vs_raw_prediction_proxy=int(sum(raw&~inside)))

def ridge_equation(k,labels,alpha,bias,indices):
    t=np.asarray(indices);need(t.dtype.kind in 'iu' and t.ndim==1 and len(set(t.tolist()))==len(t) and np.all((0<=t)&(t<210)),'Illegal fit row index')
    n=len(t);farray(alpha,(n,6));farray(bias,(6,));need(labels.shape==(210,6),'Only210 training-label rows allowed')
    matrix=np.empty((n+1,n+1));matrix[:n,:n]=k[np.ix_(t,t)]+RIDGE*np.eye(n);matrix[:n,n]=1;matrix[n,:n]=1;matrix[n,n]=0
    rhs=np.vstack([labels[t],np.zeros((1,6))]);coef=np.vstack([alpha,bias]);rel=float(np.linalg.norm(matrix@coef-rhs)/max(float(np.linalg.norm(rhs)),1e-30))
    need(rel<=1e-8,'Frozen ridge equation residual exceeds1e-8');return rel

def geometry_and_kernels(blocks,indices,saved):
    n=len(blocks[0]);gp=np.zeros((n,n));gc=np.zeros((n,n));maxerr=0.
    need(set(saved)=={f'block{j}_{k}' for j in range(5) for k in ('mean','scale')},'Geometry tensor keys differ')
    for j,b in enumerate(blocks):
        mu=b[indices].mean(0);scale=np.maximum(b[indices].std(0,ddof=0),FLOOR)
        maxerr=max(maxerr,close(saved[f'block{j}_mean'],mu,'Own-fold mean',1e-12),close(saved[f'block{j}_scale'],scale,'Own-fold scale',1e-12))
        z=(b-mu)/scale/np.sqrt(b.shape[1]);g=z@z.T
        if j<3:gp+=(1/3)*g
        else:gc+=(1/2)*g
    kernels={ARMS[0]:(1.+gp)**2,ARMS[1]:(1.+(gp+gc))**2}
    need(all(np.isfinite(k).all() for k in kernels.values()),'Nonfinite reconstructed kernel');return kernels,maxerr
