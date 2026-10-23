"""Separate analytic worker fixture and prior archive fixtures; no model claims."""
import contextlib,copy,io,shutil
from core138 import *
from fixture137 import make_fixture as fixture137,Analytic,Toy as PriorToy
from fixture133 import pack
from worker137 import run as run137

def metric(theta,case,write):
    p=theta.astype(float);o=np.r_[p[:6],np.tile(p[6:10],4)];truth=safe.target_bits(case,write);bits=[int(x) for x in o[6:10]>=0];margins=(2*np.array(truth*4)-1)*o[6:];kl=float(np.mean(p[:6]**2)/2);drift=float(max(abs(p[:6])))
    g=dict(bits_match=bits==truth,response=drift<=.002,KL=kl<=.005,exact_read_margin=bool(min(margins[:4])>=.05),development_read_margin=bool(min(margins[4:])>=.05))
    if write:g.update(output_flip=bits[3]!=safe.target_bits(case,False)[3],unselected_preserved=bits[:3]==safe.target_bits(case,False)[:3])
    return dict(observed=o.tolist(),bits=bits,response_drift=drift,KL=kl,exact_margins=margins[:4].tolist(),development_margins=margins[4:].tolist(),per_anchor_KL=[kl]*12,gates=g,**{'pass':all(g.values())},theta_sha256=safe.tensor_hash(theta))

def capture(theta,desired):return np.r_[theta[:3072].astype(float),theta[6:10].astype(float),float(desired),theta[:6].astype(float)/.002,np.zeros(64)]

def worker_data():
    w=dict(mean=np.zeros(3147),scale=np.ones(3147),training_z=np.zeros((1,3147)),coefficients=np.zeros((1,DIM)),delta_mean=np.zeros(DIM));w['training_z'][0,3076]=3147;w['coefficients'][0,9]=1.4;w['delta_mean'][9]=-.7;w['delta_mean'][0]=.001
    v=np.zeros((6,DIM));v[:6,:6]=np.eye(6);a=np.full(6,-.06);control=dict(delta_sketch=np.zeros((DIM,64)));points={};cpoints={};controls={};known=set()
    for i,sp in enumerate(parent.registry()):
        sid=sp['state_id'];case=sp['case'];theta=np.zeros(DIM,np.float32);theta[6:10]=(2*np.array(safe.target_bits(case,False))-1)*np.float32(.35);theta[100]=(i+1)*.01
        raw,full=safe.reconstruct_raw(w,control['delta_sketch'],theta,capture(theta,int(case[-2])^int(case[-1])));cp=parent.proposal(theta,raw,v,a)[0]
        points[sid]=dict(spec=sp,prepared=theta,raw=raw,features=full,preparation=metric(theta,case,False));cpoints[sid]=cp
        for rnd in (0,1):
            for arm,p in ((parent.EVAL_ARMS[0],raw),(parent.EVAL_ARMS[1],cp)):controls[(sid,arm,rnd)]=dict(metrics=metric(p,case,True))
        known|={safe.tensor_hash(theta),safe.tensor_hash(raw),safe.tensor_hash(cp)}
    return dict(points=points,a=a,V=v,writer=w,control=control,C137_points=cpoints,C137_controls=controls,C137_summary=dict(decision='C137_FIXED_FINAL_ALL12_TRAINING_GATES_PASSED'),historical_hashes=known,modelpins=dict(revision='SYNTHETIC_LINEAR_NOT_GPT2',device='cpu',dtype='float32',torch='fixture'))

def dummy_inputs(root):
    paths={};pins={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('synthetic unit input '+k).encode());paths[k]=p;pins[k]=dict(path=str(p),sha256=sha(p))
    return paths,dict(contract(),production=False,pins=pins)

class Toy:
    def __init__(self,out,data,phase,calls,mode='normal'):
        self.phase=phase;self.calls=calls;self.data=data;self.pins=dict(data['modelpins']);self.mode=mode;self.measurements=0;calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
    def features(self,point,desired):
        self.phase.consume_feature(point,desired);self.calls['feature_attempts']+=1;self.calls['feature_calls']+=1;self.calls['model_forwards']+=3;x=capture(point,desired)
        if self.mode=='feature_drift' and self.phase.phase=='controls':x[0]+=.001
        case=next(sp['case'] for sp in parent.registry() if (int(sp['case'][-2])^int(sp['case'][-1]))==desired and np.array_equal((np.array(safe.target_bits(sp['case'],False))[:2]),(point[6:8]>=0).astype(int)))
        return x,np.array(metric(point,case,False)['observed'])
    def metrics(self,point,case,write=True):
        self.phase.consume(point,case,write);self.calls['metric_attempts']+=1
        if self.mode=='mutate_candidate' and self.phase.phase=='prewrite':self.data['a'][0]+=.01
        if self.mode=='interrupt' and self.phase.phase=='evaluation' and self.measurements==5:raise RuntimeError('Synthetic interrupted native endpoint')
        self.calls['metric_calls']+=1;self.calls['model_forwards']+=2
        if self.phase.phase=='evaluation':self.measurements+=1
        m=metric(point,case,write)
        if self.mode=='control_drift' and self.phase.phase=='controls':m['observed'][0]+=.001;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        if self.mode=='repeat_drift' and self.phase.phase=='evaluation' and self.measurements>48:m['observed'][0]+=.0003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        return m
    def invariant(self):pass

def prior_archive_fixture(root):
    paths,c=fixture137(root/'parents');out=root/'C137'
    with contextlib.redirect_stdout(io.StringIO()):s=run137(paths,out,c,PriorToy,Analytic)
    need(s['complete'] and s['fixed_final_training_qualification_passed'],'Synthetic C137 archive generation failed')
    shutil.copytree(ROOT/'vendor137',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p=pack(out,root/'C137_SYNTHETIC.zip')
    c=dict(c,pins=dict(c['pins'],C137=dict(path=str(p),sha256=sha(p))),source137=contract()['source137'])
    return dict(paths,C137=p),c
