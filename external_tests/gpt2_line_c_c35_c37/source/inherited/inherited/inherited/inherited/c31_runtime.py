"""Real GPT-2 adapter/chart runtime, unchanged writer and auditable preparation."""
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from raw_writer import StateWriter, prepared_input, route_from_features
from runtime_a100 import a100_device
from c27_features import ResidualWriter, tensor_hash, state_hash
from c27_protocol import PARENT_CP_SHA, REVISION
from c28_protocol import PARENT_MATRIX_SHA, PARENT_BRANCH_CP_SHA
from geometry import null_project
from correction_engine import run_correction
from c31_protocol import *


def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);os.replace(tmp,path)


def load(path):return torch.load(path,map_location='cpu',weights_only=True)


def vector(v):
    if isinstance(v,dict):v=v['theta']
    require(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and v.shape==(36864,) and bool(torch.isfinite(v).all()),'Finite original float32 memory chart required')
    return v.detach().cpu().clone()


def generate_sketch():
    matrix=torch.from_numpy((np.random.default_rng(SETTINGS['sketch_seed']).normal(size=(36864,64))/np.sqrt(36864)).astype(np.float32))
    require(tensor_hash(matrix)==SETTINGS['sketch_tensor_sha256'],'Original source sketch reconstruction differs')
    return matrix


class Runtime:
    def __init__(self,out,root):
        global w, audit
        import writer_core as w
        import run_l5_audit as audit
        self.out,self.root=Path(out),Path(root)
        self.device,self.gpu=a100_device(torch);torch.set_num_threads(min(4,os.cpu_count() or 1));w.set_seed(82931)
        import transformers,peft
        info={'torch':str(torch.__version__),'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':self.gpu}
        rf=self.out/'RUNTIME.json'
        if rf.exists():require(read(rf)==info,'Resume runtime differs')
        else:dump(rf,info)
        cfg=read(self.root/'inherited/inherited/inherited/legacy/config_audit.json');cfg.update(seed=82931,model_revision=REVISION)
        print('C31 Loading pinned GPT-2 once; C25+C27 writer immutable; prepare-only geometry, raw write endpoints.',flush=True)
        self.tok,self.model,self.params=w.make_model(cfg,self.device);self.model.config.use_cache=False
        for p in self.model.parameters():p.requires_grad_(False);p.grad=None
        pins=read(self.out/'parent/MODEL_PINS.json');ah=hashlib.sha256()
        for n,p in self.model.named_parameters():
            if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
        require(ah.hexdigest()==pins['lora_A_sha256'] and getattr(self.model.config,'_commit_hash',None)==REVISION,'Original model/A identity differs')
        require([list(p.shape) for p in self.params]==pins['chart_shapes'] and w.param_flat(self.params).abs().max()==0,'Original zero-B chart differs')
        symbols=w.single_token_pair(self.tok);self.readids=[s[1] for s in symbols]
        require(json.loads(json.dumps(symbols))==pins['symbols'],'Read tokens differ')
        rng=random.Random(82931);self.anchors=rng.sample(w.ANCHORS,cfg['anchor_count']);self.responseids=rng.sample(list(range(100,min(50000,self.tok.vocab_size))),cfg['response_dim'])
        _,self.exact,_,self.dev=audit.prompts()
        self.exact_batch=self.tok(self.exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(self.device)
        with torch.no_grad():self.logits0=w.next_logits(self.model,self.tok,self.anchors,self.device).detach();self.response0=self.logits0[:,self.responseids].mean(0)
        self.model_versions={n:p._version for n,p in self.model.named_parameters() if 'lora_B.default.' not in n}
        cp=load(self.out/'parent/C25_FINAL_CHECKPOINT.pt');branch=load(self.out/'parent/C27_BRANCH.pt')
        require(sha(self.out/'parent/C25_FINAL_CHECKPOINT.pt')==PARENT_CP_SHA and state_hash(cp['head'])==HEAD_SHA,'Actual C25 tensor identity differs')
        require(sha(self.out/'parent/C27_BRANCH.pt')==PARENT_BRANCH_CP_SHA and tensor_hash(branch['residual_matrix'])==PARENT_MATRIX_SHA,'Actual C27 branch identity differs')
        raw=StateWriter().to(self.device);raw.load_state_dict(cp['head']);raw.eval()
        self.writer=ResidualWriter(raw,load(self.out/'parent/C27_FEATURE_MAP.pt'),'state_bypass').to(self.device).eval()
        self.writer.set_matrix(branch['residual_matrix'])
        for p in self.writer.parameters():p.requires_grad_(False);p.grad=None
        self.writer_versions={n:p._version for n,p in self.writer.named_parameters()}
        norm=load(self.out/'parent/C9_CHECKPOINT.pt')['normalisation']
        require(float(norm['output_scale'])==SETTINGS['output_scale'],'Output scale differs')
        self.gnorm={k:v.to(self.device) for k,v in norm.items()};self.gextra={k:v.to(self.device) for k,v in cp['extra_norm'].items()}
        for normal,size in [(self.gnorm,3077),(self.gextra,70)]:
            require(all(k in normal and normal[k].shape==(size,) and bool(torch.isfinite(normal[k]).all()) for k in ('mean','scale')) and bool((normal['scale']>0).all()),'Invalid original normalisation')
        self.sketch=generate_sketch().to(self.device)
        save_tensor(self.out/'RECONSTRUCTED_STATE_SKETCH.pt',{'matrix':self.sketch.cpu(),'seed':615031,'source_recipe_verified':True,
                    'not_recovered_from_original_tensor_archive':True,'requires_historical_input_replay':True})
        fr={'C25_checkpoint_sha256':PARENT_CP_SHA,'C25_head_sha256':HEAD_SHA,'C27_checkpoint_sha256':PARENT_BRANCH_CP_SHA,
            'C27_matrix_sha256':PARENT_MATRIX_SHA,'sketch_tensor_sha256':tensor_hash(self.sketch),'frozen_before_new_state_generation':True}
        wf=self.out/'WRITER_FROZEN.json'
        if wf.exists():require(read(wf)==fr,'Frozen writer identity changed')
        else:dump(wf,fr)
        self.invariant()

    def sync(self):torch.cuda.synchronize(self.device)

    def invariant(self):
        require(all(p._version==self.model_versions[n] for n,p in self.model.named_parameters() if n in self.model_versions),'Frozen model/A changed')
        require(all(p._version==self.writer_versions[n] for n,p in self.writer.named_parameters()),'Frozen C27 writer parameters changed')
        require(tensor_hash(self.writer.matrix)==PARENT_MATRIX_SHA,'Frozen C27 matrix changed')
        require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()) and all(not p.requires_grad and p.grad is None for p in self.writer.parameters()),'Unexpected gradients')

    @torch.no_grad()
    def evaluate(self,theta,target,start=None):
        theta=vector(theta);w.assign_flat(self.params,theta.to(self.device))
        el=w.next_logits(self.model,self.tok,self.exact,self.device)[:,self.readids]
        dl=w.next_logits(self.model,self.tok,self.dev,self.device)[:,self.readids]
        al=w.next_logits(self.model,self.tok,self.anchors,self.device)
        bits=[int(v) for v in el.argmax(-1).cpu().tolist()]
        em=[float((r[b]-r[1-b]).cpu()) for r,b in zip(el,target)]
        dm=[float((r[b]-r[1-b]).cpu()) for r,b in zip(dl,target*3)]
        residual=al[:,self.responseids].mean(0)-self.response0;drift=float(residual.abs().max().cpu())
        kvals=(self.logits0.softmax(-1)*(self.logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1)
        kl=max(0.,float(kvals.mean().cpu()))
        gates={'bits_match':bits==target,'response':drift<=.002,'KL':kl<=.005,'exact_read_margin':min(em)>=.05,'development_read_margin':min(dm)>=.05}
        if start is not None:gates.update(output_flip=bits[3]!=start[3],unselected_preserved=bits[:3]==start[:3])
        self.invariant()
        return {'theta_sha256':tensor_hash(theta),'bits':bits,'evaluation':{'exact_margins':em,'heldout_margins':dm},
                'drift':drift,'kl':kl,'response_residual':residual.cpu().double().tolist(),'gates':gates,'pass':all(gates.values())}

    @torch.no_grad()
    def fresh_input(self,theta,desired):
        theta=vector(theta);self.sync();t0=time.perf_counter();w.assign_flat(self.params,theta.to(self.device))
        r=self.model(**self.exact_batch,use_cache=False,output_hidden_states=True)
        last=self.exact_batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=self.device)
        f=torch.cat([r.hidden_states[-1][rows,last].reshape(-1),r.logits[rows,last][:,self.readids][:,1]-r.logits[rows,last][:,self.readids][:,0],torch.tensor([desired],device=self.device,dtype=torch.float32)])
        response=w.response_vector(self.model,self.tok,self.anchors,self.responseids,self.device)
        extra=torch.cat([(response-self.response0)/.002,theta.to(self.device)@self.sketch])
        x=prepared_input(f,extra,self.gnorm,self.gextra);self.sync();self.invariant()
        return {'features':f.cpu(),'extras':extra.cpu(),'input':x.cpu(),'route':route_from_features(f).cpu(),'feature_seconds':time.perf_counter()-t0}

    @torch.no_grad()
    def raw_write(self,prep,inp):
        x=inp['input'].to(self.device);route=inp['route'].to(self.device)
        self.sync();t0=time.perf_counter();delta=self.writer(x.unsqueeze(0),route.reshape(1))[0]*SETTINGS['output_scale'];self.sync()
        theta=vector(prep)+delta.cpu();self.invariant()
        return theta,{'proposal_seconds':time.perf_counter()-t0,'proposal_count':1,'online_Jacobians':0,'online_SVD_calls':0,'post_write_correction_steps':0}

    @torch.no_grad()
    def measured(self,theta):
        w.assign_flat(self.params,theta.to(self.device));al=w.next_logits(self.model,self.tok,self.anchors,self.device)
        e=al[:,self.responseids].mean(0)-self.response0
        kl=max(0.,float((self.logits0.softmax(-1)*(self.logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1).mean().cpu()))
        return e.cpu().double().numpy(),float(e.abs().max().cpu()),kl

    def jacobian(self,theta,path):
        path=Path(path);h=tensor_hash(theta)
        if path.exists():
            g=load(path);require(g['theta_sha256']==h,'Saved preparation Jacobian belongs to a different state')
            require(g['J'].shape==(6,36864) and bool(torch.isfinite(g['J']).all()),'Invalid cached Jacobian')
            return g['J'].numpy()
        w.assign_flat(self.params,theta.to(self.device))
        for p in self.params:p.requires_grad_(True)
        try:
            with torch.enable_grad():
                response=w.response_vector(self.model,self.tok,self.anchors,self.responseids,self.device);rr=[]
                for k in range(6):
                    gg=torch.autograd.grad(response[k],self.params,retain_graph=k<5)
                    rr.append(torch.cat([v.reshape(-1) for v in gg]).detach().cpu().double())
            J=torch.stack(rr);save_tensor(path,{'J':J,'theta_sha256':h,'preparation_geometry_only':True})
        finally:
            for p in self.params:p.requires_grad_(False);p.grad=None
        self.invariant();return J.numpy()

    def correct_preparation(self,initial,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);cpfile=folder/'STATE.pt';initial_sha=tensor_hash(initial)
        resume=load(cpfile) if cpfile.exists() else None
        if resume:require(resume['initial_sha256']==initial_sha,'Preparation correction initial state differs')
        def persist(cp):save_tensor(cpfile,{**cp,'initial_sha256':initial_sha})
        def jac(theta):return self.jacobian(theta,folder/('J_'+tensor_hash(theta)+'.pt'))
        return run_correction(initial,self.measured,jac,lambda t,d:t+torch.from_numpy(d).float(),vector,persist,SETTINGS,resume,self.sync)


# These are the exact pinned identities used by C29, not newly inferred ones.
HEAD_SHA='4af097017d739e95cd143d69070028b3751be04f977c4f900394f6153574560f'
