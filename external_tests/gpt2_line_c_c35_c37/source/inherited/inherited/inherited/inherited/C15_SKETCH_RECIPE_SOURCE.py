"""Offline state expansion then frozen raw writer, with no online repair."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,time,zipfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
import c7_protocol as parent
from runtime_a100 import a100_device
from geometry import null_project
from proposer import WriterHead,normalise,one_forward_proposal
from correction_engine import run_correction
from c15_protocol import (PROTOCOL,RESULT,SETTINGS,CASES,STATES,REVISION,C6_SHA,C8_SHA,C9_SHA,HEAD_SHA,CHECKPOINT,
 train_specs,test_specs,validate_parent,source_hashes,inherited_check,sha,verify_open_zip,
 endpoint_gate,episode_gate,target_from_reads,counts,decision,snapshot)
from raw_writer import StateWriter,route_from_features,prepared_input,one_raw_proposal,strict_loss
from functional_writer import chart_spec,last_logits
ROOT=Path(__file__).resolve().parent
pack=snapshot
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)

def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);os.replace(temp,path)

def load_obj(z,name):return torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)

def vector(value):
    if isinstance(value,dict):value=value['theta']
    if not isinstance(value,torch.Tensor) or value.shape!=(36864,) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Finite original float32 chart tensor required')
    return value.detach().cpu().clone()

def head_hash(head):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()


def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inherited_check(ROOT);validate_parent(out/'parent_c9.zip');device,gpu=a100_device(torch)
    ident={'protocol':PROTOCOL,'parent_c9_sha256':C9_SHA,'settings':SETTINGS,'source_hashes':source_hashes(ROOT),
     'gpu':gpu,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
     'numpy':np.__version__,'model_revision':REVISION,'initial_c9_head_sha256':HEAD_SHA}
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('Runtime/source changed; refuse incompatible resume')
    dump(out/'identity.json',ident)
    if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_worker.py','build_worker_c15.py','build_package.py'))
    split={'training_specs':train_specs(),'original_training_ids':list(STATES),'test_specs':test_specs(),'settings':SETTINGS,'test_generation_after_final_freeze':True}
    if (out/'SPLIT.json').exists() and json.loads((out/'SPLIT.json').read_text())!=split:raise RuntimeError('Fixed prospective split changed')
    dump(out/'SPLIT.json',split)
    if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);pack(out,args.drive_dir);return
    counters=json.loads((out/'COUNTERS.json').read_text()) if (out/'COUNTERS.json').exists() else {'jacobian_builds':0,'response_autograd_calls':0,'new_head_training_steps':0}
    pack(out,args.drive_dir)
    cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
    w.set_seed(82931);print('C9C Loading pinned GPT-2 once; C9 initialization; offline expanded state-conditioned writer training.',flush=True)
    tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    frozen={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}
    def sync():torch.cuda.synchronize(device)
    def timed(fn,*a):
        sync();ts=time.perf_counter();v=fn(*a);sync();return v,time.perf_counter()-ts
    with ExitStack() as stack:
        c9=stack.enter_context(zipfile.ZipFile(out/'parent_c9.zip'))
        c8=stack.enter_context(zipfile.ZipFile(io.BytesIO(c9.read('parent_c8.zip'))))
        verify_open_zip(c8,'GPT2_C8_EXTRAPOLATION_DIAGNOSIS_A100_V1')
        c7=stack.enter_context(zipfile.ZipFile(io.BytesIO(c8.read('parent_c7.zip'))));verify_open_zip(c7,parent.PROTOCOL)
        b=c7.read('parent_c6.zip')
        if hashlib.sha256(b).hexdigest()!=C6_SHA:raise RuntimeError('Original C6 changed')
        c6=stack.enter_context(zipfile.ZipFile(io.BytesIO(b)));verify_open_zip(c6,parent.C6_PROTOCOL)
        cap=stack.enter_context(zipfile.ZipFile(io.BytesIO(c6.read('parent_c5_capsule.zip'))));parent.validate_capsule_open(cap)
        v5=stack.enter_context(zipfile.ZipFile(io.BytesIO(cap.read('parent_v5.zip'))))
        pi=json.loads(v5.read('parent/identity.json'));ah=hashlib.sha256()
        for n,p in model.named_parameters():
            if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
        if ah.hexdigest()!=pi['lora_A_sha256'] or getattr(model.config,'_commit_hash',None)!=REVISION:raise RuntimeError('Original model/LoRA-A identity mismatch')
        if [list(p.shape) for p in params]!=pi['chart_shapes']:raise RuntimeError('Original chart changed')
        if w.param_flat(params).abs().max()!=0:raise RuntimeError('Original zero-B reference required')
        symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
        if json.loads(json.dumps(symbols))!=pi['symbols']:raise RuntimeError('Read token identity changed')
        rng=random.Random(82931);anchors=rng.sample(w.ANCHORS,cfg['anchor_count'])
        response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
        with torch.no_grad():logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
        _,exact,_,dev=audit.prompts()
        exact_batch=tok(exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
        def frozen_check():
            if any(p._version!=frozen[n] for n,p in model.named_parameters() if n in frozen):
                raise RuntimeError('Frozen base/A changed')
            if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Unexpected trainable memory parameter')

        @torch.no_grad()
        def measured_response(theta):
            w.assign_flat(params,theta.to(device))
            response=w.response_vector(model,tok,anchors,response_ids,device)
            drift,kl=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
            return (response-response0).cpu().double().numpy(),float(drift),float(kl)

        @torch.no_grad()
        def evaluate(theta,target,start=None):
            if not torch.isfinite(theta).all():return {'pass':False,'reason':'NONFINITE_CANDIDATE'}
            sync();t0=time.perf_counter();w.assign_flat(params,theta.to(device));sync();commit_seconds=time.perf_counter()-t0;t0=time.perf_counter()
            bits=audit.content_bits(w,model,tok,exact,ids,device)
            ev=audit.evaluate(w,model,tok,exact,dev,[ids[b] for b in target],ids,device)
            drift,kl=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
            v={'bits':bits,'evaluation':ev,'drift':float(drift),'kl':float(kl)}
            v['pass']=endpoint_gate(v,target)
            v['gates']={'exact_read_margin':min(ev['exact_margins'])>=.05,
                        'development_read_margin':min(ev['heldout_margins'])>=.05,
                        'bits_match':bits==target,'response':drift<=.002,'KL':kl<=.005}
            if start is not None:
                v['gates'].update(output_flip=bits[3]!=start[3],unselected_preserved=bits[:3]==start[:3])
                v['episode_pass']=episode_gate(start,v)
            per_logits=w.next_logits(model,tok,anchors,device)
            per_R=(per_logits[:,response_ids]-logits0[:,response_ids]).abs().max(1).values
            per_KL=(logits0.softmax(-1)*(logits0.log_softmax(-1)-per_logits.log_softmax(-1))).sum(-1)
            v['per_anchor_diagnostics']={'selected_logit_max_abs_changes':per_R.cpu().tolist(),'KL_values':per_KL.cpu().tolist(),
                'max_selected_logit_change':float(per_R.max()),'max_KL':float(per_KL.max()),
                'acceptance_uses_average_response_and_average_KL':True,'not_additional_gates':True}
            sync();v['timing']={'commit_seconds':commit_seconds,'verification_and_diagnostics_seconds':time.perf_counter()-t0}
            frozen_check();return v

        def jacobian(theta):
            w.assign_flat(params,theta.to(device))
            for p in params:p.requires_grad_(True)
            try:
                with torch.enable_grad():
                    response=w.response_vector(model,tok,anchors,response_ids,device)
                    rows=[]
                    for i in range(response.numel()):
                        grads=torch.autograd.grad(response[i],params,retain_graph=i+1<response.numel())
                        rows.append(torch.cat([g.reshape(-1) for g in grads]).detach().cpu().double().numpy())
                    J=np.stack(rows)
                counters['jacobian_builds']+=1;counters['response_autograd_calls']+=len(rows)
            finally:
                for p in params:p.requires_grad_(False);p.grad=None
            frozen_check();dump(out/'COUNTERS.json',counters);return J

        def correct(path,initial):
            path.mkdir(parents=True,exist_ok=True);cpfile=path/'CORRECTION_STATE.pt'
            resume=torch.load(cpfile,map_location='cpu',weights_only=True) if cpfile.exists() else None
            if resume is not None and resume.get('initial_sha256')!=tensor_hash(initial):raise RuntimeError('Correction initial state changed')
            def persist(cp):
                cp['initial_sha256']=tensor_hash(initial);save_tensor(cpfile,cp);dump(path/'CORRECTION_HISTORY.json',cp['history'])
            current,h,seconds=run_correction(initial,measured_response,jacobian,
                lambda t,d:t+torch.from_numpy(d).float(),lambda t:vector(t),persist,SETTINGS,resume,sync)
            return current,h,seconds

        @torch.no_grad()
        def features(theta,desired):
            w.assign_flat(params,theta.to(device))
            result=model(**exact_batch,use_cache=False,output_hidden_states=True)
            last=exact_batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=device)
            hidden=result.hidden_states[-1][rows,last].reshape(-1)
            logits=result.logits[rows,last][:,ids]
            margins=logits[:,1]-logits[:,0]
            return torch.cat([hidden,margins,torch.tensor([desired],device=device,dtype=hidden.dtype)]).detach().cpu()

        starts={case:json.loads(v5.read(case+'/PREPARATION.json'))['target_bits'] for case in CASES}
        targets={case:target_from_reads(starts[case]) for case in CASES}
        cp=load_obj(c9,CHECKPOINT)
        if cp['steps']!=240 or cp['stage']!='functional' or cp['train_ids']!=list(STATES):raise RuntimeError('Original C9 head changed')
        c9norm={k:v.to(device) for k,v in cp['normalisation'].items()}
        baseline=WriterHead(3077,36864,32).to(device);baseline.load_state_dict(cp['head']);baseline.eval()
        for p in baseline.parameters():p.requires_grad_(False)
        if head_hash(baseline)!=HEAD_SHA:raise RuntimeError('C9 baseline identity changed')
        rng_sketch=np.random.default_rng(SETTINGS['sketch_seed'])
        sketch=torch.from_numpy((rng_sketch.normal(size=(36864,64))/np.sqrt(36864)).astype(np.float32)).to(device)
        save_tensor(out/'STATE_SKETCH.pt',{'matrix':sketch.cpu(),'seed':SETTINGS['sketch_seed'],'not_a_Jacobian':True})
        base_data={}
        for sid in STATES:
            prep=vector(load_obj(cap,'pool/'+sid+'/prepared.pt'))
            teacher=vector(load_obj(c9,'evaluation/training/'+sid+'/endpoint.pt'))
            base_data[sid]=(prep,teacher)

        def extra_features(theta):
            residual,_,_=measured_response(theta)
            return torch.cat([torch.from_numpy(residual).float().to(device)/SETTINGS['response_budget'],theta.to(device)@sketch]).cpu()

        def prepare_state(folder,t):
            folder.mkdir(parents=True,exist_ok=True);ref_file=folder/'REFERENCE.json'
            if ref_file.exists():return json.loads(ref_file.read_text())
            prep0,teacher0=base_data[t['base_id']];qfile=folder/'perturbation.pt'
            if qfile.exists():
                savedq=torch.load(qfile,map_location='cpu',weights_only=True);q=vector(savedq);info=savedq['geometry']
                if savedq['spec']!=t:raise RuntimeError('Cached perturbation specification changed')
            else:
                J=load_obj(c8,'states/'+t['base_id']+'/jacobian.pt')['J'].double().numpy()
                projected,info=null_project(J,np.random.default_rng(t['seed']).normal(size=36864));qnorm=float(np.linalg.norm(projected))
                if not qnorm>0:raise RuntimeError('Zero fixed projected perturbation; no replacement seed allowed')
                q=torch.from_numpy(projected*(t['norm']/qnorm)).float();save_tensor(qfile,{'theta':q,'geometry':info,'spec':t})
            prep,ph,ps=correct(folder/'offline_preparation',prep0+q)
            teacher,th,ts=correct(folder/'offline_teacher',teacher0+q)
            save_tensor(folder/'prepared.pt',{'theta':prep});save_tensor(folder/'teacher.pt',{'theta':teacher})
            pm=evaluate(prep,starts[t['case']]);tm=evaluate(teacher,targets[t['case']],starts[t['case']])
            feat=features(prep,targets[t['case']][3]);extra=extra_features(prep);save_tensor(folder/'FEATURES.pt',{'features':feat,'extras':extra})
            r={'spec':t,'preparation':pm,'teacher':tm,'valid':bool(pm['pass'] and tm['episode_pass']),
             'generation_retries':0,'replacement_used':False,'offline_geometry':True,'preparation_correction_seconds':ps,'teacher_correction_seconds':ts,
             'teacher_is_offline_positive_reference_not_runtime_fallback':True,'projection':info}
            dump(ref_file,r);print('C15 OFFLINE STATE',t['id'],'valid',r['valid'],flush=True);return r

        train=[]
        for sid in STATES:
            case=sid.split('__')[0];folder=out/'states'/'training'/sid;folder.mkdir(parents=True,exist_ok=True)
            ref_file=folder/'REFERENCE.json'
            if not ref_file.exists():
                prep,teacher=base_data[sid];save_tensor(folder/'prepared.pt',{'theta':prep});save_tensor(folder/'teacher.pt',{'theta':teacher})
                pm=evaluate(prep,starts[case]);tm=evaluate(teacher,targets[case],starts[case]);feat=features(prep,targets[case][3]);extra=extra_features(prep)
                save_tensor(folder/'FEATURES.pt',{'features':feat,'extras':extra})
                dump(ref_file,{'spec':{'id':sid,'case':case},'valid':bool(pm['pass'] and tm['episode_pass']),'preparation':pm,'teacher':tm,'original_C9_training_reference':True})
            ref=json.loads(ref_file.read_text());train.append({'id':sid,'case':case,'folder':folder,'reference':ref})
        for i,t in enumerate(train_specs()):
            folder=out/'states'/'training'/t['id'];ref=prepare_state(folder,t);train.append({'id':t['id'],'case':t['case'],'folder':folder,'reference':ref})
            if (i+1)%4==0:pack(out,args.drive_dir)
        usable=[r for r in train if r['reference']['valid']]
        if any(not r['reference']['valid'] for r in train[:12]):raise RuntimeError('Original C9 training replay failed; no guessed continuation')
        data=[]
        for r in usable:
            f=torch.load(r['folder']/'FEATURES.pt',map_location='cpu',weights_only=True)
            prep=vector(torch.load(r['folder']/'prepared.pt',map_location='cpu',weights_only=True));teacher=vector(torch.load(r['folder']/'teacher.pt',map_location='cpu',weights_only=True))
            data.append((prep,teacher,f['features'],f['extras']))
        extra_tensor=torch.stack([r[3] for r in data]).to(device)
        extra_norm={'mean':extra_tensor.mean(0),'scale':extra_tensor.std(0,unbiased=False).clamp_min(1e-4)}
        X=prepared_input(torch.stack([r[2] for r in data]).to(device),extra_tensor,c9norm,extra_norm)
        routes=route_from_features(torch.stack([r[2] for r in data]).to(device))
        train_ids=[r['id'] for r in usable]
        Y=torch.stack([r[1]-r[0] for r in data]).to(device)/c9norm['output_scale']
        dump(out/'TRAINING_SET.json',{'requested':36,'valid':len(usable),'training_ids':train_ids,'invalid_ids':[r['id'] for r in train if not r['reference']['valid']],
          'eligibility_rule':'fixed preparation and positive-reference gates only; no replacement states or test data'})
        w.set_seed(SETTINGS['head_seed']);head=StateWriter(3147,36864,64).to(device);head.initialise_c9(cp['head'])
        batch=tok(anchors+exact+dev,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device);spec=chart_spec(model,params)

        def fit_stage(stage,total,lr):
            folder=out/'heads'/stage;folder.mkdir(parents=True,exist_ok=True);start_hash=head_hash(head);history=[];begin=0
            optimiser=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.)
            def checkpoint(step):
                f=folder/('checkpoint_%06d.pt'%step)
                save_tensor(f,{'head':head.state_dict(),'optimiser':optimiser.state_dict() if step<total else None,'steps':step,'stage':stage,
                 'train_ids':train_ids,'start_head_sha256':start_hash,'extra_norm':{k:v.cpu() for k,v in extra_norm.items()},'history':history})
                dump(folder/'PROGRESS.json',{'checkpoint':f.name,'checkpoint_sha256':sha(f),'steps':step})
                for old in folder.glob('checkpoint_*.pt'):
                    if old!=f:old.unlink()
            progress=folder/'PROGRESS.json'
            if progress.exists():
                pr=json.loads(progress.read_text());f=folder/pr['checkpoint']
                if sha(f)!=pr['checkpoint_sha256']:raise RuntimeError('Actual C15 checkpoint digest mismatch')
                saved=torch.load(f,map_location='cpu',weights_only=True)
                if saved['stage']!=stage or saved['train_ids']!=train_ids or saved['start_head_sha256']!=start_hash:raise RuntimeError('Stage/training identity mismatch')
                begin=saved['steps'];history=saved['history']
                if begin!=pr['steps'] or not 0<=begin<=total or any(not torch.equal(saved['extra_norm'][k].to(device),v) for k,v in extra_norm.items()):raise RuntimeError('Actual progress/normalisation changed')
                head.load_state_dict(saved['head'])
                if begin<total:
                    if saved['optimiser'] is None:raise RuntimeError('Missing real optimizer for incomplete training')
                    optimiser.load_state_dict(saved['optimiser'])
                print('C15 RESTORE',stage,begin,flush=True)
            else:checkpoint(0)
            head.train()
            for step in range(begin,total):
                optimiser.zero_grad(set_to_none=True);terms=None
                if stage=='warm':pred=head(X,routes);loss=(pred-Y).square().mean()
                else:
                    j=step%len(data);pred=head(X[j:j+1],routes[j:j+1])[0]
                    candidate=data[j][0].to(device)+pred*c9norm['output_scale']
                    logits=last_logits(model,batch,candidate,spec);mse=(pred-Y[j]).square().mean()
                    loss,terms=strict_loss(logits,len(anchors),response_ids,logits0,ids,torch.tensor(targets[usable[j]['case']],device=device),mse,SETTINGS)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite offline head loss')
                loss.backward();grads=[p.grad for p in head.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid head gradients')
                gn=torch.nn.utils.clip_grad_norm_(head.parameters(),SETTINGS['gradient_clip']);optimiser.step();frozen_check()
                if (step+1)%20==0:
                    r={'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(gn)}
                    if terms:r.update({k:float(v.detach()) for k,v in terms.items()})
                    history.append(r);print('C15 FIT',stage,json.dumps(r),flush=True)
                if (step+1)%SETTINGS['checkpoint_every']==0 or step+1==total:checkpoint(step+1)
                counters['new_head_training_steps']=(1000 if stage=='functional' else 0)+step+1;dump(out/'COUNTERS.json',counters)
                interval=SETTINGS['warm_archive_every'] if stage=='warm' else SETTINGS['functional_archive_every']
                if (step+1)%interval==0:pack(out,args.drive_dir)
            return head_hash(head)
        fit_stage('warm',1000,SETTINGS['warm_lr']);fit_stage('functional',720,SETTINGS['functional_lr'])
        head.eval()
        for p in head.parameters():p.requires_grad_(False);p.grad=None
        frozen_head=head_hash(head)
        frozen_record={'head_sha256':frozen_head,'training_ids':train_ids,'selected_checkpoint':'heads/functional/checkpoint_000720.pt',
          'test_based_selection':False,'warm_steps':1000,'functional_steps':720,'new_state_generation_begins_after_freeze':True}
        if (out/'WRITER_FROZEN.json').exists() and json.loads((out/'WRITER_FROZEN.json').read_text())!=frozen_record:raise RuntimeError('Final frozen writer changed')
        dump(out/'WRITER_FROZEN.json',frozen_record);pack(out,args.drive_dir)
        print('C15 WRITER FROZEN',frozen_head,'new evaluation state generation begins now',flush=True)

        def frozen_writer_check():
            if head_hash(head)!=frozen_head or head_hash(baseline)!=HEAD_SHA or any(p.requires_grad for p in head.parameters()):raise RuntimeError('Frozen writer changed during evaluation')
            frozen_check()

        def evaluate_raw(row,cohort):
            sid=row['id'];case=row['case'];folder=out/'evaluation'/cohort/sid;folder.mkdir(parents=True,exist_ok=True);complete=folder/'COMPLETE.json'
            if complete.exists():return json.loads(complete.read_text())
            prep=vector(torch.load(row['folder']/'prepared.pt',map_location='cpu',weights_only=True))
            input_file=folder/'ONLINE_INPUT.pt'
            if input_file.exists():online=torch.load(input_file,map_location='cpu',weights_only=True)
            else:
                sync();ts=time.perf_counter();feat=features(prep,targets[case][3]);extra=extra_features(prep)
                x=prepared_input(feat.to(device),extra.to(device),c9norm,extra_norm);route=route_from_features(feat.to(device));sync()
                online={'features':feat,'input':x.cpu(),'route':route.cpu(),'input_seconds':time.perf_counter()-ts};save_tensor(input_file,online)
            metrics={};timings={}
            for arm in ('c15_raw','frozen_c9_raw'):
                f=folder/(arm+'.pt');meta_file=folder/(arm+'_PROPOSAL.json')
                if f.exists():
                    saved=torch.load(f,map_location='cpu',weights_only=True);raw=vector(saved);timing=saved['proposal_meta']
                else:
                    sync();ts=time.perf_counter()
                    if arm=='c15_raw':delta=one_raw_proposal(head,online['input'].to(device),online['route'].to(device),c9norm['output_scale']).cpu()
                    else:delta=one_forward_proposal(baseline,online['features'].to(device),c9norm).cpu()
                    raw=prep+delta;sync();timing={'proposal_seconds':time.perf_counter()-ts,'head_forward_calls':1,'writer_head_sha256':frozen_head if arm=='c15_raw' else HEAD_SHA}
                    save_tensor(f,{'theta':raw,'proposal_meta':timing});dump(meta_file,timing)
                # No Jacobian, SVD, autograd or correction call is permitted in this online path.
                before=dict(counters);v=evaluate(raw,targets[case],starts[case]);frozen_writer_check()
                if counters!=before:raise RuntimeError('Unexpected online geometry operation')
                dump(folder/(arm+'.json'),v);metrics[arm]=v;timings[arm]=timing
                print('C15 RAW',cohort,sid,arm,'pass',v.get('episode_pass',False),'drift',v.get('drift'),'KL',v.get('kl'),flush=True)
            r={'id':sid,'case':case,'reference_valid':row['reference']['valid'],'raw':metrics['c15_raw'],'baseline':metrics['frozen_c9_raw'],
               'input_seconds':online['input_seconds'],'proposal_timings':timings,'online_Jacobian_builds':0,'online_SVD_calls':0,
               'online_correction_steps':0,'head_proposals_per_arm':1,'online_commits_per_arm':1,'teacher_fallback_used':False,
               'writer_head_sha256':frozen_head,'raw_acceptance_only':True}
            dump(complete,r);return r
        training_rows=[evaluate_raw(r,'training') for r in train];pack(out,args.drive_dir)
        new_rows=[]
        for i,t in enumerate(test_specs()):
            if not (out/'WRITER_FROZEN.json').exists():raise RuntimeError('Test generation before writer freeze forbidden')
            folder=out/'states'/'new'/t['id'];ref=prepare_state(folder,t)
            r=evaluate_raw({'id':t['id'],'case':t['case'],'folder':folder,'reference':ref},'new');new_rows.append(r)
            if (i+1)%4==0:pack(out,args.drive_dir)
        summary={'protocol':PROTOCOL,'complete':True,'decision':decision(training_rows,new_rows),
         'training_raw':counts(training_rows),'new_state_raw':counts(new_rows),
         'new_states_by_case':{case:counts([r for r in new_rows if r['case']==case]) for case in CASES},
         'writer_head_sha256':frozen_head,'baseline_c9_head_sha256':HEAD_SHA,'new_head_training_steps':1720,
         'online_write_Jacobian_builds':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,
         'head_proposals_per_arm_per_state':1,'online_write_commits_per_arm_per_state':1,
         'teacher_fallback_used':False,'offline_geometry_and_training_used':True,'fresh_state_seeds':True,'fresh_evaluation':False,
         'independent_model_seed':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
         'requested_output_bit_supplied':True,'scope':SETTINGS['scope'],'parent_C9_decision_unchanged':'C9_LOCAL_STATE_GENERALISATION_PASS'}
        frozen_writer_check();w.assign_flat(params,torch.zeros(36864,device=device));frozen_check()
        dump(out/'SUMMARY.json',summary);pack(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
