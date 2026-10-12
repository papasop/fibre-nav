"""Train on four observed input classes; freeze; evaluate fixed new local states."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,zipfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
import c7_protocol as parent
from runtime_a100 import a100_device
from geometry import null_project,ridge_correction
from proposer import WriterHead,fit_normalisation,normalise,one_forward_proposal
from functional_writer import chart_spec,last_logits,objective_terms,combined_loss
from c9_protocol import *
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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    validate_c8(out/'parent_c8.zip');device,gpu=a100_device(torch)
    ident={'protocol':PROTOCOL,'parent_c8_sha256':C8_SHA,'settings':SETTINGS,'source_hashes':source_hashes(ROOT),
      'gpu':gpu,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
      'numpy':np.__version__,'model_revision':REVISION,'development_only':True,'internal_parameter_update':False,
      'controller_dispatch_tested':False,'external_jacobian_and_correction':True}
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('C9 runtime/source changed; refusing guessed resume')
    dump(out/'identity.json',ident)
    if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_package.py'))
    split={'training_ids':list(STATES),'new_state_specs':test_specs(),'settings':SETTINGS,
           'old_excluded_inputs_promoted_to_training':True,'new_states_not_generated_until_writer_frozen':True}
    if (out/'SPLIT.json').exists() and json.loads((out/'SPLIT.json').read_text())!=split:raise RuntimeError('Predeclared split changed')
    dump(out/'SPLIT.json',split)
    if (out/'SUMMARY.json').exists():
        print('C9 already complete; no repeated training/evaluation.',flush=True);print((out/'SUMMARY.json').read_text(),flush=True);pack(out,args.drive_dir);return
    pack(out,args.drive_dir)
    counters=json.loads((out/'COUNTERS.json').read_text()) if (out/'COUNTERS.json').exists() else {'jacobian_builds':0,'response_autograd_calls':0,'head_training_steps':0}
    cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
    w.set_seed(82931);print('C9 Loading pinned GPT-2 once; original memory chart, new conditional writer head.',flush=True)
    tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    frozen={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}
    with ExitStack() as stack:
        c8=stack.enter_context(zipfile.ZipFile(out/'parent_c8.zip'))
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
        spec=chart_spec(model,params)
        all_batch=tok(anchors+exact+dev,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
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
            w.assign_flat(params,theta.to(device))
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

        def branch(path,name,theta,target,start):
            v=evaluate(theta,target,start);save_tensor(path/(name+'.pt'),{'theta':theta.cpu()})
            dump(path/(name+'.json'),v)
            print('C9 BRANCH',path.name,name,json.dumps({'pass':v.get('episode_pass',False),
                  'bits':v.get('bits'),'drift':v.get('drift'),'kl':v.get('kl'),'gates':v.get('gates')}),flush=True)
            return v

        def correct(path,initial,target,start):
            state_file=path/'CORRECTION_STATE.pt';history_file=path/'CORRECTION_HISTORY.json'
            if state_file.exists():
                cp=torch.load(state_file,map_location='cpu',weights_only=True)
                current=vector(cp);begin=int(cp['iteration']);history=cp['history']
                if not 0<=begin<=SETTINGS['correction_max_steps']:raise RuntimeError('Invalid correction progress')
                print('C9 CORRECTION RESTORE',path.name,'actual_step',begin,flush=True)
                if history and history[-1].get('accepted') is False:return current
            else:current=initial.cpu().clone();begin=0;history=[]
            for step in range(begin,SETTINGS['correction_max_steps']):
                e,drift,kl=measured_response(current)
                if drift<=SETTINGS['response_budget']:
                    history.append({'iteration':step,'stop':'RESPONSE_WITHIN_BUDGET','drift':drift,'kl':kl});break
                J=jacobian(current);delta,info=ridge_correction(J,e,SETTINGS['correction_ridge'])
                accepted=False;trials=[]
                for bt in range(SETTINGS['correction_backtracking']):
                    alpha=2.**(-bt)
                    trial=current+torch.from_numpy(alpha*delta).float()
                    trial_name=f'correction_trial_{step:02d}_{bt:02d}.pt'
                    save_tensor(path/trial_name,{'theta':trial})
                    _,new_drift,new_kl=measured_response(trial)
                    finite=bool(torch.isfinite(trial).all() and np.isfinite([new_drift,new_kl]).all())
                    ok=finite and new_drift<drift*(1-1e-4*alpha)
                    trials.append({'alpha':alpha,'drift':new_drift,'kl':new_kl,'accepted_for_response_only':bool(ok),
                                   'candidate_file':trial_name})
                    if ok:current=trial;accepted=True;break
                history.append({'iteration':step+1,'measured_pre_step_residual':e.tolist(),
                                'pre_step_drift':drift,'correction':info,'trials':trials,'accepted':accepted})
                save_tensor(state_file,{'theta':current,'iteration':step+1,'history':history});dump(history_file,history)
                dump(path/'CORRECTION_PROGRESS.json',{'iteration':step+1,'stopped':not accepted})
                print('C9 CORRECTION',path.name,'step',step+1,'accepted',accepted,
                      'drift',trials[-1]['drift'],'KL',trials[-1]['kl'],flush=True)
                if (step+1)%SETTINGS['correction_checkpoint_every']==0:pack(out,args.drive_dir)
                if not accepted:break
            dump(history_file,history)
            return current

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
        data=[];refs=[]
        for sid in STATES:
            case=sid.split('__')[0];prep=vector(load_obj(cap,'pool/'+sid+'/prepared.pt'));teacher=vector(load_obj(cap,'pool/'+sid+'/teacher.pt'))
            feat=load_obj(cap,'pool/'+sid+'/features.pt')['features'].cpu()
            pm=evaluate(prep,starts[case]);tm=evaluate(teacher,targets[case],starts[case]);err=float((features(prep,targets[case][3])-feat).abs().max())
            if err>SETTINGS['feature_identity_tolerance'] or not pm['pass'] or not tm['episode_pass']:raise RuntimeError('Original training reference failed '+sid)
            r={'state':sid,'preparation':pm,'teacher':tm,'feature_max_abs_difference':err,'usable':True};refs.append(r)
            dump(out/'training_references'/(sid+'.json'),r);data.append((prep,teacher,feat))
        X=torch.stack([r[2] for r in data]).to(device);Y=torch.stack([r[1]-r[0] for r in data]).to(device)
        norm=fit_normalisation(X,Y);Xn=normalise(X,norm);Yn=Y/norm['output_scale']
        dump(out/'NORMALISATION_SPEC.json',{'training_ids':list(STATES),'test_features_used':False,'test_teachers_used':False})
        w.set_seed(SETTINGS['head_seed']);head=WriterHead(X.shape[1],Y.shape[1],SETTINGS['head_width']).to(device)
        probe=data[0][0].to(device).clone().requires_grad_(True)
        output=last_logits(model,all_batch,probe,spec)
        grad=torch.autograd.grad((output[len(anchors):,ids[1]]-output[len(anchors):,ids[0]]).sum(),probe)[0]
        w.assign_flat(params,probe.detach())
        with torch.no_grad():committed=w.next_logits(model,tok,anchors+exact+dev,device)
        mismatch=float((output.detach()-committed).abs().max())
        if mismatch>1e-4 or not torch.isfinite(grad).all() or not grad.norm()>0:raise RuntimeError('Functional chart gradient/commit audit failed')
        dump(out/'ACTUAL_FUNCTIONAL_AUDIT.json',{'gradient_norm':float(grad.norm()),'maximum_logit_mismatch':mismatch,'tolerance':1e-4})
        del probe,output,grad,committed;frozen_check()

        def fit_stage(stage,n_steps,lr):
            folder=out/'heads'/stage;folder.mkdir(parents=True,exist_ok=True);start_hash=head_hash(head)
            optimiser=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=0.);begin=0;history=[]
            def checkpoint(step):
                f=folder/('checkpoint_%06d.pt'%step)
                save_tensor(f,{'head':head.state_dict(),'optimiser':optimiser.state_dict(),'steps':step,'stage':stage,
                    'train_ids':list(STATES),'normalisation':{n:v.cpu() for n,v in norm.items()},'history':history,'start_head_sha256':start_hash})
                dump(folder/'PROGRESS.json',{'steps':step,'checkpoint':f.name,'checkpoint_sha256':sha(f)})
                for old in folder.glob('checkpoint_*.pt'):
                    if old!=f:old.unlink()
            if (folder/'PROGRESS.json').exists():
                p=json.loads((folder/'PROGRESS.json').read_text());f=folder/p['checkpoint']
                if sha(f)!=p['checkpoint_sha256']:raise RuntimeError('Actual head checkpoint digest mismatch')
                cp=torch.load(f,map_location='cpu',weights_only=True)
                if cp['stage']!=stage or cp['train_ids']!=list(STATES) or cp['start_head_sha256']!=start_hash:raise RuntimeError('Training stage/split identity changed')
                if any(not torch.equal(cp['normalisation'][n].to(device),norm[n]) for n in norm):raise RuntimeError('Training-only normalisation changed')
                begin=cp['steps'];history=cp['history'];head.load_state_dict(cp['head']);optimiser.load_state_dict(cp['optimiser'])
                if begin!=p['steps'] or not 0<=begin<=n_steps:raise RuntimeError('Invalid head progress')
                print('C9 HEAD_RESTORE',stage,begin,flush=True)
            else:checkpoint(0)
            head.train()
            for step in range(begin,n_steps):
                optimiser.zero_grad(set_to_none=True);pred=head(Xn);mse=(pred-Yn).square().mean();loss=mse;terms=None
                if stage=='functional':
                    j=step%len(STATES);case=STATES[j].split('__')[0]
                    candidate=data[j][0].to(device)+pred[j]*norm['output_scale']
                    logits=last_logits(model,all_batch,candidate,spec)
                    terms=objective_terms(logits,len(anchors),response_ids,logits0,ids,torch.tensor(targets[case],device=device))
                    loss=combined_loss(mse,terms,SETTINGS['loss_weights'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite writer loss')
                loss.backward();grads=[p.grad for p in head.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid writer gradients')
                gn=torch.nn.utils.clip_grad_norm_(head.parameters(),SETTINGS['gradient_clip']);optimiser.step();frozen_check()
                if (step+1)%25==0:
                    r={'step':step+1,'full_batch_mse':float(mse.detach()),'loss':float(loss.detach()),'gradient_norm':float(gn)}
                    if terms:r.update({n:float(v.detach()) for n,v in terms.items()})
                    history.append(r);print('C9 FIT',stage,json.dumps(r),flush=True)
                if (step+1)%SETTINGS['checkpoint_every']==0 or step+1==n_steps:checkpoint(step+1)
                interval=SETTINGS['warm_archive_every'] if stage=='warm' else SETTINGS['functional_archive_every']
                if (step+1)%interval==0:pack(out,args.drive_dir)
            return head_hash(head)

        fit_stage('warm',SETTINGS['warm_steps'],SETTINGS['warm_lr'])
        fit_stage('functional',SETTINGS['functional_steps'],SETTINGS['functional_lr'])
        head.eval()
        for p in head.parameters():p.requires_grad_(False);p.grad=None
        frozen_head=head_hash(head)
        freeze_record={'head_sha256':frozen_head,'selected_checkpoint':'heads/functional/checkpoint_000240.pt',
            'training_ids':list(STATES),'warm_steps':2000,'functional_steps':240,'test_based_selection':False}
        if (out/'WRITER_FROZEN.json').exists() and json.loads((out/'WRITER_FROZEN.json').read_text())!=freeze_record:raise RuntimeError('Frozen head changed')
        dump(out/'WRITER_FROZEN.json',freeze_record);pack(out,args.drive_dir)
        print('C9 WRITER_FROZEN',frozen_head,'new-state generation begins only now',flush=True)

        def frozen_writer_check():
            if head_hash(head)!=frozen_head or any(p.requires_grad for p in head.parameters()):raise RuntimeError('Writer changed during evaluation')
            frozen_check()

        def evaluate_writer(folder,prep,feat,case):
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE.json').exists():return json.loads((folder/'COMPLETE.json').read_text())
            if (folder/'raw.pt').exists():raw=vector(torch.load(folder/'raw.pt',map_location='cpu',weights_only=True))
            else:
                raw=prep+one_forward_proposal(head,feat.to(device),norm).cpu()
                save_tensor(folder/'raw.pt',{'theta':raw})
            raw_metrics=branch(folder,'raw',raw,targets[case],starts[case])
            theta=correct(folder,raw,targets[case],starts[case]);corrected=branch(folder,'endpoint',theta,targets[case],starts[case])
            frozen_writer_check()
            row={'case':case,'raw':raw_metrics,'corrected':corrected,'state_usable':True,'writer_head_sha256':frozen_head,
                 'raw_is_one_head_forward':True,'external_response_correction':True,'teacher_fallback_used':False}
            dump(folder/'COMPLETE.json',row);return row

        training=[]
        for sid,d in zip(STATES,data):
            row=evaluate_writer(out/'evaluation'/'training'/sid,d[0],d[2],sid.split('__')[0]);training.append(row)
        pack(out,args.drive_dir)
        new_rows=[]
        for si,t in enumerate(test_specs()):
            sid=t['id'];case=t['case'];folder=out/'new_states'/sid;folder.mkdir(parents=True,exist_ok=True)
            reference_file=folder/'REFERENCE.json';evalfolder=out/'evaluation'/'new'/sid;evalfolder.mkdir(parents=True,exist_ok=True)
            if reference_file.exists():reference=json.loads(reference_file.read_text())
            else:
                if not (out/'WRITER_FROZEN.json').exists():raise RuntimeError('Test generation attempted before writer freeze')
                prep0=vector(load_obj(cap,'pool/'+case+'__v0/prepared.pt'));teacher0=vector(load_obj(cap,'pool/'+case+'__v0/teacher.pt'))
                J=load_obj(c8,'states/'+case+'__v0/jacobian.pt')['J'].double().numpy()
                random_direction=np.random.default_rng(t['seed']).normal(size=prep0.numel())
                projected,geometry=null_project(J,random_direction)
                qnorm=np.linalg.norm(projected)
                if not qnorm>0:raise RuntimeError('Projected fixed perturbation is zero; no replacement seed allowed')
                q=torch.from_numpy(projected*(t['norm']/qnorm)).float()
                save_tensor(folder/'perturbation.pt',{'theta':q})
                pp=folder/'preparation_correction';pp.mkdir(exist_ok=True)
                prep=correct(pp,prep0+q,starts[case],None);save_tensor(folder/'prepared.pt',{'theta':prep})
                tp=folder/'teacher_reference_correction';tp.mkdir(exist_ok=True)
                teacher=correct(tp,teacher0+q,targets[case],starts[case]);save_tensor(folder/'teacher.pt',{'theta':teacher})
                pm=evaluate(prep,starts[case]);tm=evaluate(teacher,targets[case],starts[case]);feat=features(prep,targets[case][3])
                save_tensor(folder/'features.pt',{'features':feat})
                feature_distance=torch.linalg.vector_norm(normalise(X,norm)-normalise(feat.to(device),norm),dim=1)
                reference={'spec':t,'preparation':pm,'teacher':tm,'usable':bool(pm['pass'] and tm['episode_pass']),
                 'actual_preparation_delta_norm':float((prep-prep0).norm()),'projected_perturbation':geometry,
                 'nearest_training_feature_distance':float(feature_distance.min()),
                 'nearest_training_state':STATES[int(feature_distance.argmin())],
                 'teacher_constructed_by_transport_not_independent_optimisation':True,
                 'generated_after_writer_freeze':True,'generation_retries':0,'state_replacement_used':False}
                dump(reference_file,reference)
            if not reference['usable']:
                row={'case':case,'state_usable':False,'reason':'PREDECLARED_STATE_OR_TEACHER_REFERENCE_FAILED',
                     'raw':{'episode_pass':False},'corrected':{'episode_pass':False},'writer_evaluation_skipped':True}
                dump(evalfolder/'COMPLETE.json',row)
            else:
                prep=vector(torch.load(folder/'prepared.pt',map_location='cpu',weights_only=True))
                feat=torch.load(folder/'features.pt',map_location='cpu',weights_only=True)['features']
                row=evaluate_writer(evalfolder,prep,feat,case)
            new_rows.append(row);print('C9 NEW_STATE',sid,'reference',reference['usable'],'raw/corrected',row['raw']['episode_pass'],row['corrected']['episode_pass'],flush=True)
            if (si+1)%2==0:pack(out,args.drive_dir)
        def counts(rows):
            return {'n_requested':len(rows),'valid_state_references':sum(r['state_usable'] for r in rows),
                    'raw_passed':sum(r['raw']['episode_pass'] for r in rows),'corrected_passed':sum(r['corrected']['episode_pass'] for r in rows)}
        summary={'protocol':PROTOCOL,'complete':True,'decision':decision(training,new_rows),'training':counts(training),
          'new_state_development':counts(new_rows),'new_states_by_case':{c:counts([r for r in new_rows if r['case']==c]) for c in CASES},
          'writer_head_sha256':frozen_head,'persisted_committed_head_steps':2240,'new_model_memory_loss_optimiser_steps':0,
          'new_state_preparation_and_teacher_response_correction':True,'teacher_fallback_used':False,
          'fresh_state_seeds':True,'fresh_evaluation':False,'same_four_XOR_input_classes_seen_in_training':True,
          'same_access_prompts_and_model_seed':True,'new_state_generation_retries':0,
          'internal_parameter_update':False,'controller_dispatch_tested':False,
          'scope':'Development-only robustness on eight predeclared local perturbations around four known preparations; original inspected inputs promoted to training. Requested output bit supplied. External autograd/SVD/correction, no native update or unseen-XOR-combination claim.',
          'C8_parent_decision_unchanged':'C8_DEVELOPMENT_EXTRAPOLATION_DIAGNOSIS_COMPLETE'}
        frozen_writer_check();w.assign_flat(params,torch.zeros_like(data[0][0]).to(device));frozen_check()
        dump(out/'SUMMARY.json',summary);pack(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
