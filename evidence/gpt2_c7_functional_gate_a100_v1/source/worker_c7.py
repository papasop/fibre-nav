"""C7: matched parameter-regression and functional writer heads; development only."""
import argparse,fcntl,gc,hashlib,io,json,os,random,shutil,time,zipfile
from pathlib import Path
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
from runtime_a100 import a100_device
from geometry import null_project,ridge_correction
from c7_protocol import *
from proposer import WriterHead,fit_normalisation,normalise,one_forward_proposal
from functional_writer import chart_spec,last_logits,objective_terms,combined_loss

ROOT=Path(__file__).resolve().parent
PACK_SECONDS=0.

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2))
    os.replace(temp,path)

def save_tensor(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)

def pack(out,drive):
    global PACK_SECONDS
    pack_started=time.perf_counter()
    m={str(p.relative_to(out)):sha(p) for p in out.rglob('*')
       if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
    dump(out/'MANIFEST.json',m)
    dest=out.parent/'gpt2_c7_functional_gate_results.zip';temp=dest.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
    verified_zip(temp,PROTOCOL);os.replace(temp,dest)
    if drive:
        d=Path(drive);d.mkdir(parents=True,exist_ok=True)
        q=d/f'gpt2_c7_functional_gate_results_{time.time_ns()}.zip';tmp=q.with_suffix('.partial')
        shutil.copyfile(dest,tmp)
        if sha(tmp)!=sha(dest):raise RuntimeError('Drive copy digest mismatch')
        verified_zip(tmp,PROTOCOL);os.replace(tmp,q)
        print('DRIVE_BACKUP_OK',q,flush=True)
        valid=[]
        for old in sorted(d.glob('gpt2_c7_functional_gate_results_*.zip')):
            try:verified_zip(old,PROTOCOL);valid.append(old)
            except Exception:continue
        for old in valid[:-3]:old.unlink()
    print('RESULT_ARCHIVE='+str(dest),flush=True)
    PACK_SECONDS+=time.perf_counter()-pack_started

def load_obj(z,name):return torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)

def vector(value):
    if isinstance(value,dict):value=value['theta']
    if not isinstance(value,torch.Tensor) or value.ndim!=1 or not torch.isfinite(value).all():
        raise ValueError('Finite flat parameter tensor required')
    return value.detach().cpu().float()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=out/'parent_c6.zip';parent_id=validate_c6(parent)
    device,gpu=a100_device(torch)
    ident={'protocol':PROTOCOL,'parent_c6_sha256':C6_SHA,'settings':SETTINGS,
           'source_hashes':{p.name:sha(p) for p in ROOT.iterdir() if p.is_file()},
           'gpu':gpu,'torch':torch.__version__,'transformers':transformers.__version__,
           'peft':peft.__version__,'numpy':np.__version__,'model_revision':REVISION,
           'development_only':True,'external_jacobian_and_correction':True,
           'internal_parameter_update':False,'controller_dispatch_tested':False}
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:
        raise RuntimeError('C7 runtime/source identity changed; refusing guessed resume')
    dump(out/'identity.json',ident)
    if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    if (out/'SUMMARY.json').exists():
        print('C7 already complete; no repeated evaluation.',flush=True)
        print((out/'SUMMARY.json').read_text(),flush=True);pack(out,args.drive_dir);return
    pack(out,args.drive_dir)
    counters=json.loads((out/'COUNTERS.json').read_text()) if (out/'COUNTERS.json').exists() else {
        'jacobian_builds':0,'response_autograd_calls':0,'head_training_steps':0,'memory_loss_optimiser_steps':0}
    cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
    w.set_seed(82931)
    print('C7 Loading pinned GPT-2 once; frozen C6 baseline recheck, functional training gate, then paired refinement.',flush=True)
    tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    frozen={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}

    def frozen_check():
        if any(p._version!=frozen[n] for n,p in model.named_parameters() if n in frozen):
            raise RuntimeError('Frozen base/A changed')
        if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Unexpected trainable memory parameter')

    with zipfile.ZipFile(parent) as c6,zipfile.ZipFile(io.BytesIO(c6.read('parent_c5_capsule.zip'))) as c5,zipfile.ZipFile(io.BytesIO(c5.read('parent_v5.zip'))) as v5:
        pi=json.loads(v5.read('parent/identity.json'));ah=hashlib.sha256()
        for n,p in model.named_parameters():
            if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
        if ah.hexdigest()!=pi['lora_A_sha256'] or getattr(model.config,'_commit_hash',None)!=REVISION:
            raise RuntimeError('Original model/LoRA-A chart identity mismatch')
        if [list(p.shape) for p in params]!=pi['chart_shapes']:raise RuntimeError('Parent chart shape mismatch')
        if w.param_flat(params).abs().max()!=0:raise RuntimeError('Expected original zero-B reference')
        symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
        if json.loads(json.dumps(symbols))!=pi['symbols']:raise RuntimeError('Read token identity mismatch')
        rng=random.Random(82931);anchors=rng.sample(w.ANCHORS,cfg['anchor_count'])
        response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
        with torch.no_grad():
            logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
        _,exact,_,dev=audit.prompts()

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
            print('C7 BRANCH',path.name,name,json.dumps({'pass':v.get('episode_pass',False),
                  'bits':v.get('bits'),'drift':v.get('drift'),'kl':v.get('kl'),'gates':v.get('gates')}),flush=True)
            return v

        def correct(path,initial,target,start):
            state_file=path/'CORRECTION_STATE.pt';history_file=path/'CORRECTION_HISTORY.json'
            if state_file.exists():
                cp=torch.load(state_file,map_location='cpu',weights_only=True)
                current=vector(cp);begin=int(cp['iteration']);history=cp['history']
                if not 0<=begin<=SETTINGS['correction_max_steps']:raise RuntimeError('Invalid correction progress')
                print('C7 CORRECTION RESTORE',path.name,'actual_step',begin,flush=True)
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
                print('C7 CORRECTION',path.name,'step',step+1,'accepted',accepted,
                      'drift',trials[-1]['drift'],'KL',trials[-1]['kl'],flush=True)
                if (step+1)%SETTINGS['correction_checkpoint_every']==0:pack(out,args.drive_dir)
                if not accepted:break
            dump(history_file,history)
            return current

        prepared={case:vector(load_obj(v5,case+'/prepared.pt')) for case in CASES}
        teachers={case:vector(load_obj(v5,case+'/state.pt')) for case in CASES}
        starts={case:json.loads(v5.read(case+'/PREPARATION.json'))['target_bits'] for case in CASES}
        targets={case:target_from_reads(starts[case]) for case in CASES}
        spec=chart_spec(model,params)
        dump(out/'CHART_SPEC.json',spec)
        all_batch=tok(anchors+exact+dev,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
        exact_batch=tok(exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)

        @torch.no_grad()
        def features(theta,desired):
            w.assign_flat(params,theta.to(device))
            result=model(**exact_batch,use_cache=False,output_hidden_states=True)
            last=exact_batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=device)
            hidden=result.hidden_states[-1][rows,last].reshape(-1)
            logits=result.logits[rows,last][:,ids]
            margins=logits[:,1]-logits[:,0]
            return torch.cat([hidden,margins,torch.tensor([desired],device=device,dtype=hidden.dtype)]).detach().cpu()

        w.assign_flat(params,torch.zeros_like(prepared[CASES[0]]).to(device))
        theta_probe=prepared[CASES[0]].to(device).clone().requires_grad_(True)
        actual=last_logits(model,all_batch,theta_probe,spec)
        grad=torch.autograd.grad((actual[len(anchors):,ids[1]]-actual[len(anchors):,ids[0]]).sum(),theta_probe)[0]
        if not torch.isfinite(grad).all() or not grad.norm()>0:
            raise RuntimeError('Actual GPT-2 functional chart gradient missing/nonfinite')
        w.assign_flat(params,theta_probe.detach())
        with torch.no_grad():committed=w.next_logits(model,tok,anchors+exact+dev,device)
        mismatch=float((actual.detach()-committed).abs().max())
        if mismatch>1e-4:raise RuntimeError('Functional and physically committed chart forward differ')
        dump(out/'ACTUAL_FUNCTIONAL_AUDIT.json',{'gradient_norm':float(grad.norm()),
             'maximum_logit_mismatch':mismatch,'tolerance':1e-4,'frozen_base_A_versions_unchanged':True})
        frozen_check();del actual,grad,theta_probe,committed
        print('C7 ACTUAL_FUNCTIONAL_AUDIT_OK',mismatch,flush=True)

        pool={};reference=out/'references';reference.mkdir(exist_ok=True)
        # All teacher/preparation gates are rechecked using actual frozen GPT-2 forward passes.
        # Only the saved training-fold tensors enter losses and normalisation.
        def state_data(sid):
            return (vector(load_obj(c5,'pool/'+sid+'/prepared.pt')),
                    vector(load_obj(c5,'pool/'+sid+'/teacher.pt')),
                    load_obj(c5,'pool/'+sid+'/features.pt')['features'])
        for sid in STATES:
            case=sid.split('__')[0];prep,teacher,feature=state_data(sid)
            recordfile=reference/(sid+'.json')
            if recordfile.exists():pool[sid]=json.loads(recordfile.read_text());continue
            prepcheck=evaluate(prep,starts[case]);teachercheck=evaluate(teacher,targets[case],starts[case])
            regenerated=features(prep,targets[case][3]);err=float((feature-regenerated).abs().max())
            if err>1e-4:raise RuntimeError('Saved feature identity mismatch')
            r={'preparation':prepcheck,'teacher':teachercheck,'feature_max_abs_difference':err,
               'usable':bool(prepcheck['pass'] and teachercheck['episode_pass'])}
            dump(recordfile,r);pool[sid]=r
            print('C7 REFERENCE',sid,'usable',r['usable'],'feature_error',err,flush=True)
        if not all(r['usable'] for r in pool.values()):
            dump(out/'SUMMARY.json',{'protocol':PROTOCOL,'complete':True,'decision':'C7_REFERENCE_RECHECK_INCONCLUSIVE',
                 'references':pool,'development_only':True,'full_line_c_confirmation':False})
            pack(out,args.drive_dir);return
        pack(out,args.drive_dir)

        def checkpoint_save(folder,head,optimiser,norm,steps,history,stage,fold,train_ids,start_hash):
            cpfile=folder/('checkpoint_%06d.pt'%steps)
            save_tensor(cpfile,{'head':head.state_dict(),'optimiser':optimiser.state_dict(),
              'normalisation':norm,'steps':steps,'history':history,'stage':stage,'fold':fold,
              'train_ids':list(train_ids),'start_head_sha256':start_hash})
            dump(folder/'PROGRESS.json',{'steps':steps,'checkpoint':cpfile.name,'checkpoint_sha256':sha(cpfile)})
            for old in folder.glob('checkpoint_*.pt'):
                if old!=cpfile:old.unlink()

        def checkpoint_load(folder):
            p=json.loads((folder/'PROGRESS.json').read_text());cpfile=folder/p['checkpoint']
            if sha(cpfile)!=p['checkpoint_sha256']:raise RuntimeError('Actual checkpoint digest mismatch')
            cp=torch.load(cpfile,map_location='cpu',weights_only=True)
            if cp['steps']!=p['steps']:raise RuntimeError('Progress and checkpoint steps differ')
            return cp

        def head_hash(head):
            return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()

        def new_head(state,input_dim,output_dim):
            head=WriterHead(input_dim,output_dim,SETTINGS['head_width']).to(device);head.load_state_dict(state)
            return head

        def refine(fold,arm,train_ids,data,Xn,Yn,norm,warm_state,warm_sha):
            folder=out/'heads'/fold/arm;folder.mkdir(parents=True,exist_ok=True)
            head=new_head(warm_state,Xn.shape[1],Yn.shape[1]);head.train()
            if head_hash(head)!=warm_sha:raise RuntimeError('Paired initial head identity mismatch')
            spec_record={'fold':fold,'training_ids':list(train_ids),'warm_head_sha256':warm_sha,
             'normalisation_from_training_fold_only':True,'optimiser_reset_at_stage_boundary':True,'settings':SETTINGS}
            dump(folder/'TRAINING_SPEC.json',spec_record)
            optimiser=torch.optim.AdamW(head.parameters(),lr=SETTINGS['head_lr'],weight_decay=0.)
            history=[];begin=0
            if (folder/'PROGRESS.json').exists():
                cp=checkpoint_load(folder)
                if cp['stage']!=arm or cp['fold']!=fold or cp['train_ids']!=list(train_ids) or cp['start_head_sha256']!=warm_sha:
                    raise RuntimeError('Refinement identity mismatch')
                head.load_state_dict(cp['head']);optimiser.load_state_dict(cp['optimiser']);begin=int(cp['steps']);history=cp['history']
                if not 0<=begin<=SETTINGS['head_steps']:raise RuntimeError('Invalid refinement progress')
                print('C7 HEAD_RESTORE',fold,arm,begin,flush=True)
            else:checkpoint_save(folder,head,optimiser,norm,0,history,arm,fold,train_ids,warm_sha)
            for step in range(begin,SETTINGS['head_steps']):
                optimiser.zero_grad(set_to_none=True);pred=head(Xn);mse=(pred-Yn).square().mean();loss=mse;terms=None
                if arm=='functional':
                    j=step%len(train_ids);case=train_ids[j].split('__')[0]
                    candidate=data[j][0].to(device)+pred[j]*norm['output_scale']
                    output=last_logits(model,all_batch,candidate,spec)
                    terms=objective_terms(output,len(anchors),response_ids,logits0,ids,torch.tensor(targets[case],device=device))
                    loss=combined_loss(mse,terms,SETTINGS['loss_weights'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite refinement objective')
                loss.backward();grads=[p.grad for p in head.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid writer gradients')
                gn=torch.nn.utils.clip_grad_norm_(head.parameters(),SETTINGS['gradient_clip']);optimiser.step()
                counters['head_training_steps']+=1
                if arm=='functional':counters['functional_head_steps']=counters.get('functional_head_steps',0)+1
                frozen_check()
                if (step+1)%25==0 or step==begin:
                    r={'step':step+1,'full_batch_mse':float(mse.detach()),'total_loss':float(loss.detach()),'gradient_norm':float(gn)}
                    if terms:r.update({n:float(v.detach()) for n,v in terms.items()})
                    history.append(r);print('C7 REFINE',fold,arm,json.dumps(r),flush=True)
                if (step+1)%SETTINGS['head_checkpoint_every']==0:
                    checkpoint_save(folder,head,optimiser,norm,step+1,history,arm,fold,train_ids,warm_sha)
                    dump(out/'COUNTERS.json',counters)
                if (step+1)%SETTINGS['archive_training_every']==0:pack(out,args.drive_dir)
            head.eval()
            for p in head.parameters():p.requires_grad_(False)
            return head

        contexts={};training_gates={};baseline_rows={}
        dump(out/'PARENT_C6_DECISION.json',json.loads(c6.read('SUMMARY.json')))
        # Establish eligibility from training states only, for all four heads, before any excluded input is evaluated.
        for fi,fold in enumerate(CASES):
            train_ids=fold_states(fold);test_ids=fold_states(fold,False);data=[state_data(sid) for sid in train_ids]
            X=torch.stack([r[2] for r in data]).to(device);Y=torch.stack([r[1]-r[0] for r in data]).to(device)
            base='heads/'+fold+'/warm/';progress=json.loads(c6.read(base+'PROGRESS.json'))
            parent_cp=load_obj(c6,base+progress['checkpoint'])
            if parent_cp['steps']!=1200 or parent_cp['stage']!='warm' or parent_cp['train_ids']!=list(train_ids) or parent_cp['fold']!=fold:
                raise RuntimeError('Actual C6 head split/progress mismatch')
            expected=fit_normalisation(X,Y);norm={n:v.to(device) for n,v in parent_cp['normalisation'].items()}
            if any(not torch.equal(norm[n],expected[n]) for n in norm):raise RuntimeError('Training normalisation mismatch')
            Xn=normalise(X,norm);Yn=Y/norm['output_scale'];head=new_head(parent_cp['head'],Xn.shape[1],Yn.shape[1]);head.eval()
            for p in head.parameters():p.requires_grad_(False)
            with torch.no_grad():mse=float((head(Xn)-Yn).square().mean())
            episodes=[]
            for sid in train_ids:
                print('C7 BASELINE',fold,sid,'training',flush=True)
                folder=out/'baseline'/fold/sid;folder.mkdir(parents=True,exist_ok=True);case=sid.split('__')[0]
                if (folder/'endpoint.json').exists():v=json.loads((folder/'endpoint.json').read_text())
                else:
                    prep,_,feat=state_data(sid);raw=prep+one_forward_proposal(head,feat.to(device),norm).cpu()
                    branch(folder,'raw',raw,targets[case],starts[case]);theta=correct(folder,raw,targets[case],starts[case])
                    v=branch(folder,'endpoint',theta,targets[case],starts[case])
                episodes.append(v);baseline_rows[fold+'__'+sid]={'fold':fold,'state':sid,'cohort':'training','variant':int(sid[-1]),'endpoint':v}
            gate={'eligible':functional_fit_gate(episodes),'training_passed':sum(v['episode_pass'] for v in episodes),
                  'n_training':9,'mse_diagnostic':mse,'mse_not_used_for_eligibility':True,
                  'excluded_input_used_for_eligibility':False,'head_sha256':head_hash(head)}
            gate_path=out/'training_gate'/(fold+'.json')
            if gate_path.exists() and json.loads(gate_path.read_text())!=gate:raise RuntimeError('Training-only eligibility changed on resume')
            dump(gate_path,gate);training_gates[fold]=gate
            contexts[fold]={'head':head,'norm':norm,'Xn':Xn,'Yn':Yn,'data':data,'train_ids':train_ids,'test_ids':test_ids}
            print('C7 TRAINING_GATE',fold,json.dumps(gate),flush=True);pack(out,args.drive_dir)
        # The frozen baseline is always measured on all four excluded inputs, including the non-eligible fold.
        for fold in CASES:
            c=contexts[fold]
            for sid in c['test_ids']:
                print('C7 BASELINE',fold,sid,'excluded_development',flush=True)
                folder=out/'baseline'/fold/sid;folder.mkdir(parents=True,exist_ok=True);case=sid.split('__')[0]
                if (folder/'endpoint.json').exists():v=json.loads((folder/'endpoint.json').read_text())
                else:
                    prep,_,feat=state_data(sid);raw=prep+one_forward_proposal(c['head'],feat.to(device),c['norm']).cpu()
                    branch(folder,'raw',raw,targets[case],starts[case]);theta=correct(folder,raw,targets[case],starts[case])
                    v=branch(folder,'endpoint',theta,targets[case],starts[case])
                baseline_rows[fold+'__'+sid]={'fold':fold,'state':sid,'cohort':'excluded_development','variant':int(sid[-1]),'endpoint':v}
            pack(out,args.drive_dir)
        dump(out/'FROZEN_BASELINE.json',baseline_rows)
        rows={}
        for fi,fold in enumerate(CASES):
            c=contexts[fold]
            if not training_gates[fold]['eligible']:
                print('C7 SKIP_REFINEMENT',fold,'actual training function gate failed; frozen excluded baseline retained',flush=True);continue
            train_ids=c['train_ids'];test_ids=c['test_ids'];norm=c['norm']
            warm_state={n:v.detach().cpu().clone() for n,v in c['head'].state_dict().items()};warm_sha=head_hash(c['head'])
            trained={arm:refine(fold,arm,train_ids,c['data'],c['Xn'],c['Yn'],norm,warm_state,warm_sha) for arm in arm_order(fi)}
            a=json.loads((out/'heads'/fold/'parameter_mse'/'TRAINING_SPEC.json').read_text())
            b=json.loads((out/'heads'/fold/'functional'/'TRAINING_SPEC.json').read_text())
            if a!=b:raise RuntimeError('Matched refinement specifications differ')
            for sid in train_ids+test_ids:
                print('C7 EVALUATION',fold,sid,'training' if sid in train_ids else 'excluded_development',flush=True)
                path=out/'evaluation'/fold/sid;path.mkdir(parents=True,exist_ok=True)
                if (path/'COMPLETE.json').exists():rows[fold+'__'+sid]=json.loads((path/'COMPLETE.json').read_text());continue
                prep,_,feat=state_data(sid);case=sid.split('__')[0];results={};raw_results={}
                for arm in arm_order(fi):
                    folder=path/arm;folder.mkdir(exist_ok=True)
                    if (folder/'endpoint.json').exists():
                        results[arm]=json.loads((folder/'endpoint.json').read_text());raw_results[arm]=json.loads((folder/'raw.json').read_text());continue
                    raw=prep+one_forward_proposal(trained[arm],feat.to(device),norm).cpu()
                    if not torch.isfinite(raw).all():raise RuntimeError('Nonfinite candidate')
                    raw_results[arm]=branch(folder,'raw',raw,targets[case],starts[case]);theta=correct(folder,raw,targets[case],starts[case])
                    results[arm]=branch(folder,'endpoint',theta,targets[case],starts[case])
                r={'fold':fold,'state':sid,'variant':int(sid[-1]),'cohort':'training' if sid in train_ids else 'excluded_development',
                   'parameter_mse':results['parameter_mse'],'functional':results['functional'],
                   'raw':raw_results,'teacher_fallback_used':False,'training_function_eligible':True}
                dump(path/'COMPLETE.json',r);rows[fold+'__'+sid]=r
                if len(rows)%4==0:pack(out,args.drive_dir)
            del trained;gc.collect();torch.cuda.empty_cache()
        eligible=[f for f in CASES if training_gates[f]['eligible']]
        baseline={kind:counts_single([r['endpoint'] for r in baseline_rows.values() if r['cohort']==kind]) for kind in ('training','excluded_development')}
        baseline_original=counts_single([r['endpoint'] for r in baseline_rows.values() if r['cohort']=='excluded_development' and r['variant']==0])
        cohorts={kind:matched_counts([r for r in rows.values() if r['cohort']==kind]) for kind in ('training','excluded_development')}
        originals=matched_counts([r for r in rows.values() if r['cohort']=='excluded_development' and r['variant']==0])
        committed=sum(json.loads(p.read_text())['steps'] for p in (out/'heads').rglob('PROGRESS.json')) if (out/'heads').exists() else 0
        summary={'protocol':PROTOCOL,'complete':True,'decision':'C7_DEVELOPMENT_FUNCTIONAL_GATE_COMPARISON_COMPLETE',
         'frozen_baseline':baseline,'frozen_baseline_original_four_inputs':baseline_original,
         'training_gates':training_gates,'eligible_folds':eligible,'requested_folds':list(CASES),
         'paired_cohorts':cohorts,'paired_original_excluded_inputs':originals,'paired_original_denominator':len(eligible),
         'paired_refinement_not_run_on_folds':[f for f in CASES if f not in eligible],
         'n_completed':len(rows),'persisted_committed_new_head_steps':committed,
         'new_warm_training_steps':0,'new_controller_training_steps':0,'counters':counters,'settings':SETTINGS,
         'parent_C6_decision_unchanged':json.loads(c6.read('SUMMARY.json'))['decision'],
         'scope':'New development protocol after observing C6. All four frozen heads have an excluded-input baseline. Only training-function-qualified heads enter paired 200-step refinement; subset denominators explicit. MSE remains a loss/diagnostic, not an eligibility threshold. Fixed C6 weights, training-only normalisation and unchanged read/response/KL acceptance. No full-parent-C5 revalidation claim from capsule alone.',
         'positive_references_all_pass':True,'development_only':True,'fresh_evaluation':False,'independent_seeds_tested':False,
         'full_line_c_confirmation':False,'internal_parameter_update':False,'controller_dispatch_tested':False}
        dump(out/'SUMMARY.json',summary);pack(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
