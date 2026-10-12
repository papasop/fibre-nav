"""Audit archived deltas using real GPT-2 derivatives and finite forwards. No optimiser."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,zipfile
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
from runtime_a100 import a100_device
from c8_protocol import *
from diagnostic_math import feature_coverage,span_basis,delta_comparison,linear_vs_finite,committed_delta
ROOT=Path(__file__).resolve().parent

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)

def save_tensor(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);os.replace(temp,path)

def load_obj(z,name):return torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)

def vector(obj):
    value=obj['theta'] if isinstance(obj,dict) else obj
    if not isinstance(value,torch.Tensor) or value.ndim!=1 or value.dtype!=torch.float32 or value.numel()!=36864 or not torch.isfinite(value).all():
        raise ValueError('Expected finite original 36864-coordinate float32 chart state')
    return value.detach().cpu().clone()

def median_defined(values):
    values=[x for x in values if x is not None]
    return float(np.median(values)) if values else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='')
    args=ap.parse_args();out=Path(args.out)
    lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=validate_c7(out/'parent_c7.zip');device,gpu=a100_device(torch)
    ident={'protocol':PROTOCOL,'parent_c7_sha256':info['sha256'],'settings':SETTINGS,'source_hashes':source_hashes(ROOT),
      'gpu':gpu,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
      'numpy':np.__version__,'model_revision':REVISION,'development_only':True,
      'external_jacobian_autograd':True,'new_training_steps':0,'new_correction_steps':0}
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:
        raise RuntimeError('C8 runtime/source identity changed; refusing guessed resume')
    dump(out/'identity.json',ident)
    if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_package.py'))
    dump(out/'PARENT_C7_SUMMARY.json',info['summary'])
    if (out/'SUMMARY.json').exists():
        print('C8 already complete; no repeated model diagnostics.',flush=True)
        print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,args.drive_dir);return
    snapshot(out,args.drive_dir)
    cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
    w.set_seed(82931);print('C8 Loading pinned GPT-2 once; 12 saved states, 120 candidates if three folds eligible; zero new training.',flush=True)
    tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
    for p in model.parameters():p.requires_grad_(False)
    frozen={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}
    origin=w.param_flat(params).cpu()
    if origin.abs().max()!=0:raise RuntimeError('Original zero-B reference required')

    def frozen_check():
        if any(p._version!=frozen[n] for n,p in model.named_parameters() if n in frozen):raise RuntimeError('Frozen base/A changed')
        if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Unexpected trainable parameter after audit')

    with ExitStack() as stack:
        c7=stack.enter_context(zipfile.ZipFile(out/'parent_c7.zip'))
        c6=stack.enter_context(zipfile.ZipFile(io.BytesIO(c7.read('parent_c6.zip'))))
        cap=stack.enter_context(zipfile.ZipFile(io.BytesIO(c6.read('parent_c5_capsule.zip'))))
        v5=stack.enter_context(zipfile.ZipFile(io.BytesIO(cap.read('parent_v5.zip'))))
        pi=json.loads(v5.read('parent/identity.json'));ah=hashlib.sha256()
        for n,p in model.named_parameters():
            if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
        if ah.hexdigest()!=pi['lora_A_sha256'] or getattr(model.config,'_commit_hash',None)!=REVISION:
            raise RuntimeError('Original model/LoRA-A identity mismatch')
        if [list(p.shape) for p in params]!=pi['chart_shapes']:raise RuntimeError('Original chart shapes changed')
        symbols=w.single_token_pair(tok);ids=[x[1] for x in symbols]
        if json.loads(json.dumps(symbols))!=pi['symbols']:raise RuntimeError('Read token identity mismatch')
        rng=random.Random(82931);anchors=rng.sample(w.ANCHORS,cfg['anchor_count'])
        response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
        _,exact,_,dev=audit.prompts()
        with torch.no_grad():logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
        starts={case:json.loads(v5.read(case+'/PREPARATION.json'))['target_bits'] for case in CASES}
        targets={case:target_from_reads(starts[case]) for case in CASES}
        pool={}
        for sid in STATES:
            pool[sid]={'prep':vector(load_obj(cap,'pool/'+sid+'/prepared.pt')),
                       'teacher':vector(load_obj(cap,'pool/'+sid+'/teacher.pt')),
                       'feature':load_obj(cap,'pool/'+sid+'/features.pt')['features'].detach().cpu()}
            if pool[sid]['feature'].shape!=(3077,) or not torch.isfinite(pool[sid]['feature']).all():raise ValueError('Original features invalid')

        @torch.no_grad()
        def measure(theta,target,start=None):
            w.assign_flat(params,theta.to(device))
            al=w.next_logits(model,tok,anchors,device)
            el=w.next_logits(model,tok,exact,device);dl=w.next_logits(model,tok,dev,device)
            R=al[:,response_ids].mean(0);H=el[:,ids[1]]-el[:,ids[0]];Dh=dl[:,ids[1]]-dl[:,ids[0]]
            signs=torch.tensor([2*b-1 for b in target],device=device)
            ex=(H*signs).cpu().tolist();de=(Dh*signs.repeat(3)).cpu().tolist()
            bits=el[:,ids].argmax(-1).cpu().tolist()
            akl=(logits0.softmax(-1)*(logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1)
            diff=al[:,response_ids]-logits0[:,response_ids]
            drift=float((R-response0).abs().max());kl=float(akl.mean())
            ev={'exact_margins':ex,'heldout_margins':de,'minimum_exact_margin':min(ex),'minimum_heldout_margin':min(de),
                'exact_accuracy':sum(x>0 for x in ex)/4,'heldout_accuracy':sum(x>0 for x in de)/12}
            v={'bits':bits,'evaluation':ev,'drift':drift,'kl':kl,'R':R.cpu().double().tolist(),'H':H.cpu().double().tolist(),
               'per_anchor_selected_logit_drift':diff.abs().max(1).values.cpu().tolist(),
               'per_anchor_KL':akl.cpu().tolist(),'per_anchor_values_diagnostic_not_new_acceptance_gates':True}
            v['pass']=endpoint_gate(v,target)
            v['gates']={'exact_read_margin':min(ex)>=.05,'development_read_margin':min(de)>=.05,
                        'bits_match':bits==target,'response':drift<=.002,'KL':kl<=.005}
            if start is not None:
                v['gates'].update(output_flip=bits[3]!=start[3],unselected_preserved=bits[:3]==start[:3]);v['episode_pass']=episode_gate(start,v)
            frozen_check();return v

        @torch.no_grad()
        def features(theta,desired):
            w.assign_flat(params,theta.to(device));batch=tok(exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
            r=model(**batch,use_cache=False,output_hidden_states=True);last=batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=device)
            hidden=r.hidden_states[-1][rows,last].reshape(-1);l=r.logits[rows,last][:,ids]
            return torch.cat([hidden,l[:,1]-l[:,0],torch.tensor([desired],device=device,dtype=hidden.dtype)]).cpu()

        def derivatives(theta):
            w.assign_flat(params,theta.to(device))
            for p in params:p.requires_grad_(True)
            try:
                with torch.enable_grad():
                    R=w.response_vector(model,tok,anchors,response_ids,device);rr=[]
                    for i in range(6):
                        gs=torch.autograd.grad(R[i],params,retain_graph=i<5)
                        rr.append(torch.cat([g.reshape(-1) for g in gs]).detach().cpu())
                    l=w.next_logits(model,tok,exact,device);H=l[:,ids[1]]-l[:,ids[0]];hh=[]
                    for i in range(4):
                        gs=torch.autograd.grad(H[i],params,retain_graph=i<3)
                        hh.append(torch.cat([g.reshape(-1) for g in gs]).detach().cpu())
                result={'J':torch.stack(rr),'DH':torch.stack(hh)}
                if any(not torch.isfinite(v).all() for v in result.values()):raise RuntimeError('Nonfinite actual Jacobian')
            finally:
                for p in params:p.requires_grad_(False);p.grad=None
            frozen_check();return result

        states={}
        for index,sid in enumerate(STATES):
            folder=out/'states'/sid;cf=folder/'COMPLETE.json';jf=folder/'jacobian.pt';case=sid.split('__')[0]
            if cf.exists():
                rec=json.loads(cf.read_text())
                if not jf.exists() or sha(jf)!=rec['jacobian_sha256']:raise RuntimeError('Saved Jacobian digest mismatch')
                states[sid]=rec;print('C8 STATE_RESTORE',sid,flush=True);continue
            prep=pool[sid]['prep'];teacher=pool[sid]['teacher'];pm=measure(prep,starts[case]);tm=measure(teacher,targets[case],starts[case])
            regenerated=features(prep,targets[case][3]);err=float((regenerated-pool[sid]['feature']).abs().max())
            if err>SETTINGS['feature_identity_tolerance']:raise RuntimeError('Original saved feature mismatch '+sid)
            if not pm['pass'] or not tm['episode_pass']:raise RuntimeError('Positive preparation/teacher control failed '+sid)
            jac=derivatives(prep);save_tensor(jf,jac)
            td=committed_delta(prep.numpy(),teacher.numpy())
            rec={'state':sid,'preparation':pm,'teacher':tm,'feature_max_abs_difference':err,'jacobian_sha256':sha(jf),
                 'teacher_response':linear_vs_finite(jac['J'].numpy(),td,pm['R'],tm['R']),
                 'teacher_readout':linear_vs_finite(jac['DH'].numpy(),td,pm['H'],tm['H']),
                 'Jacobian_evaluated_at':'actual saved preparation', 'usable':True}
            dump(cf,rec);states[sid]=rec
            print('C8 STATE',sid,'original feature verified; actual J and DH saved; positive references pass',flush=True)
            if (index+1)%4==0:snapshot(out,args.drive_dir)
        contexts={}
        for fold in CASES:
            base='heads/'+fold+'/warm/';p=json.loads(c6.read(base+'PROGRESS.json'));cp=load_obj(c6,base+p['checkpoint'])
            train_ids=list(fold_states(fold));norm=cp['normalisation']
            if cp['fold']!=fold or cp['train_ids']!=train_ids or cp['steps']!=1200:raise RuntimeError('Training feature normalisation split changed')
            mean=norm['mean'].cpu().double().numpy();scale=norm['scale'].cpu().double().numpy()
            if np.any(scale<=0):raise RuntimeError('Invalid saved training scale')
            X=np.stack([(pool[s]['feature'].double().numpy()-mean)/scale for s in train_ids])
            Y=np.stack([committed_delta(pool[s]['prep'].numpy(),pool[s]['teacher'].numpy()) for s in train_ids])
            basis,svd=span_basis(Y,SETTINGS['span_rtol'],SETTINGS['span_atol'])
            contexts[fold]={'mean':mean,'scale':scale,'X':X,'basis':basis,'train_ids':train_ids}
            coverage={}
            for sid in STATES:
                query=(pool[sid]['feature'].double().numpy()-mean)/scale
                c=feature_coverage(X,query,SETTINGS['span_rtol'],SETTINGS['span_atol'])
                c['nearest_training_state']=train_ids[c['nearest_training_index']];coverage[sid]=c
            dump(out/'coverage'/(fold+'.json'),{'fold':fold,'train_ids':train_ids,'standardisation':'saved C6 training-only mean/scale; diagnostic arithmetic float64',
                  'feature_coverage':coverage,'training_teacher_delta_span':svd,'no_head_training':True,
                  'coverage_or_numerical_rank_is_not_feasibility_certificate':True})
        rows=[]
        for index,job in enumerate(info['jobs']):
            cf=out/'candidates'/(job['key']+'.json')
            if cf.exists():rows.append(json.loads(cf.read_text()));continue
            sid=job['state'];case=sid.split('__')[0];context=contexts[job['fold']]
            prep=pool[sid]['prep'];teacher=pool[sid]['teacher'];raw=vector(load_obj(c7,job['member']+'/raw.pt'));final=vector(load_obj(c7,job['member']+'/endpoint.pt'))
            pm=states[sid]['preparation'];rm=measure(raw,targets[case],starts[case]);fm=measure(final,targets[case],starts[case])
            jac=torch.load(out/'states'/sid/'jacobian.pt',map_location='cpu',weights_only=True)
            td=committed_delta(prep.numpy(),teacher.numpy());rd=committed_delta(prep.numpy(),raw.numpy());fd=committed_delta(prep.numpy(),final.numpy())
            cd=committed_delta(raw.numpy(),final.numpy())
            archived={stage:json.loads(c7.read(job['member']+'/'+stage+'.json')) for stage in ['raw','endpoint']}
            matches={stage:bool(measured['gates']==archived[stage]['gates'] and measured['episode_pass']==archived[stage]['episode_pass'])
                     for stage,measured in [('raw',rm),('endpoint',fm)]}
            record={**job,'raw':rm,'corrected':fm,'parameter_delta':delta_comparison(rd,td,context['basis']),
                    'corrected_delta':delta_comparison(fd,td,context['basis']),
                    'raw_response':linear_vs_finite(jac['J'].numpy(),rd,pm['R'],rm['R']),
                    'raw_readout':linear_vs_finite(jac['DH'].numpy(),rd,pm['H'],rm['H']),
                    'corrected_response':linear_vs_finite(jac['J'].numpy(),fd,pm['R'],fm['R']),
                    'corrected_readout':linear_vs_finite(jac['DH'].numpy(),fd,pm['H'],fm['H']),
                    'saved_correction_effect':{'norm':float(np.linalg.norm(cd)),
                     'actual_readout_change':(np.asarray(fm['H'])-np.asarray(rm['H'])).tolist(),
                     'actual_response_change':(np.asarray(fm['R'])-np.asarray(rm['R'])).tolist(),
                     'no_new_correction_computed':True},
                    'archived_gate_agreement':matches,
                    'archived_metrics_difference':{stage:{k:measured[k]-archived[stage][k] for k in ['drift','kl']}
                       for stage,measured in [('raw',rm),('endpoint',fm)]},
                    'linearisation_state':sid+' saved preparation, for both raw and corrected total displacements',
                    'teacher_fallback_used':False}
            dump(cf,record);rows.append(record)
            print('C8 CANDIDATE',index+1,'/',len(info['jobs']),job['key'],
                  'raw/corrected',rm['episode_pass'],fm['episode_pass'],'output flips',rm['gates']['output_flip'],fm['gates']['output_flip'],flush=True)
            if (index+1)%SETTINGS['snapshot_every_candidates']==0:snapshot(out,args.drive_dir)
        aggregate={}
        for family,arms in [('frozen_baseline',['baseline']),('paired',list(ARMS))]:
            aggregate[family]={}
            for arm in arms:
                aggregate[family][arm]={}
                for cohort in ['training','excluded_development']:
                    rr=[r for r in rows if r['family']==family and r['arm']==arm and r['cohort']==cohort]
                    aggregate[family][arm][cohort]={'n':len(rr),
                      'raw_passed':sum(r['raw']['episode_pass'] for r in rr),'corrected_passed':sum(r['corrected']['episode_pass'] for r in rr),
                      'corrected_gate_counts':{g:sum(r['corrected']['gates'][g] for r in rr) for g in ['bits_match','exact_read_margin','development_read_margin','response','KL','output_flip','unselected_preserved']},
                      'median_delta_cosine_with_teacher':median_defined([r['parameter_delta']['cosine_with_teacher'] for r in rr]),
                      'median_relative_teacher_delta_error':median_defined([r['parameter_delta']['relative_teacher_error'] for r in rr])}
        mismatch=sum(not all(r['archived_gate_agreement'].values()) for r in rows)
        summary={'protocol':PROTOCOL,'complete':True,'decision':'C8_DEVELOPMENT_EXTRAPOLATION_DIAGNOSIS_COMPLETE',
                 'n_prepared_states':len(states),'n_saved_candidates':len(rows),'n_endpoint_forwards':2*len(rows),
                 'eligible_paired_folds':info['eligible'],'aggregate':aggregate,'archived_gate_disagreement_candidates':mismatch,
                 'actual_jacobian_rows_saved':10*len(states),'positive_references_all_pass':True,
                 'new_training_steps':0,'new_correction_steps':0,'candidate_regeneration':False,
                 'teacher_fallback_used':False,'fresh_evaluation':False,'internal_parameter_update':False,
                 'controller_dispatch_tested':False,'certified_curvature_bounds':False,
                 'C7_parent_decision_unchanged':info['summary']['decision'],
                 'interpretation':'Coverage, teacher-alignment, local linear changes and actual finite effects are post hoc diagnostics. No generalisation, causality, full Line C or native-write certificate is inferred.'}
        w.assign_flat(params,origin.to(device));frozen_check();dump(out/'SUMMARY.json',summary)
        snapshot(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
