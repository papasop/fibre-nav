"""Read-only six-parent binding and independent full84 C160 arithmetic audit."""
from core161 import *
from inputs160 import load as load160
from audit147 import source_check
def audit160(path,ctx,out,c,budget=lambda:None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with archive155.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source160'],'C160');s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C160_FULL84_FP64_OBJECTIVE_GRADIENT_BASELINE_COMPLETE_NO_TRAINING');compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['errors'],[])
  for k,v in dict(new_model_forwards=2019,functional_FP64_forwards=1682,new_backward_calls=168,scalar_backward_attempts=168,records_completed=1008,requests_consumed=1008,training_states_gradient_evaluated=84,new_training_steps=0,new_head_fits=0,excluded_input_states_evaluated=0).items():compare(s[k],v,'C160 denominator/'+k)
  for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','head_updated','parent_decisions_reclassified','LC2_acceptance_thresholds_changed'):compare(s[k],False)
  compare(s['original_inputs_unchanged'],{k:True for k in previous.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in previous.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C160 original link/'+k)
  compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),previous.design());chain=previous.constants(ctx['data']);saved=z.arrays('FROZEN_FULL84_CHAIN_CONSTANTS.npz');compare(set(chain),set(saved))
  for n,a in chain.items():exact(a,saved[n],'C160 actual constant/'+n)
  for n,h in z.json('SEALED_BEFORE_MODEL.json')['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C160 actual sealed artifact/'+n)
  refs=z.arrays('FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz');exact(refs['response'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'C160 original response reference');exact(refs['baseline_logits'],ctx['baseline_logits']['FP32_logits_cast64'],'C160 original KL logits')
  coefs=z.arrays('SEVEN_COEFFICIENT_TENSORS.npz');points=z.arrays('ALL588_FP64_POINTS.npz');native=z.arrays('ALL588_ORIGINAL_FP32_POINTS.npz');binding=z.json('POINT_BINDINGS.json');compare(len(binding),588);compare(set(coefs),set(previous.LABELS));by={}
  for b in binding:
   budget();sid,n=b['state_id'],b['label']
   need((sid,n) not in by and sid in {s['state_id'] for s in registry()} and n in previous.LABELS,'C160 unique registered point');by[(sid,n)]=b;p=chain['P0']+float(previous.LABELS[n])*chain['deltaP'];exact(coefs[n],p,'C160 exact head');pt,e=mapped(chain,p,sid);actual=points[sid+'__'+n];close(actual,pt,'C160 continuous point arithmetic',1e-12);compare(point_hash(actual),b['FP64_point_sha256']);original=ctx['data']['points'][n+'__'+sid];exact(native[sid+'__'+n],original,'C160 actual original FP32 point');compare(safe.tensor_hash(original),b['original_FP32_point_sha256'])
  compare(set(points),{sid+'__'+n for sid,n in by});compare(set(native),set(points))
  rq=z.json('ALL1008_REQUESTS_FROZEN.json');compare(rq,previous.requests());rows=[clean_json(line.encode()) for line in z.raw('ALL1008_RECORDS.jsonl').decode().splitlines()];compare([{k:r[k] for k in ('stage','state_id','label')} for r in rows],rq);need(len(rows)==1008,'C160 complete retained records');stages={}
  for r in rows:
   stages.setdefault((r['stage'],r['label']),[]).append(r);sid,n=r['state_id'],r['label'];pt=points[sid+'__'+n]
   if r['stage'] in ('AD','finite','repeat'):
    compare(r['FP64_point_sha256'],point_hash(pt));compare(r['parameters_digest'],parameter_hash(coefs[n]));_,e=mapped(chain,coefs[n],sid);compare(r['objective_parts'],loss(r['observed'],r['KL'],pt,chain[sid+'__prepared'],e,r['case']),'C160 stored full objective recomputed',atol=1e-10,rtol=1e-9)
   else:
    pp=native[sid+'__'+n];m=archive155.full_metric(r['metrics']['observed'],r['metrics']['KL'],pp,r['case']);q=archive155.endpoint_gates(m,r['case'],pp,chain[sid+'__prepared'].astype(np.float32),True);compare(m,r['metrics']);compare(q,r['qualification'])
  gr=stages[('AD','zero')];compare(z.json('ALL84_GRADIENT_BASELINE.json'),[{k:v for k,v in r.items() if k!='stage'} for r in gr]);g,agg=aggregate(gr,chain['P0'].shape);gs=z.arrays('COMPLETE84_GRADIENT_BASELINE.npz');exact(gs['aggregate_gradient'],g,'C160 aggregate gradient');exact(gs['original_deltaP'],chain['deltaP'],'C160 original direction');exact(gs['response_gradients'],np.asarray([r['response_gradient'] for r in gr]),'C160 response gradients');exact(gs['guard_gradients'],np.asarray([r['guard_gradient'] for r in gr]),'C160 guard gradients');compare(agg,s['gradient_baseline_aggregation']);dot=float(np.sum(g*chain['deltaP']));compare(dot,s['aggregate_gradient_dot_original_delta']);compare(float(np.linalg.norm(g)),s['aggregate_gradient_L2'])
  panels={n:objective(stages[('finite',n)]) for n in previous.LABELS};compare(panels,s['all_panel_objectives'])
  for a,b in zip(gr,stages[('finite','zero')]):close(a['observed'],b['observed'],'C160 AD/no-grad center',2e-9);need(abs(a['KL']-b['KL'])<=2e-9,'C160 center KL')
  for n in ('zero','plus_one'):
   for a,b in zip(stages[('finite',n)],stages[('repeat',n)]):close(a['observed'],b['observed'],'C160 FP64 repeats',2e-9);need(abs(a['KL']-b['KL'])<=2e-9,'C160 FP64 repeated KL')
  for stage,key in [('native0','native_alpha0_round0'),('native1','native_alpha0_round1')]:compare(native_summary(stages[(stage,'zero')]),s[key])
  for a,b in zip(stages[('native0','zero')],stages[('native1','zero')]):r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'C160 native repeat')
  fd=[]
  for h,m,p in previous.SCALES:
   r=previous.previous.previous.previous.parent.fd(panels['zero']['objective'],panels[m]['objective'],panels[p]['objective'],h);fd.append(dict(**r,AD_directional=dot,central_minus_AD=r['central']-dot,positive_minus_AD=r['positive']-dot,negative_minus_AD=r['negative']-dot))
  compare(fd,s['three_scale_complete_objective_FD']);receipt=dict(decision=s['decision'],members_hashed=members,all84_states_and1008_records_reconstructed=True,archived_gradients_independently_recomputed=False,new_model_forwards=0,LC2_closed=False);dump(out/'C160_ARITHMETIC_AUDIT.json',receipt);return dict(**receipt,initial_aggregation=agg,initial_gradient=g,initial_native=s['native_alpha0_round0'])
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exact six original inputs')
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole ZIP before payload/'+k)
 for n,h in c['source160'].items():compare(sha(ROOT/'vendor160'/n),h,'Trusted C160 source/'+n)
 ctx=load160({k:paths[k] for k in previous.INPUT_KEYS},out/'C155_to_C159',c['parent'],budget);ctx['completed160']=audit160(paths['C160'],ctx,out/'C160',c,budget);return ctx
