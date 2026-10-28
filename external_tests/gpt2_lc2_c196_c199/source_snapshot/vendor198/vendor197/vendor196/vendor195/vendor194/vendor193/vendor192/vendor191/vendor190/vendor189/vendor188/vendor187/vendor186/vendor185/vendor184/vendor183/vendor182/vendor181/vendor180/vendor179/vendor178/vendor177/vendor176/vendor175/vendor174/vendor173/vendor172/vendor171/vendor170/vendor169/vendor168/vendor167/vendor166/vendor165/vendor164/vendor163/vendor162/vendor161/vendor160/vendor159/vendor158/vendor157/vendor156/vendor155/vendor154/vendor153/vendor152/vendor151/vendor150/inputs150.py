"""Six original pins; exact row/tensor bindings and recorded native gate reconstruction.
C149 is a sealed prior audit receipt, not claimed as re-executed across NumPy builds.
All new model replays still precede scalar backward calls.
"""
from core150 import *
from audit147 import source_check


def read_bound_inputs(archives,c,out):
    z=archives['C148'];counts={k:a.verify_manifest() for k,a in archives.items()}
    source_check(archives['C149'],c['source149'],'C149');source_check(z,c['binding149']['source148'],'C148')
    prior=archives['C149'].json('SUMMARY.json');compare(prior['complete'],True);compare(prior['errors'],[]);compare(prior['decision'],'C149_READONLY_ROUTING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(prior['parent_decision'],z.json('SUMMARY.json')['decision'])
    for k in auditcore.INPUT_KEYS:compare(archives['C149'].json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C149 original link/'+k)
    for k in old.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C148 original link/'+k)
    for k in ('C146','C148'):compare(archives[k].json('ALL84_TRAINING_REGISTRY_FROZEN.json'),registry(),'Exact original registry/'+k)
    base_registry=[{k:v for k,v in sp.items() if k!='C146_membership'} for sp in registry()];compare(archives['C145'].json('ALL84_EXPOSED_TRAINING_REGISTRY.json'),base_registry,'C145 exact original registry')
    arrays=z.arrays('ALL84_TRAINING_INPUTS.npz');names={sp['state_id']+'__'+k for sp in registry() for k in ('prepared','raw','features')};need(set(arrays)==names,'Exact full84 tensor inventory');points={sp['state_id']:{k:arrays[sp['state_id']+'__'+k] for k in ('prepared','raw','features')} for sp in registry()}
    for origin,filename in [('C143','ALL84_TRAINING_INPUTS.npz'),('C145','ALL84_FROZEN_TRAINING_INPUTS.npz'),('C146','ALL84_TRAINING_INPUTS.npz')]:
        duplicate=archives[origin].arrays(filename);need(set(duplicate)==names,'All84 origin inventory/'+origin)
        for n in names:exact(duplicate[n],arrays[n],'Exact originating coordinate/feature row/'+origin+'/'+n)
    expected143=[dict(sp,prepared_sha256=safe.tensor_hash(points[sp['state_id']]['prepared']),raw_sha256=safe.tensor_hash(points[sp['state_id']]['raw'])) for sp in base_registry];compare(archives['C143'].json('ALL84_TRAINING_REGISTRY_FROZEN.json'),expected143,'Original C143 coordinate-bound registry')
    V=z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'];a=z.arrays('INITIAL_C146_CONTROL.npz')['a'];A=z.arrays('FINAL_context_routed_STEP12.npz')['parameters'];router=z.arrays('FROZEN_TRAINING_ONLY_ROUTER.npz');rw=z.arrays('ALL84_PREWRITE_ROUTING.npz');rank=len(a);need(rank==36 if c['production'] else 1<rank<=36,'Fixed rank');farray(V,(rank,DIM));farray(A,(3,rank));farray(rw['weights'],(84,3));farray(a,(rank,));exact(a,archives['C146'].arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],'Actual C146 fixed final initialization')
    for k in ('C143','C146'):exact(V,archives[k].arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'],'Shared actual V/'+k)
    writer=z.arrays('FROZEN_ORIGINAL_WRITER.npz');control=z.arrays('FROZEN_ORIGINAL_CONTROL.npz');pins=z.json('MODEL_PINS_REFERENCE.json')
    for origin in ('C143','C146'):
        for filename,want in [('FROZEN_ORIGINAL_WRITER.npz',writer),('FROZEN_ORIGINAL_CONTROL.npz',control)]:
            got=archives[origin].arrays(filename);need(set(got)==set(want),'Actual head inventory/'+origin)
            for name,v in want.items():exact(got[name],v,'Original head identity/'+origin+'/'+name)
        compare(archives[origin].json('MODEL_PINS_REFERENCE.json'),pins,'Model reference/'+origin)
    for sp in registry():
        p=points[sp['state_id']];farray(p['prepared'],(DIM,));farray(p['raw'],(DIM,));farray(p['features'],(3211,))
    U=np.array([context(points[sp['state_id']]['prepared'],points[sp['state_id']]['raw'],V) for sp in registry()]);close(U,rw['u'],'All84 prewrite context',1e-12);need(set(router)=={'mean','scale','centers'},'Fixed router tensor inventory');farray(router['mean'],(rank,));farray(router['scale'],(rank,));farray(router['centers'],(3,rank));need(np.all(router['scale']>0),'Positive frozen normalizer')
    W=rw['weights'];weights={sp['state_id']:w.copy() for sp,w in zip(registry(),W)}
    for u,w in zip(U,W):close(auditcore.route(u,router),w,'Frozen route/cached weight',1e-12)
    refs={};pre=z.json('PREFLIGHT_PROGRESS.json');compare([(r['state_id'],r['kind']) for r in pre],[(sp['state_id'],kind) for sp in registry() for kind in ('prepared','raw','C143','C146')],'Complete original preflight denominator')
    for sp in registry():
        sid=sp['state_id']
        for kind in ('prepared','raw','C146'):refs[(sid,kind)]=[r['metrics'] for r in pre if r['state_id']==sid and r['kind']==kind]
    d=dict(points=points,V=V,a=archives['C143'].arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],C140_a=archives['C143'].arrays('INITIAL_C140_STEP12_CONTROL.npz')['a'],initial_a=a,initial_A=A,router=router,weights=weights,writer=writer,control=control,modelpins=pins,refs=refs)
    compare(z.json('FROZEN_INPUT_IDENTITIES.json'),old.fixed_digest(d),'Original C148 full frozen identity')
    # Exact step12 identity from original update, not a selected recovery head.
    last=z.arrays('training/context_routed/step12/UPDATE.npz');exact(A,last['after'],'Original step12 actual tensor');seal=z.json('BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json');compare(seal['parameter_digests']['context_routed'],row_digest(A));compare(seal['native_final_measurements_so_far'],0)
    for name,h in seal['artifact_seals'].items():compare(hashlib.sha256(z.raw(name)).hexdigest(),h,'Original sealed artifact/'+name)
    records=z.json('ALL672_FINAL_TRAINING_ENDPOINTS.json');compare([(r['round'],r['state_id'],r['arm']) for r in records],[(rnd,sp['state_id'],arm) for rnd in (0,1) for sp in registry() for arm in old.EVAL_ARMS]);compare(records,z.json('FINAL_ENDPOINT_PROGRESS.json'))
    heads={arm:z.arrays('FINAL_'+arm+'_STEP12.npz')['parameters'] for arm in old.TRAIN_ARMS};frozen=z.arrays('ALL336_FROZEN_FINAL_POINTS.npz');need(set(frozen)=={sp['state_id']+'__'+arm for sp in registry() for arm in old.EVAL_ARMS},'All336 parent sealed points');lookup={sp['state_id']:sp for sp in registry()};first={}
    for r in records:
        sid=r['state_id'];arm=r['arm'];p=points[sid]
        for key,v in lookup[sid].items():compare(r[key],v,'Original endpoint binding')
        e=np.zeros(rank) if arm==old.EVAL_ARMS[0] else a if arm==old.EVAL_ARMS[1] else old.effective(heads[arm],weights[sid],arm);point=p['raw'] if arm==old.EVAL_ARMS[0] else old.proposal(p['prepared'],p['raw'],V,e)[0];exact(frozen[sid+'__'+arm],point,'Parent actual frozen endpoint');compare(endpoint_gates(r['metrics'],r['case'],point,p['prepared'],True),r['gates'],'Original native gates');compare(r['training_objective_parts'],old.loss(r['metrics']['observed'],r['metrics']['KL'],point,p['prepared'],e,r['case']))
        if r['round']==0:first[(sid,arm)]=r['metrics']
        else:
            q=repeat_check(r['metrics'],first[(sid,arm)]);need(q['same_point_identity'] and q['within_replay_tolerance'],'Original repeated endpoint drift')
    for sp in registry():refs[(sp['state_id'],'C148')]=[r['metrics'] for r in records if r['state_id']==sp['state_id'] and r['arm']=='context_routed']
    sums,pairs=old.final_tables(records);compare(sums,z.json('FINAL_TRAINING_SUMMARIES.json'));compare(sums,z.json('SUMMARY.json')['final_training_summaries']);compare(sums,prior['final_training_summaries']);compare(pairs,z.json('PAIRED_ALL_ARMS.json'))
    states=archives['C149'].json('ALL84_ROUTING_CONTROL_AND_ENDPOINT_DIAGNOSTICS.json');compare([r['state_id'] for r in states],[sp['state_id'] for sp in registry()]);dump(out/'BASELINE_ALL84_ROUTING_AND_FAILURE_COMPARISON.json',states);dump(out/'C149_SEALED_AUDIT_RECEIPT.json',prior)
    for sp in registry():
        sid=sp['state_id'];p=points[sid]
        for kind in ('prepared','raw','C146','C148'):
            point=preflight_point(d,sid,kind)
            for m in refs[(sid,kind)]:
                gates,_,_=native_gates(m,sp['case'],point,kind!='prepared')
                if kind=='prepared':need(all(gates.values()),'Invalid archived preparation')
    receipt=dict(original_members_hashed=counts,C149_decision=prior['decision'],C148_decision=prior['parent_decision'],prior_C149_audit_reexecuted=False,prior_C149_original_receipt_reused=True,older_training_derivative_or_optimizer_audits_recomputed=False,all84_origin_coordinate_feature_duplicates_verified=True,all336_parent_points_and672_native_gates_recomputed=True,original_acceptance_thresholds_unchanged=True,no_hyperparameter_or_pairing_selection_from_state_diagnostics=True)
    return d,receipt


def load(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);need(set(paths)==set(INPUT_KEYS),'Exactly six pinned originals')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    with contextlib.ExitStack() as stack:
        archives={k:stack.enter_context(contextlib.closing(Archive(p))) for k,p in paths.items()}
        with auditcore.no_models(),safe.no_solvers():d,receipt=read_bound_inputs(archives,c,out)
    dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',registry())
    for name,arrays in [('ALL84_TRAINING_INPUTS.npz',{sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()}),('FROZEN_SHARED_ROW_BASIS.npz',dict(V=d['V'])),('INITIAL_C148_ROUTED_CONTROL.npz',dict(A=d['initial_A'])),('FROZEN_TRAINING_ONLY_ROUTER.npz',d['router']),('ALL84_PREWRITE_ROUTING.npz',dict(weights=np.array([d['weights'][sp['state_id']] for sp in registry()]))),('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])]:np.savez_compressed(out/name,**arrays)
    dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'FROZEN_INPUT_IDENTITIES.json',fixed_digest(d));dump(out/'PARENT_AUDIT_RECEIPT.json',receipt);return d
