"""Synthetic archive facades test bindings; production ZIP pins/manifests stay mandatory."""
import copy,tempfile,unittest,contextlib,io
from core150 import *
from inputs150 import read_bound_inputs,load
from fixture150 import worker_data,dummy_inputs
from fixture138 import metric

class ArchiveDouble:
    def __init__(self):self.data={};self.z=self
    def namelist(self):return []
    def verify_manifest(self):return len(self.data)
    def json(self,n):return copy.deepcopy(self.data[n])
    def arrays(self,n):return {k:v.copy() for k,v in self.data[n].items()}

def fixture():
    d,_=worker_data();a={k:ArchiveDouble() for k in INPUT_KEYS};z=a['C148'];c=dict(production=False,pins={k:dict(sha256=k) for k in INPUT_KEYS},source149={},binding149=dict(source148={}));base=[{k:v for k,v in sp.items() if k!='C146_membership'} for sp in registry()];arrays={sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()};A=d['initial_A'];rank=len(d['initial_a']);heads=dict(global_continuation=d['initial_a'],context_routed=A);records=[];frozen={}
    for k in ('C143','C146','C148'):
        a[k].data.update({'ALL84_TRAINING_REGISTRY_FROZEN.json':registry(),'ALL84_TRAINING_INPUTS.npz':arrays,'FROZEN_SHARED_ROW_BASIS.npz':dict(V=d['V']),'FROZEN_ORIGINAL_WRITER.npz':d['writer'],'FROZEN_ORIGINAL_CONTROL.npz':d['control'],'MODEL_PINS_REFERENCE.json':d['modelpins']})
    a['C143'].data['ALL84_TRAINING_REGISTRY_FROZEN.json']=[dict(sp,prepared_sha256=safe.tensor_hash(d['points'][sp['state_id']]['prepared']),raw_sha256=safe.tensor_hash(d['points'][sp['state_id']]['raw'])) for sp in base]
    a['C143'].data['FINAL_SHARED_CONTROL_STEP12.npz']=dict(a=d['a']);a['C143'].data['INITIAL_C140_STEP12_CONTROL.npz']=dict(a=d['C140_a']);a['C146'].data['FINAL_SHARED_CONTROL_STEP12.npz']=dict(a=d['initial_a']);a['C145'].data={'ALL84_EXPOSED_TRAINING_REGISTRY.json':base,'ALL84_FROZEN_TRAINING_INPUTS.npz':arrays}
    z.data.update({'INITIAL_C146_CONTROL.npz':dict(a=d['initial_a']),'FINAL_context_routed_STEP12.npz':dict(parameters=A),'FINAL_global_continuation_STEP12.npz':dict(parameters=d['initial_a']),'FROZEN_TRAINING_ONLY_ROUTER.npz':d['router'],'ALL84_PREWRITE_ROUTING.npz':dict(u=np.array([context(p['prepared'],p['raw'],d['V']) for p in d['points'].values()]),weights=np.array([d['weights'][sp['state_id']] for sp in registry()])),'FROZEN_INPUT_IDENTITIES.json':old.fixed_digest(d),'training/context_routed/step12/UPDATE.npz':dict(after=A),'BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json':dict(parameter_digests=dict(context_routed=row_digest(A)),native_final_measurements_so_far=0,artifact_seals={})})
    z.data['PREFLIGHT_PROGRESS.json']=[]
    for sp in registry():
        sid=sp['state_id'];p=d['points'][sid]
        for kind in ('prepared','raw','C143','C146'):
            point=p[kind] if kind in ('prepared','raw') else old.proposal(p['prepared'],p['raw'],d['V'],d['a'] if kind=='C143' else d['initial_a'])[0];z.data['PREFLIGHT_PROGRESS.json'].append(dict(**sp,kind=kind,metrics=metric(point,sp['case'],kind!='prepared')))
    for rnd in (0,1):
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid]
            for arm in old.EVAL_ARMS:
                e=np.zeros(rank) if arm==old.EVAL_ARMS[0] else d['initial_a'] if arm==old.EVAL_ARMS[1] else old.effective(heads[arm],d['weights'][sid],arm);point=p['raw'] if arm==old.EVAL_ARMS[0] else old.proposal(p['prepared'],p['raw'],d['V'],e)[0];frozen[sid+'__'+arm]=point;m=metric(point,sp['case'],True);records.append(dict(**sp,arm=arm,round=rnd,metrics=m,gates=endpoint_gates(m,sp['case'],point,p['prepared'],True),training_objective_parts=old.loss(m['observed'],m['KL'],point,p['prepared'],e,sp['case'])))
    sums,pairs=old.final_tables(records);z.data.update({'ALL672_FINAL_TRAINING_ENDPOINTS.json':records,'FINAL_ENDPOINT_PROGRESS.json':records,'ALL336_FROZEN_FINAL_POINTS.npz':frozen,'FINAL_TRAINING_SUMMARIES.json':sums,'PAIRED_ALL_ARMS.json':pairs,'SUMMARY.json':dict(decision='C148_FIXED_FINAL_COMPARISON_COMPLETE_NO_CANDIDATE_SELECTED',final_training_summaries=sums),'INPUTS_FROZEN.json':{k:c['pins'][k] for k in old.INPUT_KEYS}});a['C149'].data.update({'SUMMARY.json':dict(complete=True,errors=[],decision='C149_READONLY_ROUTING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED',parent_decision=z.data['SUMMARY.json']['decision'],final_training_summaries=sums),'INPUTS_FROZEN.json':{k:c['pins'][k] for k in auditcore.INPUT_KEYS},'ALL84_ROUTING_CONTROL_AND_ENDPOINT_DIAGNOSTICS.json':registry()});return a,c

class BindingChecks(unittest.TestCase):
    def run_fixture(self,mutation=None):
        a,c=fixture()
        if mutation:mutation(a,c)
        with tempfile.TemporaryDirectory() as out:return read_bound_inputs(a,c,Path(out))
    def test_all84_origin_rows_parent_points_and_gates_bind(self):
        d,r=self.run_fixture();self.assertEqual(len(d['points']),84);self.assertTrue(r['all336_parent_points_and672_native_gates_recomputed']);self.assertFalse(r['prior_C149_audit_reexecuted'])
    def test_origin_coordinate_change_blocks(self):
        def change(a,c):a['C143'].data['ALL84_TRAINING_INPUTS.npz']=copy.deepcopy(a['C143'].data['ALL84_TRAINING_INPUTS.npz']);a['C143'].data['ALL84_TRAINING_INPUTS.npz'][registry()[0]['state_id']+'__prepared'][50]+=.001
        with self.assertRaises(ValueError):self.run_fixture(change)
    def test_registry_permutation_blocks(self):
        with self.assertRaises(ValueError):self.run_fixture(lambda a,c:a['C146'].data['ALL84_TRAINING_REGISTRY_FROZEN.json'].reverse())
    def test_cached_weight_change_blocks(self):
        def change(a,c):a['C148'].data['ALL84_PREWRITE_ROUTING.npz']['weights'][0]=[1.,0.,0.]
        with self.assertRaises(ValueError):self.run_fixture(change)
    def test_forged_parent_gate_blocks(self):
        def change(a,c):a['C148'].data['ALL672_FINAL_TRAINING_ENDPOINTS.json'][0]['gates']['complete_pass']=False
        with self.assertRaises(ValueError):self.run_fixture(change)
    def test_final_head_is_actual_step12_not_changed_recovery(self):
        def change(a,c):a['C148'].data['FINAL_context_routed_STEP12.npz']['parameters']=d=np.zeros_like(a['C148'].data['FINAL_context_routed_STEP12.npz']['parameters'])
        with self.assertRaises(ValueError):self.run_fixture(change)
    def test_c149_link_change_blocks(self):
        def change(a,c):a['C149'].data['INPUTS_FROZEN.json']=copy.deepcopy(a['C149'].data['INPUTS_FROZEN.json']);a['C149'].data['INPUTS_FROZEN.json']['C148']['sha256']='changed'
        with self.assertRaises(ValueError):self.run_fixture(change)
    def test_wrong_production_pin_rejected_before_archive_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths,c=dummy_inputs(Path(tmp)/'inputs')
            with self.assertRaises(ValueError):load(paths,Path(tmp)/'out')
if __name__=='__main__':unittest.main(verbosity=2)
