import contextlib,io,json,tempfile,unittest,zipfile
from unittest.mock import patch
from core133 import *
from worker133 import run
from fixture133 import make_fixture,pack

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.path,cls.c=make_fixture(cls.root/'fixture');cls.serial=0
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def fresh(self):
        type(self).serial+=1;p=self.root/('case'+str(self.serial));p.mkdir();return p
    def execute(self,path=None,c=None):
        out=self.fresh()/'out'
        with contextlib.redirect_stdout(io.StringIO()):s=run(path or self.path,out,c or self.c)
        return s,out
    def changed(self,change,rebind=True):
        folder=self.fresh();root=folder/'parent';root.mkdir()
        with zipfile.ZipFile(self.path) as z:z.extractall(root)
        change(root)
        if rebind:
            seals=clean_json((root/'ALL_ARTIFACT_SEALS_BEFORE_ENDPOINTS.json').read_bytes());dump(root/'ALL_ARTIFACT_SEALS_BEFORE_ENDPOINTS.json',{n:sha(root/n) for n in seals})
        p=pack(root,folder/'changed.zip');return p,dict(self.c,pin=dict(path=str(p),sha256=sha(p)))
    def blocked(self,p,c,fragment):
        s,_=self.execute(p,c);self.assertFalse(s['complete']);self.assertIn(fragment,' '.join(s['errors']));self.assertFalse(s['LC2_closed'])
    def test_full_archive_decomposition_without_model_solve_or_SVD(self):
        with patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')),patch('numpy.linalg.svd',side_effect=AssertionError('SVD forbidden')):
            s,out=self.execute()
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['local_decomposition_records_completed'],36);self.assertEqual(s['arm_vs_raw_contrasts_completed'],24);self.assertEqual(s['next_diagnostic_branch'],'FIRST_ORDER_ENDPOINT_WITHIN_BUDGET_FINITE_ENDPOINT_FAILURE');self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['physical_curvature_cause_established']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);self.assertNotIn('torch',sys.modules)
        q=s['local_decomposition_summaries']['0'];self.assertEqual(q[ARMS[0]]['finite_response_within_budget'],6);self.assertEqual(q[ARMS[1]]['finite_response_within_budget'],0);self.assertEqual(q[ARMS[1]]['first_order_response_within_budget'],6)
        with np.load(out/'FINITE_STEP_COMPONENTS.npz',allow_pickle=False) as f:self.assertGreater(f['record00__coordinate_cancellation_amount'][0],.09)
        missing=clean_json((out/'SIX_PROBE_LOCAL_DERIVATIVES_NOT_AVAILABLE.json').read_bytes());self.assertEqual(len(missing),6);self.assertEqual(s['historical_probe_local_decompositions_claimed'],0)
    def test_production_whole_hash_rejects_fixture_before_payload(self):
        self.blocked(self.path,contract(),'Wrong fixed original C132 ZIP')
    def test_source_pin_tamper_blocks_even_rebound(self):
        p,c=self.changed(lambda r:(r/'source/NEVER_EXECUTE.py').write_text('raise AssertionError("changed")'));self.blocked(p,c,'delivered source changed')
    def test_manifest_tamper_blocks(self):
        p,c=self.changed(lambda r:None)
        with zipfile.ZipFile(p,'a') as z:z.writestr('unlisted.json','{}')
        c['pin']['sha256']=sha(p);self.blocked(p,c,'manifest inventory')
    def test_unsafe_path_blocks(self):
        p,c=self.changed(lambda r:None)
        with zipfile.ZipFile(p,'a') as z:z.writestr('../unsafe','x')
        c['pin']['sha256']=sha(p);self.blocked(p,c,'Unsafe ZIP path')
    def test_pre_endpoint_seal_tamper_blocks_even_manifest_rebound(self):
        p,c=self.changed(lambda r:dump(r/'INPUTS_FROZEN.json',{'changed':True}),False);self.blocked(p,c,'pre-endpoint seal changed')
    def test_prepared_coordinate_digest_change_blocks(self):
        def change(r):
            p=r/'FIXED12_ARCHIVED_INPUTS.npz'
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            a[registry()[0]['state_id']+'__prepared'][20]=1;np.savez_compressed(p,**a)
        p,c=self.changed(change);self.blocked(p,c,'prepared binding')
    def test_jacobian_state_binding_blocks(self):
        def change(r):
            p=r/'offline_geometry/jacobian_1.npz'
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            a['prepared'][20]=1;np.savez_compressed(p,**a)
        p,c=self.changed(change);self.blocked(p,c,'Jacobian prepared-coordinate binding')
    def test_wrong_local_jacobian_shape_blocks(self):
        def change(r):
            p=r/'offline_geometry/jacobian_1.npz'
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            a['J']=a['J'][:9];np.savez_compressed(p,**a)
        p,c=self.changed(change);self.blocked(p,c,'shape/dtype')
    def test_relabel_probe_as_geometry_blocks(self):
        def change(r):
            p=r/'FIXED12_BINDINGS.json';a=clean_json(p.read_bytes());a[1]['role']='geometry_training';dump(p,a)
        p,c=self.changed(change);self.blocked(p,c,'Fixed state identity/role')
    def test_threshold_change_blocks(self):
        def change(r):
            p=r/'DESIGN_FROZEN_BEFORE_MODEL.json';a=clean_json(p.read_bytes());a['response_budget']=.003;dump(p,a)
        p,c=self.changed(change);self.blocked(p,c,'scalar differs')
    def test_actual_point_change_blocks_even_with_rebound_seal(self):
        def change(r):
            p=r/'ALL_AVAILABLE_CANDIDATE_POINTS.npz'
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            a[registry()[0]['state_id']+'__'+ARMS[1]][20]=.2;np.savez_compressed(p,**a)
        p,c=self.changed(change);self.blocked(p,c,'Actual float32 candidate differs')
    def test_missing_endpoint_record_blocks(self):
        def change(r):
            p=r/'ALL72_REQUESTED_ENDPOINT_RECORDS.json';a=clean_json(p.read_bytes());dump(p,a[:-1])
        p,c=self.changed(change);self.blocked(p,c,'All72 endpoint')
    def test_gate_forgery_is_recomputed(self):
        def change(r):
            p=r/'ALL72_REQUESTED_ENDPOINT_RECORDS.json';a=clean_json(p.read_bytes());a[1]['metrics']['gates']['response']=True
            for name in ('ALL72_REQUESTED_ENDPOINT_RECORDS.json','ENDPOINT_PROGRESS.json'):dump(r/name,a)
            rec=a[1];dump(r/'endpoints'/rec['state_id']/f"{rec['arm']}_round0.json",rec)
        p,c=self.changed(change);self.blocked(p,c,'boolean differs')
    def test_response_scalar_forgery_is_recomputed(self):
        def change(r):
            p=r/'ALL72_REQUESTED_ENDPOINT_RECORDS.json';a=clean_json(p.read_bytes());a[1]['metrics']['response_drift']=0.
            for name in ('ALL72_REQUESTED_ENDPOINT_RECORDS.json','ENDPOINT_PROGRESS.json'):dump(r/name,a)
            rec=a[1];dump(r/'endpoints'/rec['state_id']/f"{rec['arm']}_round0.json",rec)
        p,c=self.changed(change);self.blocked(p,c,'Response scalar differs')
    def test_summary_forgery_is_recomputed(self):
        def change(r):
            p=r/'ENDPOINT_SUMMARIES.json';a=clean_json(p.read_bytes());a['all12']['0'][ARMS[1]]['passed']=12;dump(p,a)
        p,c=self.changed(change);self.blocked(p,c,'integer differs')
    def test_nonfinite_array_blocks(self):
        def change(r):
            p=r/'FROZEN_SHARED_OPERATORS.npz'
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            a['B'][0,0]=np.nan;np.savez_compressed(p,**a)
        p,c=self.changed(change);self.blocked(p,c,'nonfinite tensor')
    def test_unavailable_arm_retains24_requested_endpoints_and12_local_rows(self):
        folder=self.fresh();p,c=make_fixture(folder/'unavailable',True);s,out=self.execute(p,c);self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['evaluated_local_decompositions'],24);self.assertEqual(s['unavailable_local_decompositions'],12);self.assertEqual(s['arm_vs_raw_contrasts_completed'],24);self.assertEqual(s['parent_endpoint_summaries']['all12']['0'][ARMS[2]]['n_unavailable'],12)
    def test_known_sign_units_and_cancellation_decomposition(self):
        j=np.zeros((10,DIM));j[:,:10]=np.eye(10);theta=np.zeros(DIM,np.float32);point=theta.copy();point[0]=-.05
        before=np.zeros(10);after=np.zeros(10);after[0]=.001;d=point.astype(float)
        values,stats=decompose(j,theta,point,before,after,d,d)
        self.assertAlmostEqual(values['linear_actual'][0],-.05,7);self.assertAlmostEqual(values['finite_step_remainder'][0],.051,7);self.assertAlmostEqual(values['coordinate_cancellation_amount'][0],.1,7);self.assertEqual(stats['response_opposed_coordinates'],1)
    def test_rounding_and_cap_linear_actions_are_separate(self):
        j=np.zeros((10,DIM));j[:,:10]=np.eye(10);theta=np.zeros(DIM,np.float32);pre=np.zeros(DIM);pre[0]=.123456789123;post=pre/2;point=post.astype(np.float32);after=np.zeros(10);after[0]=.2
        values,_=decompose(j,theta,point,np.zeros(10),after,pre,post);self.assertGreater(abs(values['linear_float32_commit_effect'][0]),0);self.assertAlmostEqual(values['linear_cap_effect'][0],-pre[0]/2);self.assertAlmostEqual(values['linear_actual'][0],float(point[0]))
    def test_radius_cap_reconstruction_and_commit_identity(self):
        theta=np.zeros(DIM,np.float32);raw=theta.copy();raw[9]=40;op=dict(V=np.zeros((0,DIM)),A=np.zeros((4,DIM)),B=np.zeros((DIM,4)),response_stack=np.zeros((36,DIM)))
        p,u,post,diag=reconstruct(theta,raw,ARMS[1],op,True);self.assertTrue(diag['radius_active']);self.assertEqual(u[9],40);self.assertEqual(post[9],20);self.assertEqual(p[9],20)
    def test_arm_contrast_uses_same_state_same_prewrite(self):
        a=dict(endpoint=np.ones(10),linear_actual=np.ones(10)*2,finite_step_remainder=-np.ones(10));b=dict(endpoint=np.ones(10)*3,linear_actual=np.ones(10)*4,finite_step_remainder=-np.ones(10));values,stats=contrast(a,b);self.assertEqual(values['endpoint_difference'][0],2);self.assertEqual(values['remainder_difference'][0],0);self.assertEqual(stats['arithmetic_closure_Linf'],0)
    def test_diagnostic_branch_not_a_writer_or_curvature_claim(self):
        def row(first,finite):return dict(arm=ARMS[1],status='DECOMPOSED',statistics=dict(first_order_response_within_budget=first,finite_response_within_budget=finite))
        self.assertEqual(choose_branch([row(True,False)]),'FIRST_ORDER_ENDPOINT_WITHIN_BUDGET_FINITE_ENDPOINT_FAILURE');self.assertEqual(choose_branch([row(False,False)]),'LINEAR_ACTION_OR_BASE_RESIDUAL_ALREADY_OUTSIDE_BUDGET');self.assertEqual(choose_branch([row(True,False),row(False,False)]),'MIXED_LINEAR_AND_FINITE_STEP_FAILURE');self.assertEqual(choose_branch([row(True,True)]),'NO_PROJECTED_GEOMETRY_FAILURE_RECORDED')

if __name__=='__main__':unittest.main()
