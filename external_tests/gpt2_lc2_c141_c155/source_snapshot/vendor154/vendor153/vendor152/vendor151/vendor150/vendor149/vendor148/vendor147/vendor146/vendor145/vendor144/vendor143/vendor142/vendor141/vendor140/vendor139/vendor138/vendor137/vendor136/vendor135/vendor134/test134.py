import contextlib,io,tempfile,unittest
from unittest.mock import patch
from core134 import *
from fixture134 import make_fixture,Toy
from worker134 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c=make_fixture(cls.root/'fixture');cls.serial=0
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def fresh(self):type(self).serial+=1;p=self.root/('case'+str(self.serial));p.mkdir();return p
    def execute(self,factory=Toy,paths=None,c=None):
        out=self.fresh()/'out'
        with contextlib.redirect_stdout(io.StringIO()):s=run(paths or self.paths,out,c or self.c,factory)
        return s,out
    def data(self,unavailable=False):
        theta=np.zeros(DIM,np.float32);theta[6:10]=[-.35,-.35,.35,-.35];end=theta.copy();end[9]=.35
        points={r['state_id']:dict(prepared=theta.copy()) for r in registry()};maps={(r['state_id'],a):dict(point=None if unavailable and a==ARMS[2] else end.copy()) for r in registry() for a in ARMS}
        return dict(points=points,maps=maps)
    def test_full_forward_contract_keeps252_and_all_failures(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['evaluated_path_measurements'],252);self.assertEqual(s['path_measurement_records'],252);self.assertEqual(s['new_model_forwards'],553);self.assertEqual(s['runtime_calls']['preflight_metrics'],24);self.assertTrue(s['path_interpretation_qualified']);self.assertTrue(s['repeatable']);self.assertTrue(s['all_path_points_frozen_before_model']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['real_model_execution_claim']);self.assertNotIn('torch',sys.modules)
        self.assertEqual(s['path_summaries']['0'][ARMS[0]][-1]['all_declared_gates_passed'],6);self.assertEqual(s['path_summaries']['0'][ARMS[1]][-1]['response_passed'],0)
        records=clean_json((out/'ALL252_PATH_MEASUREMENTS.json').read_bytes());self.assertEqual(len(records),252);self.assertTrue(any(not r['recomputed_gates']['complete_pass'] for r in records));self.assertTrue(all(r['statistics']['response_slope_disagreement_Linf'] is None for r in records if r['alpha']==0))
    def test_fixed_registry_and_positive_dyadic_grid(self):
        self.assertEqual(ALPHAS,(0.,.015625,.0625,.25,.5,.75,1.));self.assertEqual(len(registry()),6);self.assertTrue(all('xor_10' not in r['state_id'] for r in registry()))
    def test_path_endpoints_keep_exact_coordinate_bytes(self):
        theta=np.linspace(-.1,.1,DIM,dtype=np.float32);end=theta+.01;exact(path_point(theta,end,0.),theta,'zero');exact(path_point(theta,end,1.),end,'one')
    def test_unregistered_fraction_and_boolean_rejected(self):
        a=np.zeros(DIM,np.float32)
        for alpha in (True,1,.3,float('nan')):
            with self.assertRaises(ValueError):path_point(a,a,alpha)
    def test_interpolation_uses_actual_committed_endpoint(self):
        theta=np.full(DIM,.1,np.float32);end=theta.copy();end[0]=.23456789;p=path_point(theta,end,.25);expected=np.float32(float(theta[0])+.25*(float(end[0])-float(theta[0])));self.assertEqual(p[0],expected)
    def test_all126_points_frozen_without_collapsing_identical_zeros(self):
        arrays,rows,maps=freeze_paths(self.data());self.assertEqual(len(arrays),126);self.assertEqual(len(rows),126);self.assertEqual(sum(r['alpha']==0 for r in rows),18);self.assertEqual(len(maps),126)
    def test_global_unavailable_arm_retains42_point_outcomes(self):
        arrays,rows,maps=freeze_paths(self.data(True));self.assertEqual(len(arrays),84);self.assertEqual(sum(not r['available'] for r in rows),42)
    def test_phase_blocks_paths_until_all_preflight_calls_consumed(self):
        d=self.data();_,_,maps=freeze_paths(d);p=Phase(d,maps)
        with self.assertRaises(ValueError):p.begin_paths()
    def ready_phase(self):
        d=self.data();_,_,maps=freeze_paths(d);p=Phase(d,maps)
        for key,(h,case,write) in p.preflight.items():
            sid=key[1];point=d['points'][sid]['prepared'] if key[0]=='prepared' else d['maps'][(sid,key[2])]['point'];p.authorize(key,point,case,write);p.consume(point,case,write)
        p.begin_paths();return d,maps,p
    def test_unsealed_point_blocks_model_call(self):
        d,m,p=self.ready_phase();r=registry()[0];point=m[(r['state_id'],ARMS[0],0)].copy();point[20]=1
        with self.assertRaises(ValueError):p.authorize((0,r['state_id'],ARMS[0],0),point,r['case'],False)
    def test_duplicate_call_blocks_but_second_registered_round_allowed(self):
        d,m,p=self.ready_phase();r=registry()[0];sid=r['state_id'];point=m[(sid,ARMS[0],0)];key=(0,sid,ARMS[0],0);p.authorize(key,point,r['case'],False);p.consume(point,r['case'],False)
        with self.assertRaises(ValueError):p.authorize(key,point,r['case'],False)
        p.authorize((1,sid,ARMS[0],0),point,r['case'],False);p.consume(point,r['case'],False)
    def test_target_cannot_be_relabelled_for_intermediate_point(self):
        d,m,p=self.ready_phase();r=registry()[0];sid=r['state_id']
        with self.assertRaises(ValueError):p.authorize((0,sid,ARMS[0],1),m[(sid,ARMS[0],1)],r['case'],False)
    def test_unconsumed_authorization_blocks_next_request(self):
        d,m,p=self.ready_phase();r=registry()[0];sid=r['state_id'];point=m[(sid,ARMS[0],0)];p.authorize((0,sid,ARMS[0],0),point,r['case'],False)
        with self.assertRaises(ValueError):p.authorize((1,sid,ARMS[0],0),point,r['case'],False)
    def test_incomplete_paths_cannot_finish(self):
        d,m,p=self.ready_phase()
        with self.assertRaises(ValueError):p.finish()
    def test_original_pin_failure_blocks_before_model(self):
        c=dict(self.c,pins={k:dict(v) for k,v in self.c['pins'].items()});c['pins']['C133']['sha256']='0'*64
        s,_=self.execute(lambda *args:(_ for _ in ()).throw(AssertionError('Model must not load')),c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('Wrong fixed original C133',' '.join(s['errors']))
    def test_C133_source_pin_change_blocks_before_model(self):
        c=dict(self.c,source133=dict(self.c['source133']));first=next(iter(c['source133']));c['source133'][first]='0'*64;s,_=self.execute(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('C133 delivered source changed',' '.join(s['errors']))
    def test_model_pin_mismatch_blocks_before_measurements(self):
        s,_=self.execute(lambda *args:Toy(*args,wrong_pin=True));self.assertFalse(s['complete']);self.assertEqual(s['runtime_calls']['path_metrics'],0);self.assertEqual(s['runtime_calls']['preflight_metrics'],0)
    def test_preflight_drift_blocks_before_any_path_measurement(self):
        s,out=self.execute(lambda *args:Toy(*args,preflight_drift=.0003));self.assertFalse(s['complete']);self.assertEqual(s['runtime_calls']['path_metrics'],0);self.assertTrue(s['all_path_points_frozen_before_model']);self.assertTrue((out/'ALL126_PATH_OUTCOMES_FROZEN.json').is_file())
    def test_mid_path_failure_retains_partial_progress_and_no_acceptance(self):
        s,out=self.execute(lambda *args:Toy(*args,fail_path_attempt=5));self.assertFalse(s['complete']);self.assertEqual(s['path_measurement_records'],4);self.assertEqual(s['runtime_calls']['path_metric_attempts'],5);self.assertFalse(s['LC2_closed']);self.assertEqual(len(clean_json((out/'PATH_PROGRESS.json').read_bytes())),4)
    def test_path_replay_drift_keeps_full_registry_and_blocks_interpretation(self):
        s,_=self.execute(lambda *args:Toy(*args,path_drift=.0003));self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['path_measurement_records'],252);self.assertFalse(s['path_interpretation_qualified']);self.assertIn('REPLAY_DRIFT',s['decision']);self.assertFalse(s['LC2_closed'])
    def test_unavailable_arm_retains84_null_measurement_records(self):
        paths,c=make_fixture(self.fresh()/'unavailable',True);s,out=self.execute(paths=paths,c=c);self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['unavailable_path_measurements'],84);self.assertEqual(s['evaluated_path_measurements'],168);self.assertEqual(s['new_model_forwards'],373)
        rows=clean_json((out/'ALL252_PATH_MEASUREMENTS.json').read_bytes());self.assertEqual(sum(r['metrics'] is None for r in rows),84)
    def test_fixed_fraction_slope_has_correct_sign_units_and_quadratic_remainder(self):
        j=np.zeros((10,DIM));j[0,0]=2;theta=np.zeros(DIM,np.float32);end=theta.copy();end[0]=.5;alpha=.25;point=path_point(theta,end,alpha);observed=np.zeros(10);observed[0]=2*float(point[0])+3*float(point[0])**2;v,s=analyse(j,theta,end,point,alpha,np.zeros(10),observed)
        self.assertAlmostEqual(v['full_directional_action'][0],1);self.assertAlmostEqual(v['linear_actual'][0],.25);self.assertAlmostEqual(v['finite_step_remainder'][0],3*.125**2);self.assertAlmostEqual(v['remainder_div_alpha_squared'][0],.75);self.assertAlmostEqual(v['finite_fraction_slope'][0],observed[0]/alpha)
    def test_zero_fraction_has_no_divide_by_zero_or_slope(self):
        j=np.zeros((10,DIM));a=np.zeros(DIM,np.float32);v,s=analyse(j,a,a,a,0.,np.zeros(10),np.zeros(10));self.assertNotIn('finite_fraction_slope',v);self.assertIsNone(s['response_slope_disagreement_Linf'])
    def test_no_solver_or_SVD_is_used_during_worker(self):
        with patch('numpy.linalg.svd',side_effect=AssertionError('SVD forbidden')),patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')):s,_=self.execute()
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_Jacobians'],0);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_SVD_calls'],0)

if __name__=='__main__':unittest.main()
