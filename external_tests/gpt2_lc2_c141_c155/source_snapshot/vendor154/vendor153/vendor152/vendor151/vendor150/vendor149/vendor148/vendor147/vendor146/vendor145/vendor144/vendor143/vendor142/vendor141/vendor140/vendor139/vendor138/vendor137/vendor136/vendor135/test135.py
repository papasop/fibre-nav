import contextlib,io,tempfile,unittest
from unittest.mock import patch
from core135 import *
from worker135 import run,historical_rule
from fixture135 import make_fixture,Toy

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c=make_fixture(cls.root/'fixture');cls.serial=0
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def fresh(self):type(self).serial+=1;p=self.root/('case'+str(self.serial));p.mkdir();return p
    def execute(self,factory=Toy,c=None):
        out=self.fresh()/'out'
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,c or self.c,factory)
        return s,out
    def data(self):
        v=np.zeros((1,DIM));v[0,0]=1;theta=np.zeros(DIM,np.float32);raw=theta.copy();raw[0]=.1;raw[9]=.7;projected=raw.copy();projected[0]=0
        return dict(V=v,points={r['state_id']:dict(prepared=theta.copy(),raw=raw.copy()) for r in registry()},maps={(r['state_id'],a):dict(point=raw.copy() if a==ARMS[0] else projected.copy()) for r in registry() for a in ARMS[:2]})
    def test_full_three_archive_forward_contract_keeps168_and_control(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['endpoint_measurement_records'],168);self.assertEqual(s['new_model_forwards'],409);self.assertEqual(s['runtime_calls']['preflight_metrics'],36);self.assertTrue(s['interpretation_qualified']);self.assertTrue(s['all_coefficient_points_frozen_before_model']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['coefficient_selected']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['real_model_execution_claim']);self.assertNotIn('torch',sys.modules)
        self.assertNotIn(1.,s['noncontrol_coefficients_meeting_historical_rule']);self.assertIn(33/32,s['noncontrol_coefficients_meeting_historical_rule'])
        records=clean_json((out/'ALL168_COEFFICIENT_MEASUREMENTS.json').read_bytes());self.assertEqual(len(records),168);self.assertEqual(sum(r['local_statistics']['local_derivative_available'] for r in records),84);self.assertTrue(all(not r['local_statistics']['finite_remainder_claimed'] for r in records if r['role']=='historical_seen_center_probe'))
    def test_fixed_seven_coefficients_and_all12_roles(self):
        self.assertEqual(BETAS,(0.,.875,.9375,.96875,1.,1.03125,1.0625));self.assertEqual(len(registry()),12);self.assertEqual(sum(r['role']=='geometry_training' for r in registry()),6);self.assertTrue(all('xor_10' not in r['state_id'] for r in registry()))
    def test_raw_control_keeps_exact_bytes(self):
        d=self.data();p=next(iter(d['points'].values()));q,parts,diag=proposal(p['prepared'],p['raw'],d['V'],1.);exact(q,p['raw'],'raw');self.assertTrue(diag['raw_control_identity']);self.assertEqual(diag['clip_factor'],1.)
    def test_zero_coefficient_removes_only_declared_row_space_component(self):
        d=self.data();p=next(iter(d['points'].values()));q,parts,_=proposal(p['prepared'],p['raw'],d['V'],0.);self.assertEqual(q[0],0);self.assertEqual(q[9],p['raw'][9]);self.assertEqual(parts['normal'][9],0)
    def test_normal_multiplier_leaves_preclip_tangent_unchanged(self):
        d=self.data();p=next(iter(d['points'].values()));_,parts,_=proposal(p['prepared'],p['raw'],d['V'],15/16);self.assertEqual(parts['ideal'][9],parts['tangent'][9]);self.assertAlmostEqual(parts['ideal'][0],15/16*float(p['raw'][0]))
    def test_above_one_amplifies_normal_rather_than_whole_update(self):
        d=self.data();p=next(iter(d['points'].values()));q,parts,_=proposal(p['prepared'],p['raw'],d['V'],17/16);self.assertGreater(q[0],p['raw'][0]);self.assertEqual(q[9],p['raw'][9]);self.assertAlmostEqual(parts['ideal'][0],17/16*float(p['raw'][0]))
    def test_radius_cap_is_reported_and_scales_both_components(self):
        d=self.data();theta=np.zeros(DIM,np.float32);raw=theta.copy();raw[0]=19;raw[9]=19;q,parts,diag=proposal(theta,raw,d['V'],17/16);self.assertTrue(diag['radius_active']);self.assertTrue(diag['postcap_is_not_pure_normal_adjustment']);self.assertLess(parts['limited'][9],parts['tangent'][9]);self.assertLessEqual(float(np.linalg.norm(q.astype(float))),20.00001)
    def test_unregistered_or_boolean_coefficient_blocks(self):
        d=self.data();p=next(iter(d['points'].values()))
        for beta in (True,1,.99,float('nan')):
            with self.assertRaises(ValueError):proposal(p['prepared'],p['raw'],d['V'],beta)
    def test_nonfinite_basis_blocks(self):
        d=self.data();p=next(iter(d['points'].values()));d['V'][0,0]=np.nan
        with self.assertRaises(ValueError):proposal(p['prepared'],p['raw'],d['V'],1.)
    def test_wrong_coordinate_shape_blocks(self):
        d=self.data();p=next(iter(d['points'].values()))
        with self.assertRaises(ValueError):proposal(p['prepared'][:-1],p['raw'],d['V'],1.)
    def test_all84_outcomes_and_control_identities_are_frozen(self):
        arrays,rows,maps,parts=freeze_coefficients(self.data());self.assertEqual(len(arrays),84);self.assertEqual(len(rows),84);self.assertEqual(sum(r['control']=='original_raw' for r in rows),12);self.assertEqual(sum(r['control']=='original_projected' for r in rows),12)
    def ready_phase(self):
        d=self.data();_,_,maps,_=freeze_coefficients(d);p=Phase(d,maps)
        for key,(h,case,write) in p.preflight.items():
            sid=key[1];point=d['points'][sid]['prepared'] if key[0]=='prepared' else d['maps'][(sid,key[2])]['point'];p.authorize(key,point,case,write);p.consume(point,case,write)
        p.begin_measurements();return d,maps,p
    def test_all36_preflight_controls_required(self):
        d=self.data();_,_,maps,_=freeze_coefficients(d);p=Phase(d,maps)
        with self.assertRaises(ValueError):p.begin_measurements()
    def test_unsealed_point_blocks(self):
        d,m,p=self.ready_phase();r=registry()[0];q=m[(r['state_id'],0)].copy();q[22]=1
        with self.assertRaises(ValueError):p.authorize((0,r['state_id'],0),q,r['case'],True)
    def test_duplicate_blocks_and_registered_repeat_is_allowed(self):
        d,m,p=self.ready_phase();r=registry()[0];sid=r['state_id'];q=m[(sid,0)];p.authorize((0,sid,0),q,r['case'],True);p.consume(q,r['case'],True)
        with self.assertRaises(ValueError):p.authorize((0,sid,0),q,r['case'],True)
        p.authorize((1,sid,0),q,r['case'],True);p.consume(q,r['case'],True)
    def test_final_write_target_cannot_be_relabelled(self):
        d,m,p=self.ready_phase();r=registry()[0]
        with self.assertRaises(ValueError):p.authorize((0,r['state_id'],0),m[(r['state_id'],0)],r['case'],False)
    def test_incomplete_measurements_cannot_finish(self):
        d,m,p=self.ready_phase()
        with self.assertRaises(ValueError):p.finish()
    def test_wrong_C134_hash_blocks_before_model(self):
        c=dict(self.c,pins={k:dict(v) for k,v in self.c['pins'].items()});c['pins']['C134']['sha256']='0'*64;s,_=self.execute(lambda *args:(_ for _ in ()).throw(AssertionError('Model before pins')),c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('Wrong fixed original C134',' '.join(s['errors']))
    def test_C134_source_identity_change_blocks_before_model(self):
        c=dict(self.c,source134=dict(self.c['source134']));c['source134'][next(iter(c['source134']))]='0'*64;s,_=self.execute(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('C134 delivered source changed',' '.join(s['errors']))
    def test_preflight_drift_blocks_before_coefficient_measurements(self):
        s,out=self.execute(lambda *args:Toy(*args,preflight_drift=.0003));self.assertFalse(s['complete']);self.assertEqual(s['runtime_calls']['path_metrics'],0);self.assertTrue((out/'ALL84_COEFFICIENT_OUTCOMES_FROZEN.json').is_file())
    def test_model_pin_change_blocks_before_preflight(self):
        s,_=self.execute(lambda *args:Toy(*args,wrong_pin=True));self.assertFalse(s['complete']);self.assertEqual(s['runtime_calls']['preflight_metrics'],0)
    def test_interrupt_retains_partial_records_and_no_acceptance(self):
        s,out=self.execute(lambda *args:Toy(*args,fail_path_attempt=5));self.assertFalse(s['complete']);self.assertEqual(s['endpoint_measurement_records'],4);self.assertEqual(s['runtime_calls']['path_metric_attempts'],5);self.assertFalse(s['LC2_closed']);self.assertEqual(len(clean_json((out/'ENDPOINT_PROGRESS.json').read_bytes())),4)
    def test_control_drift_retains168_and_blocks_historical_qualification(self):
        s,_=self.execute(lambda *args:Toy(*args,path_drift=.0003));self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['endpoint_measurement_records'],168);self.assertFalse(s['interpretation_qualified']);self.assertEqual(s['noncontrol_coefficients_meeting_historical_rule'],[])
    def test_no_new_SVD_or_solve_in_full_worker(self):
        with patch('numpy.linalg.svd',side_effect=AssertionError('SVD forbidden')),patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')):s,_=self.execute()
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_Jacobians'],0);self.assertEqual(s['new_backward_calls'],0)
    def test_raw_control_never_counts_as_new_historical_improvement(self):
        sums={'all12':{str(rnd):[dict(passed=12,n_evaluated=12) for b in BETAS] for rnd in (0,1)}};pairs={'all12':{str(rnd):[dict(rescued_ids=['x'],destroyed_ids=[]) for b in BETAS] for rnd in (0,1)}}
        self.assertNotIn(1.,historical_rule(sums,pairs,True));self.assertEqual(historical_rule(sums,pairs,False),[])
    def test_missing_local_derivative_is_not_replaced_by_shared_average(self):
        v,s=local_diagnostic(None,None,1.,None,None);self.assertIsNone(v);self.assertFalse(s['local_derivative_available']);self.assertFalse(s['finite_remainder_claimed'])
    def test_known_linear_normal_units_and_finite_remainder(self):
        j=np.zeros((10,DIM));j[0,0]=2;j[1,9]=3;d=self.data();p=next(iter(d['points'].values()));point,parts,diag=proposal(p['prepared'],p['raw'],d['V'],15/16);after=np.zeros(10);after[0]=.01;v,s=local_diagnostic(j,parts,15/16,np.zeros(10),after);self.assertAlmostEqual(v['linear_normal'][0],2*float(p['raw'][0]));self.assertAlmostEqual(v['linear_actual'][1],3*float(point[9]));self.assertAlmostEqual(v['finite_step_remainder'][0],.01-2*float(point[0]))

if __name__=='__main__':unittest.main()
