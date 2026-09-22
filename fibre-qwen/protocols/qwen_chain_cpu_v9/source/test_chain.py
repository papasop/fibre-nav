"""Engineering tests use archived V6 records as a deterministic mock backend, not new model evidence."""
import contextlib,copy,io,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from protocol import schedule,programs,TOTAL
from execution import execute_item,decoded_bit
from evaluation import summarize,truth
from interface_compiler import parse_input,compile_content
import run_cpu
ROOT=Path(__file__).resolve().parent
BASE=json.loads((ROOT/'baseline.json').read_text())
MAP={(q['task'],int(q['pair'][0]),int(q['pair'][1])):(q,r) for q,r in zip(BASE['queries'],BASE['records'])}

def engine(text,forced_bit=None,invalid=False):
    f=parse_input(text);q,r=MAP[(f['op'],f['a'],f['b'])]
    if forced_bit is not None:
        r=next(v[1] for v in MAP.values() if v[1]['constrained']['raw']==str(forced_bit))
    obs=dict(accepted=True,parse_error=None,parsed=f,compiled_content=q['content'],compiled_prompt=q['prompt'],model_invocations=1,
             **{k:copy.deepcopy(r[k]) for k in ('scores','constrained','numeric')})
    if invalid:obs['constrained']['raw']='invalid';obs['constrained']['output_ids']=[42,151645]
    return obs

def run_mock(evaluator=engine):
    rows=[]
    for item in schedule():rows.append(execute_item(item,rows,evaluator))
    return rows

class ChainTests(unittest.TestCase):
    def test_inventory(self):
        self.assertEqual(len(schedule()),112);self.assertEqual(len(programs()),32)
        self.assertEqual(len({(q['op1'],q['op2']) for q in programs()}),16)
        self.assertEqual(len({(q['a'],q['b'],q['c']) for q in programs()}),8)
        self.assertTrue(all('expected' not in json.dumps(q) for q in schedule()))
    def test_all_pass_is_scoped_and_no_v8_assumption(self):
        result=summarize(run_mock(),schedule(),BASE,True)
        self.assertTrue(result['overall_pass']);self.assertEqual(result['bounded_correct'],112)
        self.assertEqual(result['end_to_end_correct'],32);self.assertEqual(result['direct_replays_passed'],96)
        self.assertFalse(result['v8_final_pass_confirmed']);self.assertFalse(result['native_planning'])
        self.assertFalse(result['error_propagation_test_observed'])
    def test_no_truth_called_by_controller(self):
        with patch('evaluation.truth',side_effect=AssertionError('Oracle entered execution')):
            rows=run_mock()
        self.assertEqual(len(rows),112)
    def test_actual_wrong_intermediate_is_propagated_not_repaired(self):
        plan=schedule();rows=run_mock();idx=next(i for i,p in enumerate(plan) if p['kind']=='step1' and p['program']['op2']=='XOR')
        rows=rows[:idx];q=plan[idx]['program'];wrong=1-truth(q['op1'],q['a'],q['b'])
        rows.append(execute_item(plan[idx],rows,lambda text:engine(text,forced_bit=wrong)))
        rows.append(execute_item(plan[idx+1],rows,engine));rows.append(execute_item(plan[idx+2],rows,engine))
        self.assertEqual(rows[-2]['fields']['a'],wrong);self.assertEqual(rows[-1]['fields']['a'],1-wrong)
        case=summarize(rows,plan,BASE,True)['cases'][-1]
        self.assertTrue(case['upstream_error']);self.assertTrue(case['follows_actual_intermediate']);self.assertTrue(case['error_propagated'])
    def test_invalid_intermediate_skips_dependents_and_continues(self):
        plan=schedule();rows=run_mock()[:16]
        rows.append(execute_item(plan[16],rows,lambda text:engine(text,invalid=True)))
        calls=[]
        def counted(text):calls.append(text);return engine(text)
        rows.append(execute_item(plan[17],rows,counted));rows.append(execute_item(plan[18],rows,counted))
        self.assertFalse(calls);self.assertEqual(rows[-1]['observation']['model_invocations'],0)
        self.assertFalse(summarize(rows,plan,BASE,True)['overall_pass'])
        rows.append(execute_item(plan[19],rows,counted));self.assertEqual(len(calls),1)
    def test_wiring_tamper_fails_even_with_correct_answer(self):
        rows=run_mock();rows[17]['fields']['a']=1-rows[17]['fields']['a']
        with self.assertRaises(ValueError):summarize(rows,schedule(),BASE,True)
    def test_numeric_mask_environment_and_incomplete_fail(self):
        rows=run_mock()
        self.assertFalse(summarize(rows[:-1],schedule(),BASE,True)['overall_pass'])
        self.assertFalse(summarize(rows,schedule(),BASE,False)['overall_pass'])
        for mutate in ('mask','aligned'):
            rs=copy.deepcopy(rows)
            if mutate=='mask':rs[0]['observation']['constrained']['trace']['masked_bit_logits'][0]+=1
            else:rs[0]['observation']['numeric']['last_no_cache']['bit_logits'][0]+=1
            self.assertFalse(summarize(rs,schedule(),BASE,True)['overall_pass'])
    def test_resume_between_steps_uses_committed_output(self):
        plan=schedule();calls=[]
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'run';out.mkdir();rows=run_mock()[:17]
            for i,r in enumerate(rows):run_cpu.write(out/'records'/f'{i:03d}.json',r)
            original=(out/'records/016.json').read_bytes()
            def evaluator(text):calls.append(text);return engine(text)
            result=run_cpu.execute(out,plan,BASE,evaluator,True)
            self.assertTrue(result['overall_pass']);self.assertEqual(len(calls),95)
            self.assertEqual((out/'records/016.json').read_bytes(),original)
            self.assertEqual(parse_input(calls[0])['a'],decoded_bit(rows[-1]['observation']))
            run_cpu.execute(out,plan,BASE,evaluator,True);self.assertEqual(len(calls),95)
            self.assertTrue((out.parent/(out.name+'_results.zip')).exists())
    def test_expired_budget_makes_no_new_call(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(run_cpu.Pause):
                run_cpu.execute(Path(d),schedule(),BASE,lambda _:self.fail('Inference after deadline'),True,time.monotonic()-1)
    def test_record_gap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);run_cpu.write(out/'records/001.json',run_mock()[1])
            with self.assertRaises(ValueError):run_cpu.saved_records(out,schedule())

if __name__=='__main__':unittest.main(verbosity=2)
