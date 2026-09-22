"""Synthetic engineering checks; never pretrained-model evidence."""
import copy,contextlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import protocol as P
import run_cpu as R
import interface_compiler as C
from numerical_diagnostic import SETTINGS
ROOT=Path(__file__).resolve().parent;BASE=json.loads((ROOT/'baseline.json').read_text())
TABLE={'OR':'0111','AND':'0001','XOR':'0110','EQUAL':'1001'}
class Tok:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=False,add_generation_prompt=True,enable_thinking=False)
        q=BASE['queries'][0];a,b=q['prompt'].split(q['content']);return a+messages[0]['content']+b[:-len('Answer: ')]
def plan():return P.schedule(P.make_queries(Tok(),BASE))
def good(item):
    q=item['query'];bit=TABLE[q['task']][P.PAIRS.index(q['pair'])];lp=[-.1,-3.] if bit=='0' else [-3.,-.1]
    return dict(**copy.deepcopy(item),accepted=True,parse_error=None,parsed=copy.deepcopy(q['expected_fields']),compiled_content=q['compiled_content'],compiled_prompt=q['prompt'],model_invocations=1,
        scores=dict(bit_logits=lp[:],bit_logp=lp[:],eos_given_bit_logp=[-.2,-.2]),
        constrained=dict(raw=bit,output_ids=[P.BIT_IDS[int(bit)],P.EOS_ID],truncated=False,effective_config={'do_sample':False},policy=copy.deepcopy(P.POLICY),
            trace=dict(steps=[0,1],first_bit_logits=lp[:],masked_bit_logits=lp[:],finite_bit_count=2)),
        numeric={k:dict(settings=v.copy(),bit_logits=lp[:]) for k,v in SETTINGS.items()})
def fixture():
    p=plan();r=[good(i) for i in p];b=copy.deepcopy(BASE);b['records']=copy.deepcopy(r[:16]);return p,r,b
def engine_from(rows):
    lookup={r['compiled_prompt']:r for r in rows}
    def infer(prompt):return copy.deepcopy({k:lookup[prompt][k] for k in ('scores','constrained','numeric')})
    return infer
class ProtocolTests(unittest.TestCase):
    def test_frozen_inventory_canonical_equality_and_no_answer_lookup(self):
        qs=P.make_queries(Tok(),BASE);self.assertEqual(qs,json.loads((ROOT/'FROZEN_QUERIES.json').read_text()))
        self.assertEqual((len(qs),len(plan()),len({q['prompt'] for q in qs})),(64,128,16))
        with patch.object(P,'truth',side_effect=AssertionError('oracle entered inference')):
            for q in qs:
                self.assertEqual(C.parse_input(q['source_text']),q['expected_fields'])
                content,prompt=C.compile_prompt(Tok(),C.parse_input(q['source_text']))
                self.assertEqual((content,prompt),(q['compiled_content'],q['prompt']))
    def test_three_text_forms_with_and_without_rule(self):
        for q in P.make_queries(Tok(),BASE)[:48]:
            self.assertEqual(C.parse_input(q['source_text'].split('\n',1)[1]),q['expected_fields'])
            self.assertEqual(C.parse_input('\n'+q['source_text']+'\n'),q['expected_fields'])
    def test_negative_cases_never_reach_engine(self):
        self.assertEqual((P.parser_audit()['n'],P.parser_audit()['rejected']),(24,24))
        for c in json.loads((ROOT/'FROZEN_NEGATIVES.json').read_text()):
            r=C.process_input(c['text'],None,lambda _:self.fail('invalid input reached model'))
            self.assertFalse(r['accepted']);self.assertEqual(r['model_invocations'],0)
    def test_structured_validation_refuses_extra_or_noninteger_fields(self):
        for f in (dict(op='OR',a=True,b=0),dict(op='OR',a=0.0,b=0),dict(op='OR',a=0,b=0,answer=0),dict(op='OR',a=0),dict(op='NAND',a=0,b=0)):
            with self.assertRaises(C.ParseError):C.compile_content(f)
        for text in (None,'x'*4097,'{"op":'+ '['*1100+'0'+']'*1100+'}'):
            with self.assertRaises(C.ParseError):C.parse_input(text)
    def test_correct_recording_and_backend_gets_only_prompt(self):
        q=P.make_queries(Tok(),BASE)[17];seen=[]
        with patch.object(P,'truth',side_effect=AssertionError('no oracle')):
            r=C.process_input(q['source_text'],Tok(),lambda text:(seen.append(text) or {'sentinel':42}))
        self.assertEqual(seen,[q['prompt']]);self.assertEqual(r['parsed'],q['expected_fields']);self.assertEqual(r['sentinel'],42)
    def test_complete_success_scoped_to_external_compiler(self):
        p,r,b=fixture();s=P.summarize(r,p,b);self.assertTrue(s['overall_pass']);self.assertTrue(s['compiler_pass'])
        self.assertIsNone(s['native_free_pass']);self.assertFalse(s['free_generation_evaluated']);self.assertFalse(s['unseen_expression_confirmation']);self.assertFalse(s['formal_l6_pass']);self.assertFalse(s['deployment_ready']);self.assertFalse(s['v7_raw_archive_audited'])
    def test_actual_baseline_and_failure_history_preserved(self):
        g=P.group(BASE['records']);self.assertEqual(g['primary_correct'],16)
        self.assertEqual(sum(r['free']['raw'].strip() in ('0','1') for r in BASE['records']),11)
        self.assertFalse(json.loads((ROOT/'V6_PROVENANCE.json').read_text())['parent_overall_pass'])
        self.assertFalse(json.loads((ROOT/'V7_LOG_PROVENANCE.json').read_text())['v7_result_zip_audited'])
    def test_parse_or_compile_mismatch_cannot_be_hidden_by_correct_answer(self):
        for field in ('parsed','compiled_content','compiled_prompt','model_invocations'):
            p,r,b=fixture()
            r[20][field]=dict(op='OR',a=1,b=0) if field=='parsed' else 0 if field=='model_invocations' else 'wrong'
            self.assertFalse(P.summarize(r,p,b)['overall_pass'])
    def test_wrong_answer_not_repaired_by_grammar(self):
        p,r,b=fixture()
        for x in r:
            if x['query']['pair']=='00' and x['query']['task']=='EQUAL':
                x['scores'].update(bit_logp=[-.1,-3.],bit_logits=[-.1,-3.]);c=x['constrained'];c.update(raw='0',output_ids=[15,P.EOS_ID]);c['trace'].update(first_bit_logits=[-.1,-3.],masked_bit_logits=[-.1,-3.])
                for v in x['numeric'].values():v['bit_logits']=[-.1,-3.]
        b['records']=copy.deepcopy(r[:16]);s=P.summarize(r,p,b)
        self.assertTrue(s['integrity_pass']);self.assertFalse(s['overall_pass']);self.assertEqual(s['decision'],'CANONICAL_MODEL_COMPUTATION_FAILURE')
    def test_rejected_valid_input_records_failure_without_inference(self):
        p,r,b=fixture();r[16]=dict(**p[16],**C.process_input('bad',None,lambda _:self.fail('called')))
        s=P.summarize(r,p,b);self.assertFalse(s['overall_pass']);self.assertEqual(s['positive_compiler_matches'],127)
        r[16]['model_invocations']=1
        with self.assertRaisesRegex(ValueError,'REJECTED_INPUT_REACHED_MODEL'):P.summarize(r,p,b)
    def test_new_aligned_gate_and_legacy_reporting(self):
        p,r,b=fixture()
        # Identical full-head gap perturbation everywhere: old metric fails, aligned logits stay intact.
        for row in r:
            row['scores']['bit_logp'][0]+=1.2e-5;row['scores']['bit_logits'][0]+=1.2e-5
        b['records']=copy.deepcopy(r[:16]);s=P.summarize(r,p,b)
        self.assertTrue(s['overall_pass']);self.assertEqual(s['numerical_diagnostic']['legacy_cross_path_failures'],128)
        r[16]['numeric']['last_cache']['bit_logits']=[x+.001 for x in r[16]['numeric']['last_cache']['bit_logits']]
        self.assertFalse(P.summarize(r,p,b)['overall_pass'])
    def test_mask_policy_tokens_and_missing_numeric_fail(self):
        for kind in ('mask','tokens','selection'):
            p,r,b=fixture();c=r[16]['constrained']
            if kind=='mask':c['trace']['masked_bit_logits'][0]+=.001
            elif kind=='tokens':c['output_ids']=[15,42,P.EOS_ID]
            else:c['raw']='1';c['output_ids']=[16,P.EOS_ID]
            self.assertFalse(P.summarize(r,p,b)['overall_pass'])
        p,r,b=fixture();r[0]['constrained']['policy']['max_output_tokens']=3
        with self.assertRaisesRegex(ValueError,'CONSTRAINT_POLICY'):P.summarize(r,p,b)
        p,r,b=fixture();del r[0]['numeric']
        with self.assertRaises(KeyError):P.summarize(r,p,b)
    def test_partial_env_and_repeats_fail_closed(self):
        p,r,b=fixture()
        for n in (0,16,64,127):self.assertFalse(P.summarize(r[:n],p,b)['overall_pass'])
        self.assertFalse(P.summarize(r,p,b,False)['overall_pass'])
        r[-1]['constrained']['effective_config']['do_sample']=True;self.assertFalse(P.summarize(r,p,b)['overall_pass'])
    def test_invalid_scores_order_and_ties(self):
        p,r,b=fixture()
        for rows in (r+[r[0]],r[1:],r[:8]+r[9:]):
            with self.assertRaises(ValueError):P.summarize(rows,p,b)
        for bad in ([float('nan'),-1.],[0.,0.],[.1,-2.]):
            row=good(p[0]);row['scores']['bit_logp']=bad
            with self.assertRaises(ValueError):P.score(row)
        row=good(p[0]);row['scores']['bit_logp']=[-1.,-1.];self.assertIsNone(P.score(row)['primary_prediction'])
    def test_full_controller_executes_every_input_and_saves_stages(self):
        p,r,b=fixture();infer=engine_from(r[:16]);calls=[]
        def evaluator(source):calls.append(source);return C.process_input(source,Tok(),infer)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'run';out.mkdir();s=R.execute(out,p,b,evaluator,True);self.assertTrue(s['overall_pass']);self.assertEqual(calls,[x['query']['source_text'] for x in p])
            R.execute(out,p,b,lambda _:self.fail('completed'),True)
            for n in range(16,129,16):self.assertTrue((out.parent/f'run_{n:03d}_results.zip').is_file())
            with zipfile.ZipFile(out.parent/'run_results.zip') as z:
                for n,h in json.loads(z.read('run/RESULT_MANIFEST.json')).items():self.assertEqual(P.sha(z.read('run/'+n)),h)
    def test_interrupt_resume_and_expired_budget(self):
        p,r,b=fixture();infer=engine_from(r[:16]);calls=0
        def evaluator(source):
            nonlocal calls
            if calls==17:raise RuntimeError('interrupted before commit')
            calls+=1;return C.process_input(source,Tok(),infer)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'run';out.mkdir()
            with self.assertRaises(RuntimeError):R.execute(out,p,b,evaluator,True)
            self.assertEqual(len(R.saved_records(out,p)),17)
            s=R.execute(out,p,b,lambda x:C.process_input(x,Tok(),infer),True);self.assertTrue(s['overall_pass'])
            with self.assertRaises(R.Pause):R.execute(Path(d)/'empty',p,b,lambda _:self.fail('expired'),True,deadline=0)
if __name__=='__main__':unittest.main(verbosity=2)
