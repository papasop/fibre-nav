import json
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import CASES, evaluate, score


class FixtureModel:
    def reply(self, messages):
        if len(messages) == 2:
            q = messages[-1]['content']
            tool = 'read_file' if 'result.json' in q else 'recall' if 'Recall' in q else 'resume'
            args = {'path': 'result.json'} if tool == 'read_file' else {}
            return json.dumps({'tool': tool, 'arguments': args})
        value = json.loads(messages[-1]['content'].split('\n', 1)[1])
        if isinstance(value, list): answer = value[0]['value'] if value else 'UNKNOWN'
        elif 'text' in value: answer = 'REPORT_FAIL' if not json.loads(value['text'])['passed'] else 'REPORT_PASS'
        else: answer = value['next_task']['next_action'] if value['next_task'] else 'UNKNOWN'
        return json.dumps({'answer': answer})


class EvaluationTests(unittest.TestCase):
    def test_paired_fixture_is_plumbing_not_model_evidence(self):
        report = evaluate(FixtureModel())
        self.assertEqual(len(report['records']), 6)
        self.assertEqual(report['summary']['with_memory']['passed'], 3)
        self.assertEqual(report['summary']['without_memory']['passed'], 1)
        self.assertEqual(len(report['protocol_sha256']), 64)

    def test_answer_without_evidence_does_not_pass(self):
        self.assertFalse(score({'status': 'answered', 'answer': 'REPORT_FAIL', 'trace': []}, CASES[0])['pass'])

    def test_backend_errors_remain_in_denominator(self):
        class Broken:
            def reply(self, messages): raise RuntimeError('offline')
        report = evaluate(Broken())
        self.assertEqual(len(report['records']), 6)
        self.assertTrue(all(r['result']['status'] == 'error' for r in report['records']))
        self.assertEqual(report['summary']['with_memory'], {'passed': 0, 'total': 3})
