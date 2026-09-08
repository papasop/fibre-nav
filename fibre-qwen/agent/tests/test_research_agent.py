import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_agent import ResearchStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'state.db'
        self.store = ResearchStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_revisions_persist_and_do_not_resurface_stale_claims(self):
        self.store.remember('p', 'audit', 'old PASS', 'run-1', 'reported')
        self.store.remember('p', 'audit', 'new FAIL', 'run-2', 'failed')
        self.store.close()
        self.store = ResearchStore(self.path)
        self.assertEqual(self.store.recall('p', 'PASS'), [])
        self.assertEqual(self.store.recall('p')[0]['source'], 'run-2')
        self.assertEqual(len(self.store.history('p', 'audit')), 2)

    def test_project_isolation_and_evidence_required_for_completion(self):
        self.store.remember('a', '私有记录', '待验证', 'local-note', 'hypothesis')
        first = self.store.add_task('a', '审核', '读取结果')
        second = self.store.add_task('a', '复现', '冻结配置')
        self.assertEqual(self.store.recall('b'), [])
        with self.assertRaises(ValueError): self.store.complete('b', first, 'result')
        with self.assertRaises(ValueError): self.store.complete('a', first, '')
        self.assertEqual(self.store.resume('a')['next_task']['id'], first)
        self.store.complete('a', first, 'results/run.json')
        self.assertEqual(self.store.resume('a')['next_task']['id'], second)
        with self.assertRaises(ValueError): self.store.complete('a', first, 'again')

    def test_literal_queries_and_empty_state(self):
        self.assertIsNone(self.store.resume('new')['next_task'])
        self.store.remember('p', '中文', '预算 100%', 'note', 'reported')
        self.assertEqual(len(self.store.recall('p', '中文')), 1)
        self.assertEqual(self.store.recall('p', "' OR 1=1 --"), [])
        with self.assertRaises(ValueError):
            self.store.remember('p', 'x', 'y', ' ', 'reported')


if __name__ == '__main__':
    unittest.main()
