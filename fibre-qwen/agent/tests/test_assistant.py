import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistant import ChatModel, ReadTools, run
from research_agent import ResearchStore


class Scripted:
    def __init__(self, actions):
        self.actions = iter(actions)
    def reply(self, messages):
        return json.dumps(next(self.actions))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'result.json').write_text('{"passed": false}', encoding='utf-8')
        self.store = ResearchStore(self.root / 'state.db')
        self.tools = ReadTools(self.root, self.store, 'p')
    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_http_model_to_tool_to_answer(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append(request)
                if len(captured) == 1:
                    action = {'tool': 'read_file', 'arguments': {'path': 'result.json'}}
                else:
                    result = json.loads(request['messages'][-1]['content'].split('\n', 1)[1])
                    action = {'answer': 'Reported pass=' + str(json.loads(result['text'])['passed'])}
                body = json.dumps({'choices': [{'message': {'content': json.dumps(action)}}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = ChatModel(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', 'test')
            result = run('Check result', model, self.tools)
            self.assertEqual(result['answer'], 'Reported pass=False')
            self.assertEqual(len(result['trace'][0]['result']['sha256']), 64)
            self.assertEqual(len(captured), 2)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_tool_boundaries(self):
        (self.root / '.hidden.md').write_text('secret')
        (self.root / 'alias.md').symlink_to(self.root / 'result.json')
        (self.root / 'large.txt').write_text('a' * 17000)
        for path in ['../outside', '/etc/passwd', '.hidden.md', 'alias.md', 'state.db', 'large.txt']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.tools.call('read_file', {'path': path})
        with self.assertRaises(ValueError): self.tools.call('shell', {'command': 'echo bad'})
        with self.assertRaises(ValueError): self.tools.call('recall', {'query': '', 'project': 'other'})

    def test_resume_and_no_automatic_completion(self):
        task = self.store.add_task('p', 'Review', 'Check results')
        model = Scripted([{'tool': 'resume', 'arguments': {}}, {'answer': 'Review pending.'}])
        result = run('Continue', model, self.tools)
        self.assertEqual(result['trace'][0]['result']['next_task']['id'], task)
        self.assertEqual(self.store.resume('p')['next_task']['status'], 'pending')

    def test_loop_and_invalid_actions(self):
        action = {'tool': 'list_files', 'arguments': {}}
        result = run('Check', Scripted([action, action]), self.tools, max_steps=2)
        self.assertEqual(result['status'], 'step_limit')
        self.assertIsNone(result['answer'])
        with self.assertRaises(ValueError): run('Check', Scripted([{'surprise': True}]), self.tools)
        result = run('Check', Scripted([{'tool': 'delete', 'arguments': {}}, {'answer': 'Unavailable'}]), self.tools)
        self.assertIn('error', result['trace'][0]['result'])

    def test_single_json_fence_is_accepted_but_extra_text_is_not(self):
        class Fenced:
            def reply(self, messages):
                return '```json\n{"answer":"UNKNOWN"}\n```'
        self.assertEqual(run('Check', Fenced(), self.tools)['answer'], 'UNKNOWN')
        class Extra:
            def reply(self, messages):
                return 'Here is the result: ```json\n{"answer":"UNKNOWN"}\n```'
        self.assertEqual(run('Check', Extra(), self.tools, max_steps=1)['status'], 'step_limit')

    def test_invalid_json_repair_is_bounded_and_preserves_raw(self):
        class Repair:
            def __init__(self): self.calls = 0
            def reply(self, messages):
                self.calls += 1
                return 'answer: UNKNOWN' if self.calls == 1 else '{"answer":"UNKNOWN"}'
        result = run('Check', Repair(), self.tools, max_steps=2)
        self.assertEqual(result['answer'], 'UNKNOWN')
        self.assertEqual(result['trace'][0]['result']['raw_reply'], 'answer: UNKNOWN')

    def test_endpoint_validation(self):
        for endpoint in ['http://remote.example/v1', 'https://user:pass@example.com', 'https://example.com/?key=x']:
            with self.assertRaises(ValueError): ChatModel(endpoint, 'model')


if __name__ == '__main__': unittest.main()
