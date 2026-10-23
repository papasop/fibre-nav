#!/usr/bin/env python3
"""A2 synthetic development benchmark; never an MFI or generalization certificate."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time

from assistant import ChatModel, ReadTools, run
from research_agent import ResearchStore

CASES = [
    {'id': 'read_result', 'question': 'Read result.json. Answer exactly REPORT_FAIL if its passed field is false, otherwise REPORT_PASS.',
     'expected': 'REPORT_FAIL', 'tool': 'read_file'},
    {'id': 'current_memory', 'question': 'Recall the current release_decision memory. Answer exactly its value, or UNKNOWN if unavailable.',
     'expected': 'HOLD_FOR_REPLICATION', 'tool': 'recall'},
    {'id': 'continue_task', 'question': 'Use resume and answer exactly the next pending task next_action, or UNKNOWN if none.',
     'expected': 'CHECK_FROZEN_RESULTS', 'tool': 'resume'},
]
PROTOCOL = {'version': 'A2-repair-2', 'cases': CASES, 'max_steps': 6,
            'arms': ['without_memory', 'with_memory'], 'result_file': {'passed': False},
            'revisions': ['READY_TO_RELEASE', 'HOLD_FOR_REPLICATION'],
            'task': 'CHECK_FROZEN_RESULTS', 'decoding': 'greedy/temperature-zero'}


class LocalQwen:
    """Optional backend; imports ML dependencies only when explicitly selected."""
    def __init__(self, model_id, revision, device='cpu'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False,
            torch_dtype=torch.float32 if device == 'cpu' else torch.float16).to(device).eval()
        self.device = device
        self.revision = getattr(self.model.config, '_commit_hash', None) or revision

    def reply(self, messages):
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to(self.device)
        if tokens.shape[1] > 12000:
            raise ValueError('Context exceeds local benchmark limit.')
        with self.torch.inference_mode():
            output = self.model.generate(tokens, max_new_tokens=512, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True).strip()


def score(result, case):
    exact = result.get('status') == 'answered' and result.get('answer', '').strip() == case['expected']
    observed = any(t['tool'] == case['tool'] and not (
        isinstance(t['result'], dict) and 'error' in t['result']) for t in result.get('trace', []))
    return {'exact': exact, 'required_tool_observed': observed, 'pass': exact and observed}


def evaluate(model):
    records = []
    # Alternate arm order by case. Same model/tools/question; only state differs.
    for index, case in enumerate(CASES):
        arms = PROTOCOL['arms'] if index % 2 == 0 else list(reversed(PROTOCOL['arms']))
        for arm in arms:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / 'evidence'; root.mkdir()
                (root / 'result.json').write_text(json.dumps(PROTOCOL['result_file']))
                store = ResearchStore(base / 'state.db')
                try:
                    if arm == 'with_memory':
                        for value in PROTOCOL['revisions']:
                            store.remember('audit', 'release_decision', value, 'synthetic:decision', 'reported')
                        store.add_task('audit', 'Review', PROTOCOL['task'])
                    started = time.monotonic()
                    try:
                        result = run(case['question'], model, ReadTools(root, store, 'audit'), PROTOCOL['max_steps'])
                    except Exception as error:
                        # Preserve a failed row and continue the paired experiment.
                        result = {'status': 'error', 'error_type': type(error).__name__, 'answer': None, 'trace': []}
                    records.append({'case': case['id'], 'arm': arm, 'seconds': time.monotonic() - started,
                                    'scores': score(result, case), 'result': result})
                finally:
                    store.close()
    return {'protocol': PROTOCOL, 'protocol_sha256': hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(), 'records': records,
        'summary': {arm: {'passed': sum(r['scores']['pass'] for r in records if r['arm'] == arm),
                          'total': len(CASES)} for arm in PROTOCOL['arms']},
        'boundary': 'Synthetic public development tasks; exact-string plus tool-use scores. '
                    'Memory arms intentionally differ in available information. '
                    'No MFI, broad personalization, independent replication or significance claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['http', 'qwen'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True, help='Immutable model revision or operator-supplied server version')
    p.add_argument('--endpoint')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', required=True, help='New JSON report path; existing files are never overwritten')
    a = p.parse_args()
    output = Path(a.output).expanduser()
    if output.exists(): p.error('Output already exists; choose a new run path.')
    if a.backend == 'http' and not a.endpoint: p.error('--endpoint is required for http')
    if a.backend == 'qwen' and not re.fullmatch(r'[0-9a-f]{40}', a.revision):
        p.error('Qwen revision must be a full 40-character commit SHA.')
    try:
        model = LocalQwen(a.model, a.revision, a.device) if a.backend == 'qwen' else ChatModel(
            a.endpoint, a.model, os.environ.get('FIBRE_AGENT_API_KEY'))
    except (ImportError, OSError, ValueError) as error:
        p.error(f'Backend unavailable ({type(error).__name__}); no evaluation was run.')
    report = evaluate(model)
    report['model'] = {'backend': a.backend, 'id': a.model,
                       'revision': getattr(model, 'revision', a.revision), 'device': a.device if a.backend == 'qwen' else None}
    report['code_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ['evaluate.py', 'assistant.py', 'research_agent.py']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle: json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report['summary']))
    return 2 if any(r['result']['status'] == 'error' for r in report['records']) else 0


if __name__ == '__main__': raise SystemExit(main())
