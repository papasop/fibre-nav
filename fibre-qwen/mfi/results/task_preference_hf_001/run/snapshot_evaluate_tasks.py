"""Frozen one-bit preference task smoke; independent scoring, not an optimizer."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from core import Budget, MFIController
from qwen_l1 import QwenChart, MODEL, REVISION

TASKS = [
    'Explain why the sky looks blue.',
    'Help me prepare for a job interview.',
    'Explain how to organize a weekly study plan.',
    'Give me advice on keeping a tidy desk.',
]
SEEDS = [84031, 84047, 84061]
PROTOCOL = {
    'id': 'MFI_PREFERENCE_TASK_SMOKE_001', 'model': MODEL, 'revision': REVISION,
    'seeds': SEEDS, 'budget': asdict(Budget()), 'tasks': TASKS,
    'label_orders': [['A', 'B'], ['B', 'A']],
    'meaning': {'0': 'brief', '1': 'detailed'},
    'arms': ['mfi', 'external', 'none'],
    'program': 'WRITE opposite initial bit; OVERWRITE initial bit only after accepted WRITE',
    'scoring': 'exact choice of desired format label, restricted two-token logits; no free-text quality claim',
    'mfi': 'committed bit read through MFI then supplied as preference text to updated Qwen',
    'baselines': 'initial frozen chart; external receives target preference text; none receives no preference',
    'failures': 'rejected or skipped MFI stages score zero for all planned items; retained denominator',
    'evaluation': 'no feedback into gradients, anchors, budget, early stopping or controller decisions',
    'gates': 'report operation gates separately; MFI workflow accuracy must exceed none and be >= external',
    'scope': 'deterministic external controller; hybrid read-to-prompt task; four task contexts and label reversals are correlated probes, not independent subjects or broad generalization evidence',
}


def prompt(task, bit, labels):
    preference = 'No saved user preference is available.' if bit is None else (
        'Saved user preference: ' + ('brief answers.' if bit == 0 else 'detailed answers.'))
    return (f'{preference}\nUser request: {task}\n'
            f'Choose the response format. {labels[0]} = brief; {labels[1]} = detailed.\n'
            'Output only the format label.\nAnswer:')


def summarize(rows):
    summary = {}
    for arm in PROTOCOL['arms']:
        items = [x for r in rows for x in r['items'] if x['arm'] == arm]
        summary[arm] = {'correct': sum(x['correct'] for x in items), 'total': len(items),
                        'accuracy': sum(x['correct'] for x in items) / len(items) if items else None}
    return summary


def evaluate_stage(chart, controller, target, operation_ok):
    saved = chart.vector().clone()
    rows = []
    try:
        for arm in PROTOCOL['arms']:
            chart.set_vector(saved if arm == 'mfi' else chart.initial)
            bit = controller.read()['value'] if arm == 'mfi' and operation_ok else (
                target if arm == 'external' else None)
            for task_id, task in enumerate(TASKS):
                for labels in PROTOCOL['label_orders']:
                    # The reference target is used only by the scorer, never in the MFI task prompt.
                    expected = labels[target]
                    predicted = None
                    if arm != 'mfi' or operation_ok:
                        with torch.no_grad():
                            logits = chart.logits(prompt(task, bit, labels))
                        ids = [chart.single(label) for label in labels]
                        predicted = labels[int(logits[ids].argmax())]
                    rows.append({'arm': arm, 'task_id': task_id, 'labels': labels,
                                 'memory_value': bit, 'prediction': predicted, 'expected': expected,
                                 'correct': predicted == expected,
                                 'status': 'operation_unavailable' if arm == 'mfi' and not operation_ok else 'scored'})
    finally:
        chart.set_vector(saved)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(PROTOCOL, indent=2))
    names = ['evaluate_tasks.py', 'qwen_l1.py', 'core.py']
    hashes = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    (a.out/'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    for name in names:
        (a.out/('snapshot_'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    torch.set_num_threads(4); started = time.monotonic(); rows = []
    for seed in SEEDS:
        chart = QwenChart(seed=seed); controller = MFIController(chart)
        initial = controller.read()['value']
        for operation, target in [('WRITE', 1-initial), ('OVERWRITE', initial)]:
            result = None; error = None
            try:
                if operation == 'WRITE' or controller.written:
                    result = controller.execute(operation, target)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            ok = result is not None and result['accepted']
            before_eval = chart.vector().clone()
            items = evaluate_stage(chart, controller, target, ok)
            if not torch.equal(before_eval, chart.vector()):
                raise RuntimeError('Evaluation modified committed parameter state')
            row = {'seed': seed, 'operation': operation, 'target': target, 'result': result,
                   'error': error, 'operation_pass': bool(ok), 'items': items}
            rows.append(row)
            (a.out/f'{seed}_{operation.lower()}.json').write_text(json.dumps(row, indent=2))
        del controller, chart
    scores = summarize(rows)
    gates = {'all_operations_pass': all(r['operation_pass'] for r in rows),
             'task_beats_no_memory': scores['mfi']['accuracy'] > scores['none']['accuracy'],
             'task_matches_external': scores['mfi']['accuracy'] >= scores['external']['accuracy']}
    report = {'protocol': PROTOCOL, 'scores': scores, 'gates': gates,
              'passed': all(gates.values()), 'operation_pass_count': sum(r['operation_pass'] for r in rows),
              'operation_total': len(rows), 'seconds': time.monotonic()-started,
              'code_sha256': hashes,
              'versions': {n: importlib.metadata.version(n) for n in ['torch','transformers','safetensors']}}
    (a.out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
