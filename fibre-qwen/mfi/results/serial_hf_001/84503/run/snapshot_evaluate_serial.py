"""Frozen 20-operation endurance audit; original-reference gates, no tuning."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch
import evaluate_tasks as tasks
from core import Budget
from qwen_l1 import MODEL, REVISION
from repair_kl import QwenKLChart
from repair_response import ResponseKLController

SEEDS = [84503, 84521]
TASKS = ['Explain how a rainbow forms.', 'Help me plan a weekend walk.',
         'Explain how to back up my photos.', 'Suggest a simple morning routine.']
STEPS = 20

def run_sequence(chart, controller, evaluate, emit, steps=STEPS):
    initial_bit = controller.read()['value']
    initial_vector = chart.initial.clone()
    rows = []
    stopped = False
    for index in range(steps):
        target = 1-initial_bit if index % 2 == 0 else initial_bit
        operation = 'WRITE' if index == 0 else 'OVERWRITE'
        before = chart.vector().clone()
        result = None
        error = None
        skipped = stopped
        if not stopped:
            try:
                result = controller.execute(operation, target)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
        ok = result is not None and bool(result['accepted'])
        rollback_verified = None if ok or skipped else torch.equal(before, chart.vector())
        if rollback_verified is False:
            raise RuntimeError('Writer failed to restore pre-operation state')
        if not torch.equal(chart.initial, initial_vector):
            raise RuntimeError('Initial parameter reference changed')
        committed = chart.vector().clone()
        audit = chart.audit(target)
        if ok and not controller.passes(audit):
            raise RuntimeError('Committed state failed independent gate recheck')
        items = evaluate(chart, controller, target, ok)
        if not torch.equal(committed, chart.vector()):
            raise RuntimeError('Task evaluation changed committed parameters')
        row = dict(index=index+1, operation=operation, target=target, result=result,
                   error=error, skipped=skipped, operation_pass=ok,
                   rollback_verified=rollback_verified, committed_audit=audit, items=items)
        rows.append(row)
        emit(row)
        stopped = stopped or not ok
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--new-seeds', action='store_true', help='Run second frozen seed')
    a = parser.parse_args()
    seed = SEEDS[int(a.new_seeds)]
    tasks.TASKS[:] = TASKS
    protocol = dict(id='MFI_SERIAL_HF_001', model=MODEL, revision=REVISION,
        seed=seed, frozen_seeds=SEEDS, operations=STEPS, budget=asdict(Budget()),
        writer_source_commit='50d8da2d7ec775af8337c73217b2be28ae0ca711',
        tasks=TASKS, program='WRITE then 19 OVERWRITEs, alternating opposite/initial bit',
        reference='One initial Qwen/LoRA state and anchor responses for all 20 operations; never rebase',
        failure='Rollback failed operation; skip remaining operations; keep all planned items in denominator',
        evaluation='Independent restricted two-token format choice after each operation; no scoring feedback',
        scope='Single bit, hybrid read-to-prompt preference; no free-text quality, L2-L5 or long-term memory claim')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    names = ['evaluate_serial.py','evaluate_tasks.py','qwen_l1.py','core.py','repair_kl.py','repair_response.py']
    hashes = {}
    for name in names:
        data = Path(__file__).with_name(name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        (a.out/('snapshot_'+name)).write_bytes(data)
    (a.out/'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    torch.set_num_threads(4)
    started = time.monotonic()
    chart = QwenKLChart(seed=seed)
    controller = ResponseKLController(chart)
    def emit(row):
        row['seed'] = seed
        (a.out/f'{seed}_{row["index"]:02d}.json').write_text(json.dumps(row, indent=2))
        print(f'SERIAL {row["index"]}/{STEPS}: pass={row["operation_pass"]} skipped={row["skipped"]}',flush=True)
    rows = run_sequence(chart, controller, tasks.evaluate_stage, emit)
    scores = tasks.summarize(rows)
    gates = dict(all_operations_pass=all(r['operation_pass'] for r in rows),
        task_beats_no_memory=scores['mfi']['accuracy']>scores['none']['accuracy'],
        task_matches_external=scores['mfi']['accuracy']>=scores['external']['accuracy'])
    report = dict(protocol=protocol,scores=scores,gates=gates,passed=all(gates.values()),
        operation_pass_count=sum(r['operation_pass'] for r in rows),operation_total=STEPS,
        seconds=time.monotonic()-started,code_sha256=hashes,
        versions={n:importlib.metadata.version(n) for n in ['torch','transformers','safetensors']})
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['passed'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
