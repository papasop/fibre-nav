"""Re-audit immutable checkpoint payloads without loading the pretrained model."""
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT / 'fibre-qwen/protocols/qwen_compiler_cpu_v8/source'
sys.path.insert(0, str(SOURCE))
import protocol as p


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit():
    sha = lambda b: hashlib.sha256(b).hexdigest()
    source_manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
    for name, digest in source_manifest.items():
        require(sha((SOURCE / name).read_bytes()) == digest, name)
    root = 'cpu_compiled_interface/'
    payloads = {}
    hashes = {}
    counts = {}
    for path in sorted(HERE.glob('*results.zip')):
        with zipfile.ZipFile(path) as z:
            require(z.testzip() is None, path.name + ': CRC')
            data = {n[len(root):]: z.read(n) for n in z.namelist()}
            require(len(data) == len(z.namelist()) and all(n.startswith(root) for n in z.namelist()), 'ZIP layout')
            manifest = json.loads(data['RESULT_MANIFEST.json'])
            require(set(data) == set(manifest) | {'RESULT_MANIFEST.json'}, 'ZIP inventory')
            for name, digest in manifest.items():
                require(sha(data[name]) == digest, path.name + ': ' + name)
            for name in [*source_manifest, 'MANIFEST.json']:
                require(data['frozen_source/' + name] == (SOURCE / name).read_bytes(), 'Source drift: ' + name)
            payloads[path.name] = data
            hashes[path.name] = {'sha256': sha(path.read_bytes()), 'verified_payload_hashes': len(manifest)}
            counts[path.name] = len([n for n in data if n.startswith('records/')])
    data = payloads['cpu_compiled_interface_results.zip']
    for name, stage in payloads.items():
        for entry, raw in stage.items():
            if entry.startswith('records/'):
                require(data[entry] == raw, name + ': checkpoint prefix drift')
    lock = json.loads(data['RUN_LOCK.json'])
    require(lock['config'] == p.CONFIG and lock['source_manifest_sha256'] == sha(p.canonical(source_manifest)), 'Run lock')
    env = json.loads(data['environment.json'])
    baseline = json.loads((SOURCE / 'baseline.json').read_text())
    frozen_queries = json.loads((SOURCE / 'FROZEN_QUERIES.json').read_text())
    plan = p.schedule(frozen_queries)
    saved_protocol = json.loads(data['protocol.json'])
    require(saved_protocol['schedule'] == plan and saved_protocol['queries'] == frozen_queries, 'Schedule')
    require(saved_protocol['config'] == p.CONFIG, 'Config')
    require(saved_protocol['baseline_sha256'] == sha((SOURCE / 'baseline.json').read_bytes()), 'Baseline')
    names = sorted(n for n in data if n.startswith('records/'))
    require(names == [f'records/{i:03d}.json' for i in range(103)], 'Record sequence')
    rows = [json.loads(data[n]) for n in names]
    recorded = json.loads(data['summary.json'])
    computed = p.summarize(rows, plan, baseline, env == baseline['environment'])
    require(json.loads(data['PARSER_AUDIT.json']) == p.parser_audit(), 'Negative parser audit')
    # The frozen controller records a PAUSED decision after a budget or keyboard interruption.
    require(recorded['interruption']['kind'] == 'PAUSED', 'Pause status')
    computed.update(interruption=recorded['interruption'], overall_pass=False,
                    confirmation_pass=False, unseen_expression_confirmation=False,
                    semantic_pass=False, native_free_pass=None, compiled_interface_pass=False,
                    compiler_pass=False, constrained_interface_pass=False, integrity_pass=False,
                    constraint_audit_pass=False, deployment_ready=False, decision='PAUSED',
                    session_seconds=recorded['session_seconds'])
    require(recorded == computed, 'Frozen summary replay mismatch')
    log = (HERE / 'QWEN_COMPILER_V8_SESSION_OUTPUT.txt').read_text()
    logged = [json.loads(m) for m in re.findall(r'QWEN_COMPILER_V8_POINT (\{[^\n]*\})', log)]
    require(len(logged) == 103, 'Log count')
    for i, (r, q) in enumerate(zip(rows, logged), 1):
        require(q == dict(n=i, id=r['run_id'], accepted=r['accepted'], parsed=r['parsed'],
                         bounded=r['constrained']['raw'], compiler_matches=i), 'Log mismatch')
    # Independent Boolean oracle and direct raw-field checks, separate from summarize().
    correct = 0
    aligned = []
    mask_ok = 0
    for r in rows:
        a, b = map(int, r['query']['pair'])
        op = r['query']['task']
        answer = str(int({'OR': bool(a or b), 'AND': bool(a and b),
                          'XOR': a != b, 'EQUAL': a == b}[op]))
        logits = r['scores']['bit_logits']
        c = r['constrained']
        correct += int(c['raw'] == answer and str(int(logits[1] > logits[0])) == answer)
        t = c['trace']
        mask_ok += int(t['first_bit_logits'] == t['masked_bit_logits'] and t['finite_bit_count'] == 2)
        aligned.extend(abs(x-y) for k in ['last_no_cache', 'last_cache']
                       for x,y in zip(t['first_bit_logits'],r['numeric'][k]['bit_logits']))
    repeats = rows[64:]
    equal = sum(all(r[k] == rows[i][k] for k in ['scores', 'constrained', 'numeric'])
                for i,r in enumerate(repeats))
    return dict(status='PAUSED_INCOMPLETE', completed=103, expected=128, remaining=25,
                original_overall_pass=False, original_decision=recorded['decision'],
                all_original_summary_fields_reproduced=True,
                independent_primary_and_bounded_correct=correct,
                same_forward_masks_exact=mask_ok, aligned_max_abs_logit_delta=max(aligned),
                first_pass={b:{'primary_correct':computed['groups'][b]['primary_correct'],
                               'bounded_correct':computed['constrained_groups'][b]['correct'],
                               'gate':computed['constrained_groups'][b]['pass_gate']} for b in p.BLOCKS},
                negative_inputs_rejected=computed['parser_negative_audit']['rejected'],
                available_repeats=len(repeats), repeats_all_score_output_numeric_fields_equal=equal,
                legacy_cross_path_failures=computed['numerical_diagnostic']['legacy_cross_path_failures'],
                environment_matches_v6=env == baseline['environment'],
                session_seconds=recorded['session_seconds'], stage_record_counts=counts,
                archive_hashes=hashes, pretrained_model_rerun=False,
                next_run_id=plan[103]['run_id'],
                scope='Frozen external compiler interface only; no parameter-memory or native generalization claim')


if __name__ == '__main__':
    result = audit()
    if sys.argv[1:] == ['--write']:
        (HERE / 'AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        require(not sys.argv[1:], 'Usage: audit_checkpoint.py [--write]')
        require(result == json.loads((HERE / 'AUDIT.json').read_text()), 'Audit drift')
    print(json.dumps({k:v for k,v in result.items() if k not in ['archive_hashes']}, indent=2))
