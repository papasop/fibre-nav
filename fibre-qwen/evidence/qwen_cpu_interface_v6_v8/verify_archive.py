"""Verify this archival snapshot without loading model weights (Python stdlib)."""
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'fibre-qwen/protocols/qwen_compiler_cpu_v8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(data):
    return json.loads(data.decode('utf-8'))


def truth(pair, op):
    a, b = map(int, pair)
    return str({'OR': a | b, 'AND': a & b, 'XOR': a ^ b,
                'EQUAL': int(a == b)}[op])


def points(path, prefix, count):
    text = path.read_text(encoding='utf-8')
    rows = [json.loads(m) for m in re.findall(re.escape(prefix) + r' (\{[^\n]*\})', text)]
    require([r['n'] for r in rows] == list(range(1, count + 1)), str(path))
    return text, rows


def report():
    source_manifest = json.loads((SOURCE / 'source/MANIFEST.json').read_text())
    for name, digest in source_manifest.items():
        require(sha((SOURCE / 'source' / name).read_bytes()) == digest, name)
    with zipfile.ZipFile(SOURCE / 'qwen_compiler_cpu_v8.zip') as z:
        require(z.testzip() is None, 'V8 ZIP CRC')
        for name in [*source_manifest, 'MANIFEST.json']:
            matches = [n for n in z.namelist() if n.endswith('/' + name) or n == name]
            require(len(matches) == 1, 'V8 ZIP membership: ' + name)
            require(z.read(matches[0]) == (SOURCE / 'source' / name).read_bytes(), name)
    with zipfile.ZipFile(HERE / 'v6/cpu_operation_rule_results.zip') as z:
        require(z.testzip() is None, 'V6 ZIP CRC')
        prefix = 'cpu_operation_rule/'
        manifest = json_bytes(z.read(prefix + 'RESULT_MANIFEST.json'))
        for name, digest in manifest.items():
            require(sha(z.read(prefix + name)) == digest, 'V6: ' + name)
        record_names = sorted(n for n in z.namelist() if '/records/' in n and n.endswith('.json'))
        require(len(record_names) == 64, 'V6 record count')
        summary = json_bytes(z.read(prefix + 'summary.json'))
        require(summary['complete'] and summary['completed'] == 64, 'V6 completion')
        require(summary['overall_pass'] is False, 'V6 original verdict')
        v6 = {'status': 'FAILED_ORIGINAL_GATES', 'evidence': 'raw_results_zip',
              'completed': 64, 'expected': 64, 'overall_pass': False,
              'payload_hashes_verified': len(manifest),
              'selected_rule_original_summary': {
                  'primary_correct': summary['groups']['selected_rule']['primary_correct'],
                  'strict_free_correct': summary['groups']['selected_rule']['free_correct'],
                  'bounded_correct': summary['constrained_groups']['selected_rule']['correct'],
                  'constraint_audit_pass': summary['constraint_audit_pass'],
                  'decision': summary['decision']}}
    text7, rows7 = points(HERE / 'v7/V7_SESSION_LOG.md', 'QWEN_CONFIRM_V7_POINT', 96)
    summaries = re.findall(r'SUMMARY (\{[^\n]*\})', text7)
    require(len(summaries) == 1, 'V7 log summary')
    summary7 = json.loads(summaries[0])
    require(summary7['overall_pass'] is False and summary7['completed'] == 96, 'V7 verdict')
    counts7 = {}
    errors7 = []
    for r in rows7[:48]:
        group, pair, op = r['id'].split('-')
        correct = r['bounded'] == truth(pair, op)
        counts7[group] = counts7.get(group, 0) + int(correct)
        if not correct:
            errors7.append(r['id'])
    repeated7 = all(all(a[k] == b[k] for k in ['raw', 'bounded', 'same_forward_mask_exact'])
                    for a, b in zip(rows7[:48], rows7[48:]))
    v7 = {'status': 'FAILED_REPORTED_LOG_ONLY', 'evidence': 'session_log_only',
          'completed_reported': 96, 'overall_pass_reported': False, 'raw_results_audited': False,
          'visible_first_pass_bounded_correct': counts7, 'visible_first_pass_errors': errors7,
          'repeated_visible_fields_equal': repeated7,
          'score_and_full_numerical_audit': None}
    text8, rows8 = points(HERE / 'v8/V8_PARTIAL_SESSION_LOG.md', 'QWEN_COMPILER_V8_POINT', 18)
    require('SUMMARY {' not in text8, 'Unexpected final V8 summary')
    correct8 = 0
    for r in rows8:
        group, pair, op = r['id'].split('-')
        require(r['parsed'] == {'op': op, 'a': int(pair[0]), 'b': int(pair[1])}, 'V8 parsed fields')
        require(r['accepted'] is True and r['compiler_matches'] == r['n'], 'V8 compiler log')
        correct8 += int(r['bounded'] == truth(pair, op))
    v8 = {'status': 'PARTIAL_LOG_NO_FINAL_VERDICT', 'evidence': 'session_log_only',
          'observed_points': 18, 'expected_points': 128, 'visible_bounded_correct': correct8,
          'overall_pass': None, 'raw_results_audited': False, 'live_runtime_status': 'unknown'}
    return {'scope': 'retrospective_development_archive', 'model_family': 'Qwen3-1.7B',
            'v6': v6, 'v7': v7, 'v8': v8,
            'pretrained_model_rerun_by_archive_verifier': False,
            'training_or_parameter_memory_write': False, 'deployment_ready': False}


if __name__ == '__main__':
    observed = report()
    if sys.argv[1:] == ['--write-status']:
        (HERE / 'STATUS.json').write_text(json.dumps(observed, indent=2) + '\n')
    elif sys.argv[1:]:
        raise SystemExit('Usage: verify_archive.py [--write-status]')
    else:
        require(observed == json.loads((HERE / 'STATUS.json').read_text()), 'STATUS.json drift')
        entries = (HERE / 'MANIFEST.sha256').read_text().splitlines()
        for line in entries:
            digest, name = line.split('  ', 1)
            require(sha((ROOT / name).read_bytes()) == digest, name)
        print('Archive verified; V6 failed; V7 log-reported failure; V8 partial 18/128.')
        print('No pretrained-model experimental pass is inferred.')
