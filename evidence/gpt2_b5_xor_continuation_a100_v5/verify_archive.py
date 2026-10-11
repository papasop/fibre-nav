"""Read-only checks of archived B5 V4/V5 evidence. Standard library only.

This verifies hashes, recorded gates, traces and tensor-storage identity.
It does not load tensors, execute GPT-2 or establish independent replication.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

V5_SHA = '30ac85d7208ba1525b7cc8934ffc94e4d42bb8b593a17bba715db2f652dd66dd'
V4_SHA = '3e9ba8a1f4f37adf9458d1443d59e3ac19b0763c3b1cd80b2ce8af1ee298c301'
V5_PROTOCOL = 'GPT2_B5_XOR_CONTINUATION_A100_V5'
V4_PROTOCOL = 'GPT2_B5_XOR_TRUTH_TABLE_A100_V4'
CASES = ('xor_00', 'xor_01', 'xor_10', 'xor_11')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_zip(data, expected_sha=None):
    if expected_sha is not None:
        require(digest(data) == expected_sha, 'Published archive SHA-256 mismatch')
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    require(len(names) == len(set(names)), 'Duplicate ZIP member')
    require(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                and '\\' not in n for n in names), 'Unsafe ZIP path')
    require(z.testzip() is None, 'ZIP CRC failure')
    manifest = json.loads(z.read('MANIFEST.json'))
    require(set(n for n in names if not n.endswith('/')) == set(manifest) | {'MANIFEST.json'},
            'Manifest coverage mismatch')
    for name, sha in manifest.items():
        require(digest(z.read(name)) == sha, 'Member digest mismatch: ' + name)
    return z, len(manifest)


def read(z, name):
    return json.loads(z.read(name))


def recorded_gate_pass(gate, cfg):
    ev = gate['evaluation']
    exact = ev['exact_margins']
    para = ev['heldout_margins']  # Reused paraphrases: development, not fresh holdouts.
    require(len(exact) == 4 and len(para) == 12, 'Readout coverage mismatch')
    numbers = exact + para + [gate['drift'], gate['kl']]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers),
            'Nonfinite recorded metric')
    require(ev['minimum_exact_margin'] == min(exact), 'Exact minimum mismatch')
    require(ev['minimum_heldout_margin'] == min(para), 'Paraphrase minimum mismatch')
    require(ev['exact_accuracy'] == sum(v > 0 for v in exact) / 4,
            'Exact accuracy inconsistent with margins')
    # Float32 accuracy can differ slightly from Python's rational value.
    require(abs(ev['heldout_accuracy'] - sum(v > 0 for v in para) / 12) < 1e-6,
            'Paraphrase accuracy inconsistent with margins')
    return (ev['exact_accuracy'] == ev['heldout_accuracy'] == 1
            and min(exact) >= 0.05 and min(para) >= 0.05
            and gate['drift'] <= cfg['response_budget']
            and gate['kl'] <= cfg['endpoint_kl_gate'])


def storage_hashes(data):
    """Read tensor storage bytes without pickle execution or torch import."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return [digest(z.read(n)) for n in z.namelist() if '/data/' in n]


def successful_case(z, name, cfg):
    prep = read(z, name + '/PREPARATION.json')
    rec = read(z, name + '/record.json')
    end = read(z, name + '/COMPLETE.json')
    a, b = int(name[-2]), int(name[-1])
    initial = [a, b, 1, 1 - (a ^ b)]
    final = [a, b, 1, a ^ b]
    require(prep['target_bits'] == rec['start_bits'] == initial, 'Wrong preparation: ' + name)
    require(prep['pass'] and prep['gate']['pass'] and recorded_gate_pass(prep['gate'], cfg),
            'Preparation gate failure: ' + name)
    require(prep['gate']['observed_bits'] == initial, 'Preparation read mismatch: ' + name)
    require(end['final_bits'] == final and end['final_bits'][:3] == initial[:3],
            'Wrong final/retained contents: ' + name)
    require(initial[3] != final[3], 'Trivial no-flip episode: ' + name)
    expected_actions = ['READ 00 R0', 'READ 01 R1', 'XOR R0 R1 R2', 'WRITE 11 R2', 'HALT']
    expected_feedback = [f'R0={a}', f'R1={b}', f'R2={a ^ b}', 'WRITE accepted', 'done']
    require(rec['trace'] == end['trace'], 'Trace duplication mismatch: ' + name)
    require([t['action'] for t in rec['trace']] == expected_actions, 'Actions: ' + name)
    require([t['feedback'] for t in rec['trace']] == expected_feedback, 'Feedback: ' + name)
    require(rec['halted'] and end['pass'] and all(end['checks'].values()), 'Acceptance: ' + name)
    require(len(rec['writer_records']) == 1, 'Accepted write record missing: ' + name)
    w = rec['writer_records'][0]
    require(w['address'] == '11' and w['value'] == a ^ b and w['target_bits'] == final,
            'Writer operand/result mismatch: ' + name)
    for gate in (w['gate'], end['final_gate']):
        require(gate['pass'] and recorded_gate_pass(gate, cfg), 'Recorded endpoint gate: ' + name)
        require(gate['observed_bits'] == final, 'Endpoint read mismatch: ' + name)
    require(w['accepted'] and math.isfinite(w['controller_probe_score_drift'])
            and w['controller_probe_score_drift'] <= 1e-5, 'Writer/probe check: ' + name)
    return {'case': name, 'initial_bits': initial, 'final_bits': final,
            'minimum_exact_margin': end['final_gate']['evaluation']['minimum_exact_margin'],
            'minimum_development_paraphrase_margin': end['final_gate']['evaluation']['minimum_heldout_margin'],
            'response_drift': end['final_gate']['drift'], 'anchor_mean_kl': end['final_gate']['kl']}


def verify_v4(z):
    summary = read(z, 'SUMMARY.json')
    cfg = read(z, 'source/config_audit.json')
    require(read(z, 'identity.json')['protocol'] == summary['protocol'] == V4_PROTOCOL, 'V4 protocol')
    require(summary['complete'] and summary['n_completed'] == 4 and summary['n_passed'] == 2,
            'V4 historical outcome changed')
    require(summary['decision'] == 'B5_ACTUAL_XOR_TRUTH_TABLE_DEVELOPMENT_FAIL', 'V4 decision')
    require([summary['cases'][c]['pass'] for c in CASES] == [True, True, False, False], 'V4 case flags')
    successful_case(z, 'xor_00', cfg)
    successful_case(z, 'xor_01', cfg)
    prep10 = read(z, 'xor_10/PREPARATION.json')
    require(not prep10['pass'] and not recorded_gate_pass(prep10['gate'], cfg), 'V4 xor_10 failed preparation')
    write11 = read(z, 'xor_11/record.json')['writer_records'][0]
    require(not write11['accepted'] and not recorded_gate_pass(write11['gate'], cfg), 'V4 xor_11 rejected candidate')


def verify(path, version='v5'):
    expected = V5_SHA if version == 'v5' else V4_SHA
    z, count = checked_zip(Path(path).read_bytes(), expected)
    with z:
        if version == 'v4':
            verify_v4(z)
            return {'archive_hashes': count, 'historical_passes': 2, 'historical_cases': 4,
                    'status': 'V4_RECORDED_FAILURE_VERIFIED', 'new_model_evaluation': False}
        identity = read(z, 'identity.json')
        summary = read(z, 'SUMMARY.json')
        plan = read(z, 'CONTINUATION_PLAN.json')
        cfg = read(z, 'source/config_audit.json')
        require(identity['protocol'] == summary['protocol'] == plan['protocol'] == V5_PROTOCOL, 'V5 protocol')
        require(cfg['response_budget'] == .002 and cfg['endpoint_kl_gate'] == .005, 'Original budgets')
        require(identity['preparation_total'] == plan['preparation_total'] == 1200
                and identity['write_total'] == plan['write_total'] == 1080, 'Step limits')
        parents = {}
        parent_hash_counts = {}
        for name, key in [('parent_v4.zip', 'parent_v4_sha256'), ('parent_v3.zip', 'parent_v3_sha256'),
                          ('parent_b4.zip', 'parent_sha256')]:
            parent, hashes = checked_zip(z.read(name), identity[key])
            parents[name] = parent
            parent_hash_counts[name] = hashes
        try:
            p4 = parents['parent_v4.zip']
            require(identity['parent_v4_sha256'] == V4_SHA, 'V4 provenance hash')
            verify_v4(p4)
            for name in ('parent_b4.zip', 'parent_v3.zip'):
                require(z.read(name) == p4.read(name), 'Nested parent identity: ' + name)
            for case in ('xor_00', 'xor_01'):
                for name in z.namelist():
                    if name.startswith(case + '/'):
                        require(z.read(name) == p4.read(name), 'Inherited record changed: ' + name)
            for name in ('writer_core.py', 'run_l5_audit.py', 'per_address_repair.py', 'config_audit.json'):
                require(z.read('source/' + name) == p4.read('source/' + name), 'Writer/config changed: ' + name)
            require(z.read('controller_training.pt') == p4.read('controller_training.pt'), 'Controller checkpoint changed')
            require(not plan['thresholds_changed'] and not plan['writer_changed'] and not plan['controller_retrained']
                    and not plan['fresh_evaluation'] and not plan['independent_memory_seed'], 'Continuation scope')
            require(read(z, 'CONTROLLER_FROZEN.json')['new_training_steps'] == summary['controller_new_training_steps'] == 0,
                    'Controller training count')
            require(read(z, 'PHASE_PROGRESS.json') == {'xor_10/prepare': 1200,
                    'xor_10/write_attempt_0': 1080, 'xor_11/write_attempt_0': 1080}, 'Saved phase progress')
            rows = []
            for case in CASES:
                rows.append(successful_case(z, case, cfg))
                end = read(z, case + '/COMPLETE.json')
                require(summary['cases'][case]['pass'] == end['pass']
                        and summary['cases'][case]['checks'] == end['checks']
                        and summary['cases'][case]['final_bits'] == end['final_bits'], 'Summary consistency: ' + case)
            for case in ('xor_10', 'xor_11'):
                rec = read(z, case + '/record.json')
                w = rec['writer_records'][0]
                require(w['candidate_saved'] and w['planned_total_steps'] == 1080, 'Candidate retention: ' + case)
                require(case + '/' + w['optimizer_checkpoint'] in z.namelist(), 'Optimizer checkpoint absent: ' + case)
                candidate = storage_hashes(z.read(case + '/write_attempt_0_candidate.pt'))
                state = storage_hashes(z.read(case + '/state.pt'))
                require(len(candidate) == 1 and candidate[0] in state, 'Candidate storage not in final state: ' + case)
            for name, sha in identity['source_hashes'].items():
                require(digest(z.read('source/' + name)) == sha, 'Frozen source identity: ' + name)
            sm = read(z, 'source/SOURCE_MANIFEST.json')
            require(set('source/' + n for n in sm) | {'source/SOURCE_MANIFEST.json'} ==
                    set(n for n in z.namelist() if n.startswith('source/')), 'Source manifest coverage')
            for name, sha in sm.items():
                require(digest(z.read('source/' + name)) == sha, 'Source snapshot digest: ' + name)
            require(summary['complete'] and summary['n_planned'] == summary['n_completed'] == summary['n_passed'] == 4,
                    'V5 completion/coverage')
            require(summary['decision'] == 'B5_XOR_CONTINUATION_DEVELOPMENT_PASS', 'V5 decision')
            require(summary['development_only'] and not summary['full_b5_confirmation']
                    and not summary['internal_parameter_update'] and summary['single_backbone']
                    and not summary['shared_update_chart'], 'Evidence boundaries')
            require(summary['case_origins'] == plan['case_origins'], 'Origin declarations')
            require(plan['case_origins']['xor_00'] == plan['case_origins']['xor_01'] == 'INHERITED_V4_PASS_NOT_RERUN',
                    'Inherited case origins')
            # Convenience JSON/source copies must reproduce the original bytes.
            for name in z.namelist():
                if name.startswith('source/') or name.endswith('.json'):
                    local = Path(path).parent / (name if name.startswith('source/') else 'recorded/' + name)
                    if local.exists():
                        require(local.read_bytes() == z.read(name), 'Convenience copy changed: ' + str(local))
            return {'status': 'V5_RECORDED_DEVELOPMENT_EVIDENCE_VERIFIED', 'archive_hashes': count,
                    'parent_hashes': parent_hash_counts, 'n_passed': 4, 'n_cases': 4, 'cases': rows,
                    'new_model_evaluation': False, 'independent_b5_confirmation': False}
        finally:
            for parent in parents.values():
                parent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', nargs='?', type=Path)
    parser.add_argument('--version', choices=('v4', 'v5'), default='v5')
    args = parser.parse_args()
    default = ('gpt2_b5_xor_continuation_a100_results.zip' if args.version == 'v5'
               else 'gpt2_b5_xor_truth_table_a100_results.zip')
    path = args.archive or Path(__file__).resolve().parent / default
    print(json.dumps(verify(path, args.version), indent=2))


if __name__ == '__main__':
    main()
