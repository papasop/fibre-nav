"""Check compact record integrity and archived acceptance arithmetic; no model execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    provenance = read('PROVENANCE.json')['archives']
    for stage, info in provenance.items():
        manifest = read(stage + '/MANIFEST.json')
        for name, digest in info['copied_members_sha256'].items():
            require(hashlib.sha256((ROOT / stage / name).read_bytes()).hexdigest() == digest,
                    stage + '/' + name + ': copy hash mismatch')
            if name != 'MANIFEST.json':
                require(manifest[name] == digest, stage + ': original manifest mismatch')
    pairs = [('c52', [('baseline', 'baseline36_OOF'), ('sensitive', 'response_sensitive36_OOF')]),
             ('c54', [('reference', 'C52_reference36_OOF'), ('cumulative', 'cumulative36_OOF')])]
    failures = {}
    for stage, arms in pairs:
        summary = read(stage + '/SUMMARY.json')
        rows = read(stage + '/EVALUATION_ROWS.json')
        require(summary['complete'] and summary['decision'] == stage.upper() + '_DEVELOPMENT_FAILURE', stage)
        require(not summary['LC2_closed'] and summary['excluded_class_states_evaluated'] == 0, stage)
        require(len(rows) == len({r['spec']['id'] for r in rows}) == 36, stage)
        for arm, key in arms:
            failed = []
            sampled = 0
            for row in rows:
                record = row[arm]
                metric = record['endpoint_metric']
                gates = metric['gates']
                require(gates['response'] == (metric['response_drift'] <= .002), 'response gate')
                require(gates['KL'] == (metric['KL'] <= .005), 'KL gate')
                require(gates['exact_read_margin'] == (min(metric['exact_margins']) >= .05), 'exact gate')
                require(gates['development_read_margin'] == (min(metric['development_margins']) >= .05), 'development gate')
                passed = all(gates.values())
                require(passed == metric['pass'] == record['gates']['complete_pass'], 'gate conjunction')
                if not passed:
                    failed.append(row['spec']['id'])
                sampled += record['maximum_sampled_response'] <= .002
            require(36 - len(failed) == summary[key]['passed'] == 32, 'pass count')
            require(sampled == summary[key]['sampled_paths_passed'] == 32, 'sampled count')
            failures[(stage, arm)] = set(failed)
    audit = read('c55/SUMMARY.json')
    require(audit['complete'] and audit['decision'] == 'C55_READONLY_TRANSFER_AUDIT_COMPLETE_C54_DECISION_RETAINED', 'audit completion')
    require(audit['parent_decision'] == 'C54_DEVELOPMENT_FAILURE' and not audit['parent_decision_reclassified'], 'parent decision')
    require(not audit['LC2_closed'] and not audit['independent_model_replay'], 'scope')
    for stage in ['c52', 'c54']:
        require(audit['inputs'][stage.upper()]['sha256'] == provenance[stage]['sha256'], 'parent ZIP identity')
    require({r['id'] for r in audit['failure_states']} == failures[('c54', 'cumulative')], 'failure identities')
    rows = read('c55/PAIRED_STATES.json')
    require(len(rows) == 36, 'audit cohort')
    for arm in ['reference', 'cumulative']:
        require({r['id'] for r in rows if not r[arm + '_endpoint_pass']} == failures[('c54', arm)], 'audit failures')
        require(audit['recomputed_scores'][arm]['passed'] == 32, 'audit count')
    print('PASS: compact hashes, parent identities, 36-state gates and four retained failures.')
    print('Not performed: full source/tensor replay, GPT-2 execution, independent confirmation or LC2 acceptance.')


if __name__ == '__main__':
    main()
