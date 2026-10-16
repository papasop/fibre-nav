#!/usr/bin/env python3
"""Read-only checks of reported logs and sources; does not execute GPT-2."""
from pathlib import Path
import ast
import base64
import hashlib
import io
import json
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def check_manifest(root, name):
    entries = json.loads((root / name).read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    require(actual == set(entries) | {name}, 'Manifest coverage mismatch: '+str(root))
    for rel, digest in entries.items():
        p = Path(rel)
        require(not p.is_absolute() and '..' not in p.parts, 'Unsafe manifest member')
        require(sha((root/p).read_bytes()) == digest, 'Hash mismatch: '+rel)
    return len(entries)

def extract_summary(text):
    found = []
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c == '{':
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and 'decision' in obj:
                found.append(obj)
    require(len(found) == 1, 'Expected one top-level decision summary in log')
    return found[0]

def main():
    count = check_manifest(HERE, 'SHA256_MANIFEST.json')
    provenance = json.loads((HERE/'PROVENANCE.json').read_text())
    summaries = {}
    expected = {35:(16,3), 36:(32,5), 37:(32,7)}
    gates = {'bits_match','output_flip','unselected_preserved','response','KL',
             'exact_read_margin','development_read_margin'}
    for run, (n, raw_passes) in expected.items():
        text = (HERE/f'reported/c{run}_execution_log.md').read_text()
        s = json.loads((HERE/f'reported/c{run}_summary_from_log.json').read_text())
        require(extract_summary(text) == s, f'C{run} summary differs from supplied log')
        require(s['complete'] is True, f'C{run} incomplete in log')
        require(s['internal_parameter_update'] is False, 'Scope changed')
        require(s['LC1_independently_confirmed'] is False, 'Independent claim changed')
        require(s['controller_dispatch_tested'] is False, 'Controller claim changed')
        require(s['full_Line_C_complete'] is False, 'Full Line C claim changed')
        for k in ['online_write_Jacobians','online_write_SVD_calls',
                  'online_write_correction_steps','online_write_backtracking']:
            require(s[k] == 0, f'C{run} online geometry/repair reported')
        require(s['teacher_fallback_used'] is False, 'Teacher fallback reported')
        require(s['postwrite_response_used_for_proposal'] is False, 'Postwrite dependency')
        cohort = s['cohorts']['new_local'] if run == 35 else s['new_local']
        require(cohort['raw']['complete_passed'] == raw_passes, 'Raw count changed')
        for arm in ['raw','geometry6','selected6']:
            a = cohort[arm]
            require(a['n_requested'] == n and a['valid_preparations'] == n, 'Denominator changed')
            require(set(a['gate_counts']) == gates, 'Gate set changed')
            if arm != 'raw':
                require(a['complete_passed'] == n, f'C{run}/{arm} not all accepted')
                require(all(v == n for v in a['gate_counts'].values()), 'Gate count mismatch')
                require(a['maximum_response_drift'] <= 0.002, 'Response budget exceeded')
        summaries[run] = s
        print(f'C{run}: raw {raw_passes}/{n}; geometry6 {n}/{n}; selected6 {n}/{n}')
    identity = summaries[35]['coefficients_frozen_sha256']
    require(identity == summaries[36]['C35_coefficients_frozen_sha256'] ==
            summaries[37]['C35_frozen_sha256'] == provenance['coefficient_sha256'],
            'Reported coefficient lineage differs')

    source_root = REPO/'external_tests/gpt2_line_c_c35_c37'
    require(check_manifest(source_root/'source', 'SOURCE_MANIFEST.json') > 250,
            'Incomplete inherited source tree')
    for name, identities in provenance['sources'].items():
        p = source_root/'launchers'/(name+'.py')
        require(sha(p.read_bytes()) == identities['launcher_sha256'], 'Launcher changed')
        values = {}
        for item in ast.parse(p.read_text()).body:
            if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
                key = item.targets[0].id
                if key in ('SOURCE_SHA256','SOURCE_BASE64'):
                    values[key] = ast.literal_eval(item.value)
        payload = base64.b64decode(values['SOURCE_BASE64'],validate=True)
        require(sha(payload) == values['SOURCE_SHA256'] == identities['embedded_source_zip_sha256'],
                'Embedded source identity mismatch')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            require(z.testzip() is None, 'Embedded source CRC failure')
            for member in z.namelist():
                require(not Path(member).is_absolute() and '..' not in Path(member).parts,
                        'Unsafe embedded source path')
            if name.startswith('GPT2_C37_'):
                members = {n for n in z.namelist() if not n.endswith('/')}
                source_files = {p.relative_to(source_root/'source').as_posix()
                                for p in (source_root/'source').rglob('*') if p.is_file()
                                and '__pycache__' not in p.parts and p.suffix != '.pyc'}
                require(members == source_files, 'Inspectable tree differs from embedded C37 source')
                for member in members:
                    require(z.read(member) == (source_root/'source'/member).read_bytes(),
                            'Inspectable source changed: '+member)
    print(f'PASS: {count} evidence hashes; exact log extraction; counts, stated gates and source identities.')
    print('NOT VERIFIED: actual result ZIPs, weight tensors, endpoint measurements or independent replication.')

if __name__ == '__main__':
    main()
