"""Verified compact C29 import. No parent weights or Jacobians are embedded."""
import hashlib
import shutil
import zipfile
from pathlib import Path
from c30_core import *


def probe(z, root, exact_source=False):
    i, s = zread(z, 'identity.json'), zread(z, 'SUMMARY.json')
    require(i['protocol'] == s['protocol'] == PARENT_PROTOCOL and s['complete'] is True, 'Completed C29 result required')
    require(s['decision'] == 'C29_FROZEN_DIRECTION_DIAGNOSIS_COMPLETE', 'Wrong C29 decision')
    require(s['all_writers_unchanged'] and s['all_parent_endpoint_records_replayed'], 'C29 frozen replay not complete')
    pins = read(Path(root) / 'C29_SOURCE_PINS.json')
    require(i['source_hashes'] == pins, 'Exact C29 source revision required')
    if exact_source:
        for n, h in pins.items():
            require(hashlib.sha256(z.read('source/' + n)).hexdigest() == h, 'Archived C29 source differs: ' + n)
    for c, n in [('training', 76), ('inspected', 16)]:
        require(s['endpoint_replay'][c]['c27']['n'] == n, 'Parent denominator differs')
    return i, s


def validate_split(split):
    tr, dv = split['training'], split['prospective_new_states']
    require(len(tr) == 76 and len(dv) == 16, 'Fixed 76+16 denominator differs')
    ids = [t['id'] for t in tr + dv]
    require(len(set(ids)) == 92, 'Duplicate/overlapping state IDs')
    for group, per_case in [(tr, 19), (dv, 4)]:
        require(all(sum(t['case'] == c for t in group) == per_case for c in CASES), 'Route balance differs')
        require(all(t['case'] in CASES and t['id'].split('__')[0] == t['case'] and '/' not in t['id'] and '\\' not in t['id'] and '..' not in t['id'] for t in group), 'Invalid state ID/case')
    return tr, dv


def prepare(parent, out, root):
    out = Path(out)
    provenance = {}
    with zipfile.ZipFile(parent) as z:
        manifest = verify(z)
        _, summary = probe(z, root, True)
        split = zread(z, 'parent/SPLIT.json')
        tr, dv = validate_split(split)
        def cp(old, new):
            dest = out / new
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(old) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 << 20)
            provenance[new] = {'member': old, 'sha256': sha(dest)}
        cp('SUMMARY.json', 'parent/C29_SUMMARY.json')
        cp('identity.json', 'parent/C29_IDENTITY.json')
        cp('parent/SPLIT.json', 'parent/SPLIT.json')
        cp('FROZEN_IDENTITIES.json', 'parent/C29_FROZEN_IDENTITIES.json')
        cp('parent/C27_FROZEN.json', 'parent/C27_FROZEN.json')
        for cohort, items in [('training', tr), ('inspected', dv)]:
            for t in items:
                sid = t['id']
                for name in ('INPUT.pt', 'REFERENCE.json', 'C29_INPUT_AUDIT.json'):
                    cp(f'data/{cohort}/{sid}/{name}', f'data/{cohort}/{sid}/{name}')
                cp(f'diagnosis/{cohort}/{sid}/COMPLETE.json', f'data/{cohort}/{sid}/C29_COMPLETE.json')
        # Inspected target records are copied as opaque bytes here, never parsed for fitting.
        dump(out / 'PARENT_PROVENANCE.json', {
            'parent_zip_sha256': sha(parent), 'verified_members': len(manifest), 'members': provenance,
            'parent_source_revision_verified': True, 'parent_tensor_identity_independently_recomputed': False,
            'writer_identity_source': 'hash-verified C29 replay and frozen-identity records',
            'parent_weights_and_Jacobians_not_embedded': True, 'parent_archives_modified': False})
        dump(out / 'PREPARED.json', {'complete': True, 'training_n': 76, 'inspected_n': 16,
             'inspected_records_copied_without_target_parsing': True})
    print('C30 compact C29 import verified; no model, weights or Jacobians loaded.', flush=True)


def load_cohort(out, cohort, torch):
    import numpy as np
    out = Path(out)
    require(cohort in ('training', 'inspected'), 'Unknown cohort')
    if cohort == 'inspected':
        require((out / 'FROZEN_PREDICTORS.json').exists(), 'Freeze all predictors before inspected inputs/targets are loaded')
        verify_frozen(out)
    split = read(out / 'parent/SPLIT.json')
    tr, dv = validate_split(split)
    items = tr if cohort == 'training' else dv
    pv = read(out / 'PARENT_PROVENANCE.json')['members']
    frozen = read(out / 'parent/C29_FROZEN_IDENTITIES.json')
    X, Y, rows = [], [], []
    for t in items:
        sid = t['id']
        folder = out / 'data' / cohort / sid
        for name in ('INPUT.pt', 'REFERENCE.json', 'C29_INPUT_AUDIT.json', 'C29_COMPLETE.json'):
            rel = (folder / name).relative_to(out).as_posix()
            require(sha(folder / name) == pv[rel]['sha256'], 'Copied parent data changed: ' + rel)
        f = torch.load(folder / 'INPUT.pt', map_location='cpu', weights_only=True)
        for k, shape in [('input', (3147,)), ('features', (3077,))]:
            require(isinstance(f.get(k), torch.Tensor) and f[k].dtype == torch.float32 and tuple(f[k].shape) == shape and bool(torch.isfinite(f[k]).all()), 'Input tensor shape/dtype/finite differs: ' + sid)
        require(isinstance(f.get('route'), torch.Tensor) and f['route'].numel() == 1 and f['route'].dtype in (torch.int32, torch.int64) and int(f['route']) == CASES.index(t['case']), 'Route tensor differs')
        audit = read(folder / 'C29_INPUT_AUDIT.json')
        require(audit['missing_extras_reconstructed'] is False and audit['extras_present'] == ('extras' in f), 'Historical extras audit differs')
        ref = read(folder / 'REFERENCE.json')
        rec = read(folder / 'C29_COMPLETE.json')
        require(rec['id'] == sid and rec['case'] == t['case'] and rec['cohort'] == cohort and rec['frozen'] == frozen, 'C29 endpoint identity differs')
        endpoint = rec['endpoints']['c27']
        m = endpoint['metrics']
        e = np.array(m['response_residual'], dtype=np.float64)
        require(e.shape == (6,) and np.isfinite(e).all(), 'Invalid residual target')
        require(m['theta_sha256'] == rec['theta_sha256']['c27'], 'Residual endpoint binding differs')
        require(abs(float(np.max(np.abs(e))) - m['drift']) <= 1e-7, 'Residual/drift inconsistency')
        require(set(m['gates']) == set(GATES) and all(type(v) is bool for v in m['gates'].values()), 'Gate schema differs')
        require(m['gates']['response'] == (m['drift'] <= SETTINGS['response_budget']) and m['gates']['KL'] == (m['kl'] <= SETTINGS['KL_budget']), 'Gate budget differs')
        require(bool(ref['valid']) == endpoint['reference_valid'] == rec['reference_valid'], 'Reference validity differs')
        # Invalid states are not dropped or replaced. Integrity failure aborts this fixed experiment.
        rows.append({'id': sid, 'case': t['case'], 'valid': bool(ref['valid']),
                     'writer_endpoint_pass': bool(ref['valid'] and all(m['gates'].values())),
                     'missing_historical_extras': 'extras' not in f, 'theta_sha256': m['theta_sha256']})
        X.append(f['input'].double().numpy())
        Y.append(e)
    summary = read(out / 'parent/C29_SUMMARY.json')['endpoint_replay'][cohort]['c27']
    require(summary['n'] == len(rows), 'Parent endpoint denominator mismatch')
    if 'valid' in summary:
        require(summary['valid'] == sum(r['valid'] for r in rows), 'Recounted parent reference validity differs')
    if 'passed' in summary:
        require(summary['passed'] == sum(r['writer_endpoint_pass'] for r in rows), 'Recounted parent writer acceptance differs')
    if 'maximum_R' in summary:
        require(abs(summary['maximum_R'] - float(np.max(np.abs(Y)))) <= 1e-7, 'Recounted parent maximum drift differs')
    return np.stack(X), np.stack(Y), rows


def verify_frozen(out):
    out = Path(out)
    fr = read(out / 'FROZEN_PREDICTORS.json')
    require(fr['settings'] == SETTINGS and fr['all_predictors_frozen_before_inspected_loading'] is True, 'Predictor freeze settings differ')
    files = {p.relative_to(out).as_posix() for p in (out / 'fit').rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    require(files == set(fr['files']), 'Frozen predictor coverage differs')
    for n, h in fr['files'].items():
        require(sha(out / n) == h, 'Frozen predictor modified: ' + n)
    require(sha(out / 'PARENT_PROVENANCE.json') == fr['parent_provenance_sha256'], 'Frozen parent binding differs')
    return fr
