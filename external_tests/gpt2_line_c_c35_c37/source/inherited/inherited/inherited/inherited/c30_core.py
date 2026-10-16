"""C30: preparation-input prediction of the *post-C27-write* response residual.

All fits and selections are training-only. This module never modifies a writer.
"""
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

PROTOCOL = 'GPT2_C30_RESIDUAL_PREDICTABILITY_CPU_V1'
PARENT_PROTOCOL = 'GPT2_C29_FROZEN_DIRECTION_DIAGNOSIS_A100_V1'
RESULT = 'gpt2_c30_residual_predictability_results'
DRIVE_NAME = PROTOCOL + '_RECOVERY'
CASES = ('xor_00', 'xor_01', 'xor_10', 'xor_11')
GATES = ('bits_match', 'output_flip', 'unselected_preserved', 'response', 'KL', 'exact_read_margin', 'development_read_margin')
FAMILIES = ('route_mean', 'pre_response6', 'state70', 'full3147')
SETTINGS = {
    'training_n': 76, 'inspected_n': 16, 'training_per_case': 19,
    'response_budget': .002, 'KL_budget': .005, 'read_margin': .05,
    'families': list(FAMILIES), 'ridge_grid': [1e-6, 1e-4, 1e-2, 1., 100.],
    'outer_folds': 3, 'inner_folds': 3, 'fold_rule': 'sha256(id), route-stratified round-robin',
    'standardisation_active_std': 1e-8, 'standardisation_scale_floor': 1e-6,
    'target': 'six float32-aggregated protected-response residuals at the frozen C27 written endpoint',
    'target_source': 'diagnosis/{cohort}/{id}/COMPLETE.json:endpoints.c27.metrics.response_residual',
    'selection_loss': 'mean squared six-coordinate prediction error divided by response_budget squared',
    'selection_tie_rule': 'first family in fixed order, then first ridge in fixed ascending grid',
    'kernel': 'K=Z Z^T / active_dimension; solve (K+n*lambda*I)alpha=Y-Ymean',
    'per_route_models': True, 'CPU_only': True, 'new_writer_training_steps': 0,
    'new_parameter_writes': 0, 'new_Jacobians': 0, 'new_SVD_calls': 0,
    'fresh_evaluation': False, 'internal_parameter_update': False, 'LC1_closed': False,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read(path):
    def reject(v):
        raise ValueError('Nonfinite JSON: ' + v)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def zread(z, name):
    def reject(v):
        raise ValueError('Nonfinite JSON: ' + v)
    return json.loads(z.read(name), parse_constant=reject)


def verify(z, quiet=False):
    ns = z.namelist()
    require(len(ns) == len(set(ns)), 'Duplicate ZIP member')
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in ns), 'Unsafe ZIP path')
    require('MANIFEST.json' in ns, '请选择结果ZIP，不能使用源码ZIP')
    m = zread(z, 'MANIFEST.json')
    require(isinstance(m, dict) and set(n for n in ns if not n.endswith('/')) == set(m) | {'MANIFEST.json'}, 'Manifest coverage mismatch')
    start = time.monotonic()
    for i, (n, expected) in enumerate(m.items(), 1):
        h = hashlib.sha256()
        with z.open(n) as f:
            for b in iter(lambda: f.read(8 << 20), b''):
                h.update(b)
        require(h.hexdigest() == expected, 'Member digest differs: ' + n)
        if not quiet and (i % 1000 == 0 or time.monotonic() - start > 20):
            print('HASHES_OK', i, '/', len(m), flush=True)
            start = time.monotonic()
    return m


def sources(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def snapshot(out, backup=''):
    import shutil
    out = Path(out)
    path = out.parent / (RESULT + '.zip')
    tmp = path.with_suffix('.zip.tmp')
    m = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
         if p.is_file() and p.name != 'MANIFEST.json' and not p.name.endswith('.tmp') and '__pycache__' not in p.parts}
    dump(out / 'MANIFEST.json', m)
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for n in m:
            z.write(out / n, n)
        z.write(out / 'MANIFEST.json', 'MANIFEST.json')
    with zipfile.ZipFile(tmp) as z:
        verify(z, True)
    os.replace(tmp, path)
    if backup:
        d = Path(backup)
        d.mkdir(parents=True, exist_ok=True)
        dest = d / (RESULT + '_' + str(time.time_ns()) + '.zip')
        t = dest.with_suffix('.zip.tmp')
        shutil.copyfile(path, t)
        require(sha(t) == sha(path), 'Backup copy mismatch')
        os.replace(t, dest)
        good = []
        for p in sorted(d.glob(RESULT + '_*.zip'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            try:
                with zipfile.ZipFile(p) as z:
                    verify(z, True)
                    require(zread(z, 'identity.json')['protocol'] == PROTOCOL, 'Other protocol')
                good.append(p)
            except Exception:
                continue
        for p in good[3:]:
            p.unlink()
        print('DRIVE_BACKUP_OK', dest, flush=True)
    print('RESULT_ARCHIVE=' + str(path), flush=True)
    return path


def folds(ids, n=3):
    require(len(ids) >= n and len(set(ids)) == len(ids), 'Insufficient/duplicate fold IDs')
    order = sorted(range(len(ids)), key=lambda i: (hashlib.sha256(ids[i].encode()).hexdigest(), ids[i]))
    ans = [None] * len(ids)
    for k, i in enumerate(order):
        ans[i] = k % n
    return ans


def columns(family):
    if family == 'route_mean':
        return []
    if family == 'pre_response6':
        return list(range(3077, 3083))
    if family == 'state70':
        return list(range(3077, 3147))
    require(family == 'full3147', 'Unknown feature family')
    return list(range(3147))


def fit(X, Y, family, ridge):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    require(X.ndim == 2 and X.shape[1] == 3147 and Y.shape == (len(X), 6) and len(X) >= 2, 'Training shapes differ')
    require(np.isfinite(X).all() and np.isfinite(Y).all(), 'Nonfinite training arrays')
    ix = np.array(columns(family), dtype=np.int64)
    mu = X[:, ix].mean(0)
    sd = X[:, ix].std(0)
    active = sd > SETTINGS['standardisation_active_std']
    scale = np.maximum(sd, SETTINGS['standardisation_scale_floor'])
    Z = (X[:, ix] - mu) / scale
    Z[:, ~active] = 0
    ym = Y.mean(0)
    d = max(int(active.sum()), 1)
    if family == 'route_mean':
        require(ridge is None, 'Mean baseline must not select ridge')
        beta = np.zeros((0, 6), dtype=np.float64)
    else:
        require(ridge in SETTINGS['ridge_grid'], 'Ridge outside fixed grid')
        K = Z @ Z.T / d
        # The mean/intercept is fitted using exactly these training rows.
        alpha = np.linalg.solve(K + len(X) * ridge * np.eye(len(X)), (Y - ym) / SETTINGS['response_budget'])
        beta = (Z.T @ alpha / d) * SETTINGS['response_budget']
    require(np.isfinite(beta).all(), 'Nonfinite predictor coefficients')
    return {'family': family, 'ridge': ridge, 'ix': ix, 'mean': mu, 'scale': scale,
            'active': active, 'beta': beta, 'target_mean': ym, 'fit_n': len(X)}


def predict(model, X):
    import numpy as np
    X = np.asarray(X, dtype=np.float64)
    require(X.ndim == 2 and X.shape[1] == 3147 and np.isfinite(X).all(), 'Invalid prediction input')
    Z = (X[:, model['ix']] - model['mean']) / model['scale']
    Z[:, ~model['active']] = 0
    out = Z @ model['beta'] + model['target_mean']
    require(out.shape == (len(X), 6) and np.isfinite(out).all(), 'Nonfinite prediction')
    return out


def tune(X, Y, ids, families=FAMILIES):
    """All transformations are re-fitted inside *each* validation split."""
    import numpy as np
    f = np.array(folds(ids, SETTINGS['inner_folds']))
    candidates = []
    best_by_family = {}
    for family in families:
        grid = [None] if family == 'route_mean' else SETTINGS['ridge_grid']
        for ridge in grid:
            pred = np.empty_like(Y, dtype=np.float64)
            for k in range(SETTINGS['inner_folds']):
                train, valid = f != k, f == k
                model = fit(X[train], Y[train], family, ridge)
                pred[valid] = predict(model, X[valid])
            score = float(np.mean(((pred - Y) / SETTINGS['response_budget']) ** 2))
            row = {'family': family, 'ridge': ridge, 'CV_loss': score}
            candidates.append(row)
            if family not in best_by_family or score < best_by_family[family]['CV_loss']:
                best_by_family[family] = row
    best = min(candidates, key=lambda v: v['CV_loss'])
    return {'winner': best, 'by_family': best_by_family, 'candidates': candidates,
            'training_ids': list(ids), 'folds': dict(zip(ids, f.tolist()))}


def nested(X, Y, ids):
    """76 held-out predictions; inner tuning never observes outer validation targets."""
    import numpy as np
    f = np.array(folds(ids, SETTINGS['outer_folds']))
    predictions = {name: np.empty_like(Y) for name in (*FAMILIES, 'train_selected')}
    audits = []
    for k in range(SETTINGS['outer_folds']):
        tr, va = f != k, f == k
        ti = [sid for sid, flag in zip(ids, tr) if flag]
        vi = [sid for sid, flag in zip(ids, va) if flag]
        sel = tune(X[tr], Y[tr], ti)
        for family in FAMILIES:
            spec = sel['by_family'][family]
            model = fit(X[tr], Y[tr], family, spec['ridge'])
            predictions[family][va] = predict(model, X[va])
        spec = sel['winner']
        predictions['train_selected'][va] = predictions[spec['family']][va]
        audits.append({'outer_fold': k, 'fit_ids': ti, 'validation_ids': vi, 'inner_selection': sel})
    return predictions, audits


def report(Y, P, ids, valid):
    """Prediction diagnostics, NEVER a corrected endpoint or an acceptance claim."""
    import numpy as np
    Y, P = np.asarray(Y), np.asarray(P)
    require(Y.shape == P.shape == (len(ids), 6), 'Prediction report denominator differs')
    require(np.isfinite(Y).all() and np.isfinite(P).all(), 'Nonfinite prediction report')
    err = P - Y
    lmax = np.max(np.abs(err), axis=1)
    real = np.max(np.abs(Y), axis=1) <= SETTINGS['response_budget']
    pred = np.max(np.abs(P), axis=1) <= SETTINGS['response_budget']
    yn, pn = np.linalg.norm(Y, axis=1), np.linalg.norm(P, axis=1)
    rows = []
    for i, sid in enumerate(ids):
        rows.append({'id': sid, 'case': sid.split('__')[0], 'reference_valid': bool(valid[i]),
                     'actual_residual': Y[i].tolist(), 'predicted_residual': P[i].tolist(),
                     'error_Linf': float(lmax[i]), 'actual_R': float(np.max(np.abs(Y[i]))),
                     'predicted_R': float(np.max(np.abs(P[i]))),
                     'residual_cosine': float(Y[i] @ P[i] / (yn[i] * pn[i])) if yn[i] * pn[i] > 1e-24 else None,
                     'predicted_to_actual_norm_ratio': float(pn[i] / yn[i]) if yn[i] > 1e-12 else None,
                     'actual_response_gate': bool(real[i]), 'predicted_response_gate': bool(pred[i]),
                     'hypothetical_residual_error_is_not_a_write': True})
    cos = [r['residual_cosine'] for r in rows if r['residual_cosine'] is not None]
    return {'n_requested': len(ids), 'valid_references': sum(bool(v) for v in valid),
            'RMSE': float(np.sqrt(np.mean(err ** 2))), 'RMSE_in_budget_units': float(np.sqrt(np.mean(err ** 2)) / SETTINGS['response_budget']),
            'maximum_prediction_error_Linf': float(lmax.max()), 'median_prediction_error_Linf': float(np.median(lmax)),
            'prediction_error_within_response_budget': int(sum(v and e <= SETTINGS['response_budget'] for v, e in zip(valid, lmax))),
            'actual_endpoint_response_passed': int(sum(v and a for v, a in zip(valid, real))),
            'predicted_response_passed': int(pred.sum()),
            'false_predicted_response_passes': int(np.sum(pred & ~real)),
            'false_predicted_response_failures': int(np.sum(~pred & real)),
            'median_residual_cosine': float(np.median(cos)) if cos else None,
            'common_component_RMSE': float(np.sqrt(np.mean(err.mean(1) ** 2))),
            'contrast_RMSE': float(np.sqrt(np.mean((err - err.mean(1, keepdims=True)) ** 2))),
            'rows': rows, 'new_endpoint_acceptance_tested': False}
