from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent
with zipfile.ZipFile(root/'release_assets/gpt2_b4_a100_results.zip') as z:
    assert z.testzip() is None, 'CRC failure'
    manifest=json.loads(z.read('MANIFEST.json'))
    assert set(manifest)==set(z.namelist())-{'MANIFEST.json'}
    for name,digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==digest, name
    records=json.loads(z.read('records.json'))
    cfg=json.loads(z.read('source/config_audit.json'))
    summary=json.loads(z.read('SUMMARY.json'))
    assert set(records)=={'prepare','normal_writeback','substitute_input','substituted_writeback'}
    for name,r in records.items():
        e=r['evaluation']
        passed=(e['exact_accuracy']==e['heldout_accuracy']==1
                and min(e['exact_margins'])>=.05 and min(e['heldout_margins'])>=.05
                and r['response_drift']<=cfg['response_budget']
                and r['anchor_kl']<=cfg['endpoint_kl_gate'])
        assert passed==r['pass'], name
        assert r['target_bits']==r['observed_bits'], name
    assert summary['checks']['all_endpoint_gates']==all(r['pass'] for r in records.values())
    assert summary['b4_pass']==all(summary['checks'].values())
    assert summary['decision']=='B4_MINIMAL_DEVELOPMENT_PASS'
    assert summary['development_only'] and not summary['independent_confirmation']
print('PASS: CRC, 26 member hashes, recorded endpoint gates and summary consistency')
print('Recorded consistency only; no fresh forward evaluation; see README control limitations.')
