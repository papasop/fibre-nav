#!/usr/bin/env python3
"""Fresh-process verification against the frozen L1 result; no training updates."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from core import Budget
from qwen_l1 import QwenChart,MODEL,REVISION
from checkpoint import restore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    original=json.loads((a.source/'report.json').read_text())
    checkpoint=a.source/'chart.safetensors'
    declared={name:sha for sha,name in (line.split() for line in (a.source/'SHA256SUMS').read_text().splitlines())}
    for name in ['chart.safetensors','report.json','protocol.json']:
        if hashlib.sha256((a.source/name).read_bytes()).hexdigest()!=declared[name]:
            raise ValueError('Source manifest mismatch: '+name)
    cfg=Budget(**original['protocol']['budget']);expected=original['overwrite']['target']
    protocol={'id':'QWEN_MFI_L1_RELOAD_001','model':MODEL,'revision':REVISION,
              'checkpoint_sha256':declared['chart.safetensors'],
              'expected_bit':expected,'budget':asdict(cfg),'comparison_tolerance':1e-5,
              'source_report_sha256':declared['report.json']}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(4);start=time.monotonic();chart=QwenChart()
    initial=chart.read();initial_vector=chart.vector().clone()
    ctl,loaded=restore(chart,checkpoint,expected_sha256=declared['chart.safetensors'],
        model=MODEL,revision=REVISION,value=expected,budget=cfg)
    old=original['overwrite']['proposal_audit']
    diffs={k:abs(loaded['audit'][k]-old[k]) for k in old}
    committed=chart.vector().clone()
    try:
        restore(chart,checkpoint,expected_sha256=declared['chart.safetensors'],
            model=MODEL,revision=REVISION,value=1-expected,budget=cfg)
        rejected=False
    except ValueError:
        rejected=True
    gates={'bit_matches':loaded['read']['value']==expected,
           'all_B_exact':loaded['all_B_exact'],
           'original_metrics_reproduced':max(diffs.values())<=1e-5,
           'nonzero_parameter_change':float((committed-initial_vector).norm())>0,
           'wrong_content_rejected':rejected,
           'rejected_load_preserves_state':bool(torch.equal(chart.vector(),committed)),
           'controller_initialized':ctl.written}
    report={'protocol':protocol,'initial':initial,'loaded':loaded,'absolute_differences':diffs,
            'gates':gates,'passed':all(gates.values()),'seconds':time.monotonic()-start,
            'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                           for n in ['checkpoint.py','verify_reload.py','core.py','qwen_l1.py']},
            'scope':'Fresh-process reload of the same single-seed endpoint. Not new-seed confirmation or L2. Initial and final bits are both zero; exact tensors and logit-margin reproduction establish endpoint restoration.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
