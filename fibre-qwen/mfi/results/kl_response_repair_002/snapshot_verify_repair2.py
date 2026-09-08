"""Frozen paired development/new-seed audit for candidate KL-aware writing."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from core import Budget, MFIController
from qwen_l1 import QwenChart, MODEL, REVISION
from repair_kl import QwenKLChart
from repair_response import ResponseKLController as KLAwareController
from evaluate_tasks import evaluate_stage, summarize, PROTOCOL as TASK_PROTOCOL

PROTOCOL = {
    'id':'MFI_KL_RESPONSE_REPAIR_002', 'model':MODEL, 'revision':REVISION,
    'development_seeds':[84031,84047,84061], 'heldout_seeds':[84301,84317,84329],
    'algorithms':['original','kl_response'], 'budget':asdict(Budget()),
    'repair':{'activation_fraction':0.5,'inward_fraction':0.25,'response_activation_fraction':0.5,'response_reduction_fraction':0.5},
    'program':'WRITE opposite initial bit then OVERWRITE initial bit; failed WRITE skips OVERWRITE',
    'task_protocol':TASK_PROTOCOL,
    'gates':'all six repaired programs pass; no paired original success regresses; task scores reported separately',
    'development_baseline':'archived task_preference_001; baseline rerun only on new heldout seeds',
    'scope':'development hypothesis selected from prior failures; three unseen chart seeds frozen before any repair run; no universal reliability or significance claim',
}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    files=['core.py','qwen_l1.py','repair_kl.py','repair_response.py','verify_repair2.py','evaluate_tasks.py']
    hashes={}
    for n in files:
        data=Path(__file__).with_name(n).read_bytes();hashes[n]=hashlib.sha256(data).hexdigest()
        (a.out/('snapshot_'+n)).write_bytes(data)
    (a.out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    torch.set_num_threads(4);start=time.monotonic();runs=[]
    for cohort in ['development','heldout']:
        for seed in PROTOCOL[cohort+'_seeds']:
            for algorithm in (['kl_response'] if cohort=='development' else PROTOCOL['algorithms']):
                print('RUN',cohort,seed,algorithm,flush=True)
                chart=(QwenChart if algorithm=='original' else QwenKLChart)(seed=seed)
                ctl=(MFIController if algorithm=='original' else KLAwareController)(chart)
                initial=ctl.read()['value'];stages=[]
                for op,target in [('WRITE',1-initial),('OVERWRITE',initial)]:
                    result=None;error=None;before=chart.vector().clone()
                    try:
                        if op=='WRITE' or ctl.written:result=ctl.execute(op,target)
                    except Exception as exc:error=type(exc).__name__+': '+str(exc)
                    ok=bool(result and result['accepted'])
                    rollback_ok=ok or torch.equal(before,chart.vector())
                    saved=chart.vector().clone()
                    items=evaluate_stage(chart,ctl,target,ok)
                    if not torch.equal(saved,chart.vector()):raise RuntimeError('Task evaluation changed state')
                    stages.append({'operation':op,'result':result,'error':error,'operation_pass':ok,
                                   'rollback_ok':rollback_ok,'items':items})
                row={'cohort':cohort,'seed':seed,'algorithm':algorithm,'stages':stages,
                     'program_pass':all(s['operation_pass'] for s in stages),'scores':summarize(stages)}
                runs.append(row);(a.out/f'{seed}_{algorithm}.json').write_text(json.dumps(row,indent=2))
                print('RESULT',seed,algorithm,row['program_pass'],row['scores'],flush=True)
                del ctl,chart
    summary={}
    for cohort in ['development','heldout']:
        summary[cohort]={}
        for alg in PROTOCOL['algorithms']:
            rs=[r for r in runs if r['cohort']==cohort and r['algorithm']==alg]
            summary[cohort][alg]={'programs_passed':sum(r['program_pass'] for r in rs),'programs_total':len(rs),
                                  'scores':summarize([s for r in rs for s in r['stages']])}
    repaired=[r for r in runs if r['algorithm']=='kl_response']
    gates={'all_repaired_programs':all(r['program_pass'] for r in repaired),
           'known_development_success_retained':next(r for r in repaired if r['seed']==84031)['program_pass'],
           'no_paired_regressions':all(not r['program_pass'] or next(x for x in repaired if x['seed']==r['seed'])['program_pass'] for r in runs if r['algorithm']=='original'),
           'all_failed_operations_rollback':all(s['rollback_ok'] for r in runs for s in r['stages'])}
    report={'protocol':PROTOCOL,'summary':summary,'gates':gates,'passed':all(gates.values()),'seconds':time.monotonic()-start,'source_hashes':hashes}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
