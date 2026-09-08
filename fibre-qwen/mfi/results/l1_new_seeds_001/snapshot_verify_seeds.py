#!/usr/bin/env python3
"""Frozen new-chart seeds and matched endpoint displacement controls for L1."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import torch
from core import Budget,MFIController,near_kernel
from qwen_l1 import QwenChart,MODEL,REVISION,ANCHORS,MEMORY

SEEDS=[84031,84047,84061]
TRIALS=3
PROTOCOL={'id':'QWEN_MFI_L1_NEW_CHART_SEEDS_001','seeds':SEEDS,'random_trials_per_operation':TRIALS,
          'model':MODEL,'revision':REVISION,'anchors':ANCHORS,'memory_prompt':MEMORY,
          'budget':asdict(Budget()),'target_rule':'WRITE opposite initial bit then OVERWRITE initial bit',
          'controls':'operation-local from true operation start; random initial near-kernel direction normalized to true committed endpoint displacement norm; no retraction or backtracking',
          'qualification':'compare only accepted true operations; missing/failed stages remain failures for full-program gate',
          'gates':'all three true programs accepted; each accepted operation exceeds best random target margin; random displacement norm error <= 1e-5; no-move target gate fails'}


def controls(chart,start,end,value,seed,cfg):
    saved=chart.vector().clone()
    ctl=MFIController(chart,cfg)
    try:
        chart.set_vector(start)
        no_move=chart.audit(value)
        jac=chart.jacobian();norm=float((end-start).norm())
        rows=[]
        generator=torch.Generator().manual_seed(seed)
        for i in range(TRIALS):
            direction=near_kernel(torch.randn(start.shape,generator=generator),jac,cfg.ridge)
            if not torch.isfinite(direction).all() or direction.norm()<=1e-12:
                raise ValueError('Degenerate random direction')
            direction=direction/direction.norm()*norm
            chart.set_vector(start+direction)
            audit=chart.audit(value)
            rows.append({'trial':i,'norm':float((chart.vector()-start).norm()),
                         'norm_error':abs(float((chart.vector()-start).norm())-norm),
                         'audit':audit,'passes':ctl.passes(audit)})
        return {'rng_seed':seed,'matched_endpoint_norm':norm,'no_move':{'audit':no_move,'passes':ctl.passes(no_move)},'random':rows}
    finally:chart.set_vector(saved)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    torch.set_num_threads(4);start_time=time.monotonic();runs=[]
    for seed in SEEDS:
        print('SEED',seed,flush=True)
        chart=QwenChart(seed=seed);ctl=MFIController(chart);initial=ctl.read()['value']
        row={'seed':seed,'initial':ctl.read(),'operations':[],'error':None}
        try:
            for index,(operation,target) in enumerate([('WRITE',1-initial),('OVERWRITE',initial)]):
                before=chart.vector().clone();result=ctl.execute(operation,target);after=chart.vector().clone()
                record={'result':result,'controls':None}
                if result['accepted']:
                    record['controls']=controls(chart,before,after,target,seed+1000+index*100,Budget())
                    record['true_beats_best_random_margin']=result['proposal_audit']['margin']>max(x['audit']['margin'] for x in record['controls']['random'])
                row['operations'].append(record)
                if not result['accepted']:break
        except Exception as error:
            row['error']=type(error).__name__+': '+str(error)
        row['program_pass']=row['error'] is None and len(row['operations'])==2 and all(x['result']['accepted'] for x in row['operations'])
        runs.append(row)
        (a.out/f'seed_{seed}.json').write_text(json.dumps(row,indent=2))
        del ctl,chart
    ops=[op for r in runs for op in r['operations'] if op['controls'] is not None]
    gates={'all_three_programs':all(r['program_pass'] for r in runs),
           'six_qualified_operations':len(ops)==6,
           'true_beats_best_random_each_operation':len(ops)==6 and all(x['true_beats_best_random_margin'] for x in ops),
           'no_move_fails_each_operation':len(ops)==6 and all(not x['controls']['no_move']['passes'] for x in ops),
           'random_norms_match':len(ops)==6 and all(r['norm_error']<=1e-5 for x in ops for r in x['controls']['random'])}
    report={'protocol':PROTOCOL,'protocol_sha256':hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest(),
            'runs':runs,'gates':gates,'passed':all(gates.values()),'seconds':time.monotonic()-start_time,
            'program_pass_count':sum(x['program_pass'] for x in runs),
            'random_pass_count':sum(x['passes'] for op in ops for x in op['controls']['random']),
            'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['verify_seeds.py','qwen_l1.py','core.py']},
            'scope':'Same model/prompts/budgets, three new random-A charts. Operation-local endpoint controls, not matched optimizer trajectories; no significance or universal superiority claim. No L2.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['program_pass_count','random_pass_count','gates','passed','seconds']}))
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
