#!/usr/bin/env python3
"""Fail-closed verifier for the archived R17d aggregate and provenance."""
from pathlib import Path
import json, math, sys, zipfile

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'evidence/confirmed/pythia160m_metric_onsager_r17d'
RAW=BASE/'raw/pythia_r17d_results.zip'
EXTRACTED=BASE/'results/run_summary.json'
EXPECTED='R17D_METRIC_CONSTRAINED_ONSAGER_CONFIRMED'

def main():
    with zipfile.ZipFile(RAW) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f'corrupt raw member: {bad}')
        raw=json.loads(z.read('run_summary.json'))
    ext=json.loads(EXTRACTED.read_text())
    checks={
      'raw_extracted_identical':raw==ext,
      'protocol_exact':raw.get('protocol')=='PYTHIA160M_SST2_AGNEWS_METRIC_ONSAGER_R17D_CONFIRMATORY',
      'status_exact':raw.get('scientific_status')==EXPECTED,
      'five_seeds':len(raw.get('seeds',[]))==5,
      'four_supporting':raw.get('supporting_seed_count')==4,
      'all_gates':bool(raw.get('gates')) and all(raw['gates'].values()),
      'median_adam_margin_positive':raw['medians']['adamw_minus_metric_onsager_loss']>0,
      'median_source_margin_positive':raw['medians']['source_minus_metric_onsager_loss']>0,
      'accuracy_noninferior':raw['medians']['metric_onsager_minus_adamw_accuracy']>=-.005,
      'finite_pairs':all(math.isfinite(v) for p in raw['pairs'] for k,v in p.items() if isinstance(v,float)),
    }
    print(json.dumps(checks,indent=2))
    if not all(checks.values()): return 2
    print(EXPECTED); return 0

if __name__=='__main__': raise SystemExit(main())
