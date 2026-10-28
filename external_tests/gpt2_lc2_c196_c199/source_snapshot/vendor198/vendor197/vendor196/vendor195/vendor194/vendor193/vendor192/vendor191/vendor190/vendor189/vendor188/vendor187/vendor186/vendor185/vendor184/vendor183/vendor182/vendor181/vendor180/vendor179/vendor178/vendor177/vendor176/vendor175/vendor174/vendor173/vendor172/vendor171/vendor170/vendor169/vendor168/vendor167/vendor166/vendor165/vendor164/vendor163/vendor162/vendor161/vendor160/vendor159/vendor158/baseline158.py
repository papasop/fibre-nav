"""Full baseline records with explicit cross-run versus within-run roles."""
from core158 import *
from baseline157 import record_comparison as old_record,runtime_provenance as old_provenance
def record(out,label,actual,archived,role='cross_run_descriptive_only'):
    import io
    with contextlib.redirect_stdout(io.StringIO()):r=old_record(out,label,actual,archived)
    r.update(protocol=PROTOCOL,comparison_role=role,blocks_C158_diagnostic=role=='within_run_repeat',C157_decision_reclassified=False)
    dump(Path(out)/(label+'.json'),r);print('C158_BASELINE',label,'role',role,'max_abs_error',r['maximum_absolute_error'],'historical_or_repeat_tolerance',WITHIN_RUN_TOL,'within_tolerance',r['passed_original_replay_tolerance'],flush=True);return r
def provenance(out,torch,current,archived):
    r=old_provenance(out,torch,current,archived);r.update(protocol=PROTOCOL,delivery_revision='C158_NEW_SAME_RUN_PROTOCOL',cross_run_replay_is_descriptive_only=True,C157_decision_reclassified=False)
    dump(Path(out)/'BASELINE_RUNTIME_PROVENANCE.json',r);return r
