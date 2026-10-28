"""Persist full replay arrays and descriptive errors BEFORE any unchanged replay guard."""
from core157 import *
def record_comparison(out,label,actual,archived):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);a=np.asarray(actual);b=np.asarray(archived)
    need(a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu','Numeric baseline arrays only')
    np.savez_compressed(out/(label+'.npz'),actual=a,archived=b)
    matching=a.shape==b.shape;finite=bool(np.isfinite(a).all() and np.isfinite(b).all());report=dict(label=label,actual_shape=list(a.shape),archived_shape=list(b.shape),actual_dtype=a.dtype.str,archived_dtype=b.dtype.str,shapes_equal=matching,all_values_finite=finite,actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),archived_sha256=hashlib.sha256(b.tobytes()).hexdigest(),original_replay_tolerance=REPLAY_TOL,passed_original_replay_tolerance=False,maximum_absolute_error=None,RMSE=None,number_of_different_entries=None,worst_entries=[],full_arrays_saved=label+'.npz',references_replaced=False,threshold_changed=False,error_bound_established=False,hardware_or_build_causality_established=False)
    if matching and finite and a.size:
        delta=a.astype(np.float64)-b.astype(np.float64);ad=abs(delta);flat=ad.reshape(-1);order=np.argsort(flat,kind='stable')[-min(10,a.size):][::-1]
        report.update(maximum_absolute_error=float(ad.max()),RMSE=float(np.sqrt(np.mean(delta*delta))),number_of_different_entries=int(np.count_nonzero(delta)),passed_original_replay_tolerance=bool(ad.max()<=REPLAY_TOL))
        report['worst_entries']=[dict(index=[int(x) for x in np.unravel_index(int(i),a.shape)],actual=float(a.reshape(-1)[i]),archived=float(b.reshape(-1)[i]),signed_difference=float(delta.reshape(-1)[i])) for i in order]
    dump(out/(label+'.json'),report)
    print('C157_BASELINE_REPLAY',label,'max_abs_error',report['maximum_absolute_error'],'original_tolerance',REPLAY_TOL,'within_original_tolerance',report['passed_original_replay_tolerance'],flush=True)
    return report

def runtime_provenance(out,torch,current_pins,archived_pins):
    import platform
    value=dict(delivery_revision='R2_BASELINE_RECORDING_ONLY',python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),numpy=np.__version__,torch=str(torch.__version__),torch_git_version=getattr(torch.version,'git_version',None),torch_build_config=torch.__config__.show(),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),float32_matmul_precision=torch.get_float32_matmul_precision(),mkldnn_enabled=torch.backends.mkldnn.enabled,mkldnn_available=torch.backends.mkldnn.is_available(),mkl_available=torch.backends.mkl.is_available(),current_MODEL_PINS=current_pins,archived_C155_MODEL_PINS=archived_pins,original_replay_tolerance=REPLAY_TOL,thresholds_changed=False,build_difference_is_not_a_causal_explanation=True)
    dump(Path(out)/'BASELINE_RUNTIME_PROVENANCE.json',value)
    return value
