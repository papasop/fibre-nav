"""Pinned GPT2 CPU forward-only adapter. Every metric call is point-authorized."""
from core140 import *

def factory(out,data,phase,calls):
    sys.path.insert(0,str(ROOT/'vendor139/vendor138/vendor137/vendor136/vendor135/vendor134/runtime_vendor'))
    from runtime_base123 import Runtime
    import writer_core as w
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    class CPU(Runtime):
        def __init__(self):
            original=w.make_model
            def instrumented(*args,**kwargs):
                tok,model,params=original(*args,**kwargs);model.register_forward_pre_hook(self.count_forward);return tok,model,params
            w.make_model=instrumented
            try:super().__init__(out)
            finally:w.make_model=original
            calls['initialisation_model_forwards']=calls['model_forwards'];self.pins=clean_json((out/'MODEL_PINS.json').read_bytes());self.online=True
        def count_forward(self,module,args):calls['model_forwards']+=1
        def metrics(self,point,case,write=True):
            phase.consume(point,case,write);calls['preflight_metric_attempts' if phase.phase=='preflight' else 'path_metric_attempts']+=1
            result=super().metrics(point,case,write);calls['preflight_metrics' if phase.phase=='preflight' else 'path_metrics']+=1;return result
        def jacobian(self,*a,**k):raise ValueError('C140 forbids new derivatives')
        def jacobian10(self,*a,**k):raise ValueError('C140 forbids new derivatives')
        def teacher(self,*a,**k):raise ValueError('C140 forbids teacher/repair/backtracking')
        def features(self,*a,**k):raise ValueError('C140 reuses fixed archived proposals; no feature acquisition')
    return CPU()
