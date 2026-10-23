"""Pinned CPU native forwards only; feature and metric points are authorized."""
from core138 import *

def factory(out,data,phase,calls):
    import torch
    path=ROOT/'vendor137/vendor136/vendor135/vendor134/runtime_vendor';sys.path.insert(0,str(path))
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
        def count_forward(self,module,args):
            need(not torch.is_grad_enabled(),'C138 forbids gradient-enabled model forwards');calls['model_forwards']+=1
        def features(self,point,desired):
            phase.consume_feature(point,desired);calls['feature_attempts']+=1
            value=super().features(point,desired);calls['feature_calls']+=1;return value
        def metrics(self,point,case,write=True):
            phase.consume(point,case,write);calls['metric_attempts']+=1;value=super().metrics(point,case,write);calls['metric_calls']+=1;return value
        def jacobian(self,*args,**kwargs):raise ValueError('C138 forbids new derivatives')
        def jacobian10(self,*args,**kwargs):raise ValueError('C138 forbids new derivatives')
        def teacher(self,*args,**kwargs):raise ValueError('C138 forbids teacher, repair and backtracking')
    return CPU()
