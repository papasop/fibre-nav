"""Current entrypoint wraps the unchanged pinned native CPU runtime, forward only."""
from core154 import *
def resolve_pinned_factory():
    import runtime152 as native
    need(Path(native.__file__).resolve()==ROOT/'vendor153/vendor152/runtime152.py','Pinned native runtime source path');need(native.factory.__module__=='runtime152','Pinned native runtime factory identity');return native.factory
def factory(out,d,phase,calls):return resolve_pinned_factory()(out,d,phase,calls)
