"""Pinned native factory resolution without importing Torch or loading a model."""
from core155 import *
def resolve_pinned_factory():
    import runtime154
    need(Path(runtime154.__file__).resolve()==ROOT/'vendor154/runtime154.py','Pinned C154 runtime path');need(runtime154.factory.__module__=='runtime154','Pinned C154 factory identity');runtime154.resolve_pinned_factory();return runtime154.factory
def factory(out,d,phase,calls):return resolve_pinned_factory()(out,d,phase,calls)
