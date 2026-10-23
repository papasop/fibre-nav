"""Reuse the pinned real GPT2 CPU metric adapter; no new state/feature access."""
from core146 import *
def factory(out,data,phase,calls):
    from runtime143 import factory as original
    return original(out,data,phase,calls)
