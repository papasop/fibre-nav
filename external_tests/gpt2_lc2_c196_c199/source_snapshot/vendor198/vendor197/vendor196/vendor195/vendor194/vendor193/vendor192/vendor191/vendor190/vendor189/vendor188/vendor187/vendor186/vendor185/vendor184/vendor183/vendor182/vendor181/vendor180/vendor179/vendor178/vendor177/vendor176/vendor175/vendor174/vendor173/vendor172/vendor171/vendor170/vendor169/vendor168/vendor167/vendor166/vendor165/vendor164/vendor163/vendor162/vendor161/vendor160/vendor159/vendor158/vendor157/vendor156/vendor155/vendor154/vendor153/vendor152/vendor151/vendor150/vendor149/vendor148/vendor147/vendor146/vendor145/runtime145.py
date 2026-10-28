"""The inherited pinned CPU GPT2 metric adapter; no feature or teacher access."""
from core145 import *

def factory(out,data,phase,calls):
    from runtime143 import factory as original
    return original(out,data,phase,calls)
