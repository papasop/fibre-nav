from core152 import *
def factory(out,d,phase,calls):
    from runtime143 import factory as pinned
    return pinned(out,d,phase,calls)
