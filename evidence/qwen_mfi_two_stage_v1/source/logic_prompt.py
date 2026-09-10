"""Exact explicit question/answer definitions retained from frozen_logic_use_v1."""
TASKS=('xor','equal')
FORMATS=('digit','three_lines')
FACTS=('00','01','10','11')
def answer(task,state):
 if task not in TASKS or state not in FACTS:raise ValueError('Unknown task/state')
 same=state[0]==state[1]
 return str(int(not same if task=='xor' else same))

def question(task,fmt,facts=None):
 if task not in TASKS or fmt not in FORMATS or facts is not None and facts not in FACTS:raise ValueError('Invalid prompt specification')
 prefix='' if facts is None else f'register_00: {facts[0]}\nregister_01: {facts[1]}\nUse the explicitly provided values above, even if stored values differ.\n'
 rule=f"Return 1 if register_00 and register_01 are {'different' if task=='xor' else 'equal'}, otherwise return 0."
 output='Output one digit only.\nResult:' if fmt=='digit' else 'Output exactly three lines, replacing each <bit> with 0 or 1:\nregister_00: <bit>\nregister_01: <bit>\nRESULT: <bit>\nDo not add any other text.\nResponse:'
 return prefix+'Use the two register values.\n'+rule+'\n'+output

