"""Teacher data is used only for controller training; evaluation uses runtime feedback."""
from itertools import permutations, product
ADDRESSES=('00','01','10','11')
HELDOUT=(('00','01','11'),('01','00','11'))
TRAIN_TUPLES=tuple(t for t in permutations(ADDRESSES,3) if t not in HELDOUT)
CONTROLLER_SEED=82932
TRAIN_STEPS=600
BATCH_SIZE=8
LEARNING_RATE=0.002

def prompt(task,trace):
 a,b,c=task
 history=''.join('Action: '+t['action']+'\nFeedback: '+t['feedback']+'\n' for t in trace)
 return ('Memory VM. READ address register; XOR R0 R1 R2; WRITE address R2; HALT. '
         'Read both inputs before XOR. Write the result then halt.\n'
         f'Task: XOR cells {a} and {b} into cell {c}, preserve other cells.\n'+history+'Action:')

def training_samples():
 rows=[]
 for task in TRAIN_TUPLES:
  a,b,c=task
  for x,y in product((0,1),repeat=2):
   trace=[]
   for action,feedback in [(f'READ {a} R0',f'R0={x}'),(f'READ {b} R1',f'R1={y}'),
                           ('XOR R0 R1 R2',f'R2={x^y}'),(f'WRITE {c} R2','WRITE accepted'),('HALT','done')]:
    rows.append({'task':list(task),'prompt':prompt(task,trace),'action':action})
    trace.append({'action':action,'feedback':feedback})
 return rows
