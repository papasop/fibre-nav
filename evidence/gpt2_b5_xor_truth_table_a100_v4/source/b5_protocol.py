import hashlib,json,zipfile
from pathlib import Path
PROTOCOL='GPT2_B5_XOR_TRUTH_TABLE_A100_V4'
SEED=82931
REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
MAX_ACTIONS=12
ACTIONS=tuple([f'READ {a} R{r}' for a in ('00','01','10','11') for r in (0,1)]+['XOR R0 R1 R2']+[f'WRITE {a} R2' for a in ('00','01','10','11')]+['HALT'])
# Examples declare syntax and feedback, but never contain the evaluation address pair.
EXAMPLES='''Task: XOR cells 10 and 11 into cell 00, preserve other cells.
Action: READ 10 R0
Feedback: R0=1
Action: READ 11 R1
Feedback: R1=0
Action: XOR R0 R1 R2
Feedback: R2=1
Action: WRITE 00 R2
Feedback: WRITE accepted
Action: HALT
Feedback: done
'''
TASK='XOR cells 00 and 01 into cell 11, preserve other cells.'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verified_zip(path,protocol=None):
 with zipfile.ZipFile(path) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
  if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('Unsafe archive path')
  if z.testzip():raise ValueError('ZIP CRC failure')
  m=json.loads(z.read('MANIFEST.json'))
  files={n for n in names if not n.endswith('/')}
  if files!=set(m)|{'MANIFEST.json'}:raise ValueError('Manifest coverage mismatch')
  for n,h in m.items():
   if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Hash mismatch: '+n)
  if protocol and json.loads(z.read('identity.json')).get('protocol')!=protocol:raise ValueError('Wrong protocol')
  return m

def dispatch(action,registers,read,write):
 if action not in ACTIONS:return 'ERROR invalid instruction'
 p=action.split()
 if p[0]=='READ':
  registers[p[2]]=int(read(p[1]));return f'{p[2]}={registers[p[2]]}'
 if p[0]=='XOR':
  if not all(k in registers for k in ('R0','R1')):return 'ERROR missing read registers'
  registers['R2']=registers['R0']^registers['R1'];return f'R2={registers["R2"]}'
 if p[0]=='WRITE':
  if 'R2' not in registers:return 'ERROR missing computed register'
  return 'WRITE accepted' if write(p[1],registers['R2']) else 'ERROR endpoint gate rejected'
 return 'done'

def acceptance(trace,start,end,gates,halted):
 # This oracle is evaluation-only and is never supplied to policy or dispatcher.
 reads=[t['action'] for t in trace if t.get('feedback','').startswith(('R0=','R1='))]
 actual_write=any(t['action']=='WRITE 11 R2' and t['feedback']=='WRITE accepted' for t in trace)
 expected=start.copy();expected[3]=start[0]^start[1]
 return {'halted':halted,'input_reads':all(a in reads for a in ('READ 00 R0','READ 01 R1')),
         'xor_executed':any(t['action']=='XOR R0 R1 R2' and t['feedback'].startswith('R2=') for t in trace),
         'output_write_executed':actual_write,'correct_final_contents':end==expected,
         'unselected_retained':end[:3]==start[:3],'endpoint_gates':bool(gates),
         'required_output_flip':start[3]!=expected[3] and end[3]==expected[3]}
