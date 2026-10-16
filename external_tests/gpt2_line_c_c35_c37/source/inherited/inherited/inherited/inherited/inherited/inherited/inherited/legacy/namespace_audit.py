import ast,symtable,builtins
from pathlib import Path
def global_audit(path):
 text=Path(path).read_text();tree=ast.parse(text);defined=set(dir(builtins))|{'__file__','__name__','__package__','__doc__','__annotations__'}
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):defined.add(n.name)
  elif isinstance(n,ast.Import):defined.update(a.asname or a.name.split('.')[0] for a in n.names)
  elif isinstance(n,ast.ImportFrom):
   if any(a.name=='*' for a in n.names):raise ValueError('Wildcard imports forbidden in new worker')
   defined.update(a.asname or a.name for a in n.names)
  elif isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   for t in targets:defined.update(x.id for x in ast.walk(t) if isinstance(x,ast.Name))
 table=symtable.symtable(text,str(path),'exec');missing=set()
 def scan(t):
  for s in t.get_symbols():
   if s.is_global() and s.is_referenced() and s.get_name() not in defined:missing.add(s.get_name())
  for c in t.get_children():scan(c)
 scan(table)
 return missing
