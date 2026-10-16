"""Standard-library-only bootstrap; valid before any inherited imports."""
from pathlib import Path
import sys
def paths(root):
    cur=Path(root);out=[cur]
    while (cur/'inherited').is_dir():cur=cur/'inherited';out.append(cur)
    if (cur/'legacy').is_dir():out.append(cur/'legacy')
    return out
def configure(root):
    out=paths(root);sys.path[:0]=list(map(str,out));return out
