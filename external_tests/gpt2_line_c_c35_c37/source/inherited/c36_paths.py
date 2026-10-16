"""Bootstrap root and all inherited modules before protocol imports."""
from pathlib import Path
import sys
def configure(root):
    p=Path(root);paths=[p]
    while (p/'inherited').is_dir():
        p=p/'inherited';paths.append(p)
    if (p/'legacy').is_dir():paths.append(p/'legacy')
    sys.path[:0]=list(map(str,paths))
    return paths
