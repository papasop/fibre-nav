"""Read-only C35 archive summary."""
from pathlib import Path
import sys
from c35_paths import configure
configure(Path(__file__).resolve().parent)
from c35_protocol import *
import argparse
p=argparse.ArgumentParser();p.add_argument('zip');a=p.parse_args()
with zipfile.ZipFile(a.zip) as z:
    verify(z);print('ZIP_SHA256',sha(a.zip));n='CONSOLE_SUMMARY.json' if 'CONSOLE_SUMMARY.json' in z.namelist() else 'identity.json'
    print(json.dumps(zread(z,n),ensure_ascii=False,indent=2))
