"""Read-only C34 result summary and full member verification."""
import argparse
from c34_protocol import *
p=argparse.ArgumentParser();p.add_argument('zip');a=p.parse_args()
with zipfile.ZipFile(a.zip) as z:
    verify(z);print('ZIP_SHA256',sha(a.zip))
    n='CONSOLE_SUMMARY.json' if 'CONSOLE_SUMMARY.json' in z.namelist() else 'identity.json'
    print(json.dumps(zread(z,n),ensure_ascii=False,indent=2))
