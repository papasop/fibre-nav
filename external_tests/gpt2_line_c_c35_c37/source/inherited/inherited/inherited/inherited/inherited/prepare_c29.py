import argparse
from pathlib import Path
from c29_protocol import prepare_parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--out',required=True);a=p.parse_args();prepare_parent(a.parent,a.out,Path(__file__).resolve().parent)
