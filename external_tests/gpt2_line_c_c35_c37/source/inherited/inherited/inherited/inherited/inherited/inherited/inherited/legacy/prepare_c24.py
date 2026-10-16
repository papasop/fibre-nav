import argparse
from c24_parent import make_capsule

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--companion',required=True);p.add_argument('--capsule',required=True);a=p.parse_args();make_capsule(a.parent,a.companion,a.capsule)

if __name__=='__main__':main()
