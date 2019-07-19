import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
print(sys.argv)
print(args)
