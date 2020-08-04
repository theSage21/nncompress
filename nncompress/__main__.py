import argparse
from nncompress.core import compress

parser = argparse.ArgumentParser()
parser.add_argument("fname")
parser.add_argument("--config", default="config.json")
args = parser.parse_args()


compress(args)
