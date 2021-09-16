from argparse import Namespace
import argparse

from invariant import train_and_test_irm
from erm import train_and_test_erm

def str2bool(v):
  if type(v) == bool:
    return v
  return str(v).lower() in ("yes", "true", "t", "1")

args_dict = {
  'cges': (str2bool, True),
  'lamb': (float, 0.0006),
  'mu': (float, 0.8),
  'chvar': (float, 0.2),
  'lr': (float, 0.001),
  'num_workers': (int, 16),
  'batch_size': (int, 128),
  'epochs': (int, 500),
  'erm': (str2bool, False),
  'irm': (str2bool, False),
  'run_name': (str, '')
}

parser = argparse.ArgumentParser()
for k,v in args_dict.items():
  parser.add_argument('--' + k, type=v[0], default=v[1])

args = parser.parse_args()

print(args)

"""MAIN"""

if args.erm:
  train_and_test_erm(args)
if args.irm:
  train_and_test_irm(args)

