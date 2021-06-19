import unittest
import numpy as np
import torch
from rasp.core import *
from rasp.model import get_model
from rasp.daily import folder
import sys
import os

here = folder(folder(__file__))
sys.path.append(os.path.join(here, "primitives"))

from primitives import functional as F

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestCore(unittest.TestCase):
  
  def test_identity(self):
    # just need to check if it works
    string = "hello"

    # first pass just to check if everything works
    model = get_model()
    logits, loss = model(string)

    # second pass with loss
    target = F.identity(string)
    logits, loss = model(string, target)

if __name__ == "__main__":
  unittest.main()
