import unittest
import numpy as np
import torch


from rasp.core import *

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestCore(unittest.TestCase):
  
  def test_identity(self):
    # just need to check if it works
    model, fn = identity()
    string = "hello"

    # first pass just to check if everything works
    logits, loss = model(string)

    # second pass with loss
    attn = fn(string)
    logits, loss = model(string, (string, attn))

if __name__ == "__main__":
  unittest.main()
