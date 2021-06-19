"""

Parser
======

This code will parse the input string code and convert to Module
object that can contain transformer layers as well.

The primary objective is to keep this thing as simple as possible
and ensuring it is easily tokenized for GPTs and parsed by the
code as well.
"""

import re
import os
import importlib
from ast import literal_eval
from rasp.daily import *

import torch
from torch import nn

mod_template = '''class RaspMod(nn.Module):
  def __init__(self):
    super().__init__()
    {init}
{forward}
'''

def load(code):
  # manage the cache folder
  r_cache = os.path.join(folder(__file__), ".rasp_cache")
  os.makedirs(r_cache, exist_ok = True)
  _h = Hashlib.md5(code)
  
  fpath = os.path.join(r_cache, f"{_h}.py")
  print("-->", fpath)
  with open(fpath, "w") as f:
    f.write(code)

  spec = importlib.util.spec_from_file_location("RaspMod", fpath)
  foo = importlib.util.module_from_spec(spec)
  print(foo, dir(foo))
  return foo


  

def get_rsp(
  code,
  *variables,
):
  # args = ", ".join(*variables)
  forward = "\n".join([f"  {x}" for x in code.strip().split("\n")])
  mod = mod_template.format(init = "self.w = nn.Linear(34, 123);", forward = forward)
  print(mod)
  cls = load(mod)()
  return cls

if __name__ == "__main__":
  code_casual_attn = '''
def forward(self, x, y):
  idx = indices(x);
  selectors = select(idx, idx + 1, "<");
  return y + ein.repeat(selectors, 'h w -> b h w n', n = 1, b = y.shape[0]);
'''
  casual_attention = get_rsp(code_casual_attn)

  print(casual_attention)
