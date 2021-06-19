from rasp.manual import *
import numpy as np

def identity(input):
  attn = select(indices(input), indices(input), "==")
  attn = [[attn]] # shape: [number of blocks, number of heads in each block]
  return input, attn

def reverse(input):
  # flip = select ( indices , length - indices - 1 ,==)
  # aggregate (flip , tokens )
  i = indices(input)
  l = length(input)
  attn = select(i, l - i, "==")
  attn = [[attn]] # shape: [number of blocks, number of heads in]
  
  if isinstance(input, str):
    out = input[::-1]
  elif isinstance(input, (list, torch.Tensor, np.ndarray)):
    out = [x[::-1] for x in input]
  return out, attn
