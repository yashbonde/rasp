# Implementation of "Thinking Like Transformers" (https://arxiv.org/pdf/2106.06981.pdf)
# full repo: https://github.com/tech-srl/RASP
# @yashbonde - 18.06.2021
# MIT License
#
# Why build this?
# - learning how to write languages is the best way to learn how to minimise useless shit
#   and maximise simplicity of code + was fun to code whilst in Deep Thoughts!
#
# Where can I use this?
# - See the examples, if it's not there then will add it later.
#
# Things that are different from the paper
# -
# TODO:
# - implement conditionals
# - additional operators such as `in`, `sort`, `count`

import string
import torch
import einops as ein

vocab = {k:i for i,k in enumerate(string.ascii_lowercase + "$")}
ivocab = {i:k for k,i in vocab.items()}

# ---- built in
def tokens(x):
  """
  # tokens("hello") = [ 7.,  4., 11., 11., 14.]
  # tokens(tokens("hello")) = "hello"
  """

  if isinstance(x, str):
    return torch.Tensor([vocab[t] for t in x.lower()])
  else:
    return "".join([ivocab[t] for t in x.tolist()])

def indices(x):
  # indices("hello") = [0,1,2,3,4]
  return torch.arange(len(x)).float()

def length(x):
  # length("hello") = [5,5,5,5,5]
  return torch.ones(len(x)) * len(x)


# --- element wise
def logical(x, op, y = None):
  # logical(x, "and", y)
  def _or(x, y):
    return torch.logical_or(x.contiguous().view(-1), y.contiguous().view(-1)).view(x.shape)
  def _and(x, y):
    return torch.logical_and(x.contiguous().view(-1), y.contiguous().view(-1)).view(x.shape)
  def _not(x, y):
    return torch.logical_not(x.contiguous().view(-1)).view(x.shape)
  def _xor(x, y):
    return torch.logical_xor(x.contiguous().view(-1), y.contiguous().view(-1)).view(x.shape)
  
  assert op in ["or", "and", "not", "xor"], f"`{op}` not supported"
  if op != "not":
    assert x.shape == y.shape, f"Shapes must be same, got {x.shape}, {y.shape}"
  out = {"or": _or, "and": _and, "not": _not, "xor": _xor}[op](x, y)
  return out

def elementwise(x, op, y):
  # elementwise(x, "-", y)
  if op in ["or", "and", "not", "xor"]:
    return logical(x, op, y)

  def _add(x, y): return x + y
  def _mul(x, y): return x * y
  def _sub(x, y): return x - y
  def _div(x, y):
    out = torch.div(x, y)
    out[out == float("inf")] = 0
    out = torch.nan_to_num(out, 0)
    return out

  assert x.shape == y.shape, f"Shapes must be same, got {x.shape}, {y.shape}"
  assert op in ["+", "-", "*", "/"], f"`{op}` not supported"

  out = {"+":_add, "-":_sub, "*":_mul, "/":_div}[op](x, y)
  return out


# --- select
def select(m1: torch.Tensor, m2, op):
  # creating boolean matrices called "selectors"
  if isinstance(m2, (bool, int)):
    m2 = torch.ones(m1.shape) * m2
  
  assert len(m1.shape) == 1
  assert len(m2.shape) == 1
  
  rows = ein.repeat(m1, "w -> n w", n = m2.shape[0])
  cols = ein.repeat(m2, "h -> n h", n = m1.shape[0]).T

  init_shape = rows.shape
  out = {
    "==": torch.eq,
    "!=": lambda *x: ~torch.eq(*x),
    "<=": torch.less_equal,
    "<": torch.less,
    ">": torch.greater,
    ">=": torch.greater_equal,
  }[op](rows.contiguous().view(-1), cols.contiguous().view(-1))
  out = out.view(*init_shape)

  return out
  
# --- aggregate
def aggregate(s, x, agg = "mean"):
  # collapsing selectors and s-ops into new s-ops
  x = ein.repeat(x, "w -> n w", n = s.shape[0])
  sf = s.float()
  y = x * sf
  
  if agg == "mean":
    ym = y.sum(1) / sf.sum(1)
  else:
    raise ValueError(f"agg: `{agg}` not found")
  
  return torch.nan_to_num(ym, 0)

# --- simple select aggregate
def flip(x):
  i = indices(x); l = length(x)
  return select(i, l-i-1, "==")

# --- selector_width

# def selector_width(x):
#   pass

# def selector_width (sel ,assume_bos = False):
#   light0 = indicator ( indices == 0)
#   or0 = sel or select_eq ( indices ,0)
#   and0 =sel and select_eq ( indices ,0)
#   or0_0_frac =aggregate (or0 , light0 )
#   or0_width = 1 / or0_0_frac
#   and0_width = aggregate (and0 ,light0 ,0)
# 
#   # if has bos , remove bos from width
#   # (doesn ’t count , even if chosen by
#   # sel) and return .
#   bos_res = or0_width - 1
# 
#   # else , remove 0 - position from or0 ,
#   # and re -add according to and0 :
#   nobos_res = bos_res + and0_width
# 
#   return bos_res if assume_bos else
#   nobos_res