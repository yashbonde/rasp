# give proper credits, stole from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# but he won't mind!
# modified for yashbonde/rasp

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from rasp.manual import vocab, tokens


class TinyConfig:
  vocab_size = len(vocab)
  n_embd = 18
  block_size = 32
  dropout = 0.0
  n_layer = 1
  n_head = 1
  

class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.split_size = config.n_embd
    self.attn = nn.MultiheadAttention(
      embed_dim = config.n_embd,
      num_heads = config.n_head,
      dropout=config.dropout,
    )
    self.mlp = nn.Sequential(
      nn.Linear(config.n_embd, 4 * config.n_embd),
      nn.GELU(),
      nn.Linear(4 * config.n_embd, config.n_embd),
      nn.Dropout(config.dropout),
    )

  def forward(self, x):
    y = self.ln1(x)
    # this rearrange is not needed from torch>=1.9.0
    y = rearrange(y, "n l e -> l n e")
    y = self.attn(y, y, y)[0]
    y = rearrange(y, "l n e -> n l e")
    x = x + y
    x = x + self.mlp(self.ln2(x))
    return x

class FullTransformer(nn.Module):
  """ the full GPT language model, with a context size of block_size """

  def __init__(self, config):
    super().__init__()

    # input embedding stem
    self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
    self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
    self.drop = nn.Dropout(config.dropout)
    
    # transformer
    self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
    
    # decoder head
    self.ln_f = nn.LayerNorm(config.n_embd)
    self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.block_size = config.block_size

  @property
  def num_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def get_device(self):
    return next(self.parameters()).device
  
  def format_inputs_and_tokens(self, idx, targets):
    d = self.get_device()
    if isinstance(idx, str):
      idx = tokens(idx).to(d).view(1, -1)
    elif isinstance(idx, list) and isinstance(idx[0], str):
      idx = torch.cat([tokens(x) for x in idx], dim = 0).to(d)
    elif isinstance(idx, torch.Tensor) and len(idx.shape) == 1:
      idx = idx.view(1, -1)
    
    idx = idx.long()

    if targets is not None:
      if isinstance(targets, str):
        targets = tokens(targets).to(d).view(1, -1)
      elif isinstance(targets, list) and isinstance(targets[0], str):
        targets = torch.cat([tokens(x) for x in targets], dim = 0).to(d)
      elif isinstance(targets, torch.Tensor) and len(targets.shape) == 1:
        targets = targets.view(1, -1)
      
      targets = targets.long()
    
    return idx, targets

  def forward(self, idx, targets=None):
    idx, targets = self.format_inputs_and_tokens(idx, targets)

    b, t = idx.size()
    assert t <= self.block_size, "Cannot forward, model block size is exhausted."

    # forward the GPT model
    token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
    position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.head(x)

    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return logits, loss

