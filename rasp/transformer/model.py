# give proper credits, stole from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# but he won't mind!
# modified for yashbonde/rasp

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange


class TinyConfig:
  vocab_size = 128
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
    self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
    self.attn = nn.MultiheadAttention(
      embed_dim = config.n_embd,
      num_heads = config.n_head,
      dropout=config.dropout
    )
    self.mlp = nn.Sequential(
      nn.Linear(config.n_embd, 4 * config.n_embd),
      nn.GELU(),
      nn.Linear(4 * config.n_embd, config.n_embd),
      nn.Dropout(config.dropout),
    )

  def forward(self, x):
    y = self.ln1(x)
    qkv = self.qkv(y).split(self.split_size, 2)
    qkv = list(map(lambda x: rearrange(x, "n l e -> l n e"), qkv))
    attn_out = self.attn(*qkv)[0]
    x = x + rearrange(attn_out, "l n e -> n l e")
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
    self.apply(self._init_weights)

  def get_block_size(self):
    return self.block_size

  @property
  def num_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)

  def forward(self, idx, targets=None):
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

