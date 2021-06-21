# give proper credits, stole from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# but he won't mind!
# modified for yashbonde/rasp

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from rasp.manual import vocab, tokens


class Config:
  vocab_size = len(vocab)
  n_embd = 18
  block_size = 32
  dropout = 0.0
  n_layer = 1
  n_head = 1
  def __init__(self, **kwargs):
    for k,v in kwargs.items():
      setattr(self, k, v)


class TinyConfig:
  vocab_size = len(vocab)
  n_embd = 18
  block_size = 32
  dropout = 0.0
  n_layer = 1
  n_head = 1
  

class SelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # single qkv like GPT
    self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
    self.split_size = config.n_embd
    
    # output projection
    self.proj = nn.Linear(config.n_embd, config.n_embd)
    
    # causal mask to ensure that attention is only applied to the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                  .view(1, 1, config.block_size, config.block_size))
    self.n_head = config.n_head

  def forward(self, x, attn_mask = None):
    B, T, C = x.size()

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q,k,v = self.qkv(x).split(self.split_size, 2)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask is not None:
      att = att + attn_mask
    att = F.softmax(att, dim=-1)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    # output projection
    y = self.proj(y)
    return y, att

class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embd)
    self.ln2 = nn.LayerNorm(config.n_embd)
    self.split_size = config.n_embd
    self.attn = SelfAttention(config)
    self.mlp = nn.Sequential(
      nn.Linear(config.n_embd, 4 * config.n_embd),
      nn.GELU(),
      nn.Linear(4 * config.n_embd, config.n_embd),
      nn.Dropout(config.dropout),
    )

    self.n_head = config.n_head

  def forward(self, x):
    x, attn_mask = x
    y = self.ln1(x)
    y, att = self.attn(y, attn_mask)
    x = x + y
    x = x + self.mlp(self.ln2(x))
    return [x, att]

class FullTransformer(nn.Module):
  """A full transformer model with focus on data and I/O.
  Can consume strings, lists, arrays and torch-tensors."""

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

    # format input
    key_val_attn_mask = None

    # if isinstance(idx, str):
    #   idx = tokens(idx).to(d).unsqueeze(0)
    # elif isinstance(idx, list) and isinstance(idx[0], str):
    #   idx = torch.cat([tokens(x).unsqueeze(0) for x in idx], dim = 0).to(d)
    # elif isinstance(idx, torch.Tensor) and len(idx.shape) == 1:
    #   idx = idx.unsqueeze(0)
    if not isinstance(idx, torch.Tensor):
      idx = tokens(idx)
    if len(idx.shape) == 1:
      idx = idx.unsqueeze(0)

    if targets is not None:
      assert isinstance(targets, (list, tuple)), \
        "target needs to have a LongTensor and a list/tuple of attn for MSE"

      targets, attn_masks = targets

      # targets for cross-entropy
      if isinstance(targets, str):
        targets = tokens(targets).to(d).view(1, -1)
      elif isinstance(targets, list) and isinstance(targets[0], str):
        targets = torch.cat([tokens(x) for x in targets], dim = 0).to(d)
      elif isinstance(targets, torch.Tensor) and len(targets.shape) == 1:
        targets = targets.view(1, -1)
      targets = targets.long()

      # attention masks
      assert len(attn_masks) == len(self.blocks), "Number of attentions should be same as number of blocks"
      assert isinstance(attn_masks[0], (list, tuple, torch.Tensor)), "Each sequence in the attention should be a tuple/list/tensor"

      for i,(b,a) in enumerate(zip(self.blocks, attn_masks)):
        assert isinstance(a[0], torch.Tensor)
        assert len(a) == b.n_head, f"Number of attn != number of heads in a block. Got: {len(a), b.n_head}"
        attn_masks[i] = torch.cat([aa.float().unsqueeze(0) for aa in a], 0)

      # for i,a in enumerate(attn_masks):
      #   print(":--:", i, a.shape, self.blocks[i].n_head)

      # convert to the final tuple
      targets = (targets, attn_masks)

    return idx, key_val_attn_mask, targets

  def forward(self, idx, targets=None, output_format = None):
    """
    Args:
      idx ([type]): [description]
      targets ([type], optional): Since in rasp you calculate losses for attention matrix as well,
        this targets is a list:
          - torch.LongTensor(): with the cross entropy for entire input tokens, just like a
            normal transformer (GPT/BERT)
          - target_attn_masks:  this is the target matrices for all the attentions in the network.
            ensure that the number of heads and values are common.

    Returns:
        [type]: [description]
    """
    idx, attn_mask, targets = self.format_inputs_and_tokens(idx, targets)

    b, t = idx.size()
    assert t <= self.block_size, "Cannot forward, model block size is exhausted."

    # forward the GPT model
    token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
    position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
    x = self.drop(token_embeddings + position_embeddings)
    all_attn = []
    for b in self.blocks:
      x, att = b([x, attn_mask])
      all_attn.append(att)
    x = self.ln_f(x)
    logits = self.head(x)

    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None:
      targets, attn_masks = targets
      
      # Cross Entropy loss
      ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

      # MSE-loss, for each head, manually
      mse_loss = 0
      for a,t in zip(all_attn, attn_masks):
        t = torch.tile(t, [a.shape[0], 1, 1, 1])
        # print(a.shape, t.shape) # [b, n_head, s, s]

        mse_loss += F.mse_loss(a, t)
      loss = ce_loss + mse_loss

    if output_format is not None:
      raise NotImplementedError('this functionality has not been build, hold on!')

    return logits, loss


def get_model(**kwargs):
  config = Config(**kwargs)
  return FullTransformer(config)

