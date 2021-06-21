# give proper credits, stole from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# but he won't mind!
# modified for yashbonde/rasp

import math
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import einops as ein

from rasp.manual import vocab, tokens

# ------ configurations ------ #

class Config:
  def __init__(self, **kwargs):
    self.vocab_size = len(vocab)
    self.n_embd = 18
    self.block_size = 32
    self.dropout = 0.0
    self.n_layer = 1
    self.n_head = 1
    for k,v in kwargs.items():
      setattr(self, k, v)

  def get_json(self):
    return json.dumps(vars(self))

# ------ response ------ #

class Response:
  def __init__(self, logits, loss, attns):
    self.logits = logits
    self.loss = loss
    self.attns = attns
    self.tokens = tokens(logits.argmax(-1))

# ------ model ------ #

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

    self.config = config

  @property
  def num_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def get_device(self):
    return next(self.parameters()).device
  
  def format_inputs_and_tokens(self, idx, targets):
    d = self.get_device()
    P_token = "$"; P_id = vocab[P_token];

    # format input and convert to tokens
    if not isinstance(idx, torch.Tensor):
      idx = tokens(idx)
    if len(idx.shape) == 1:
      idx = idx.unsqueeze(0)
    
    # create attention masks as follows:
    #
    # input --> "wd"
    #
    # [[       0.,        0.,        0.,        0.,        0.],
    #  [       0.,        0.,        0.,        0.,        0.],
    #  [       0.,        0., -1000000., -1000000., -1000000.],
    #  [       0.,        0., -1000000., -1000000., -1000000.],
    #  [       0.,        0., -1000000., -1000000., -1000000.]]
    #
    # this is not the fastest method out there, but hey gets the job done.
    m = torch.zeros((len(idx), idx.shape[1], idx.shape[1]))
    for _i,t in enumerate(idx):
      if P_id in t[1:]:
        l_ = (t[1:] == P_id).long().argmax(-1)
        l_ = min(l_ + 1, t.shape[0])
        m[_i, l_:, l_:] = -1e6

    if targets is not None:
      assert isinstance(targets, (list, tuple)), \
        "target needs to have a LongTensor and a list/tuple of attn for MSE"

      targets, attn_masks = targets

      # convert the target to proper tokens
      if not isinstance(targets, torch.Tensor):
        targets = tokens(targets)
      if len(targets.shape) == 1:
        targets = targets.unsqueeze(0)

      # verify the shapes of attention masks
      assert len(attn_masks) == len(self.blocks), \
        "Number of attentions should be same as number of blocks. " +\
        f"{len(attn_masks)} != {len(self.blocks)}"
      assert isinstance(attn_masks[0], (list, tuple, torch.Tensor)), \
        "Each sequence in the attention should be a tuple/list/tensor"

      for i,(b,a) in enumerate(zip(self.blocks, attn_masks)):
        assert isinstance(a[0], torch.Tensor)
        assert len(a) == b.n_head, \
          f"Number of attn != number of heads in a block. Got: {len(a), b.n_head}"
        attn_masks[i] = torch.cat([aa.float().unsqueeze(0) for aa in a], 0)

      # for i,a in enumerate(attn_masks):
      #   print(":--:", i, a.shape, self.blocks[i].n_head)

      # convert to the final tuple
      targets = (targets, attn_masks)

    return idx, m, targets

  def forward(self, idx, targets=None, output_dict = False):
    """
    Args:
      idx: Can take in following objects:
        - string
        - list[string]
        - torch.LongTensor (1D)
        - torch.LongTensor (2D)
      targets (optional): Since in rasp you calculate losses for attention matrix
        as well, this targets is a list:
          - torch.LongTensor(): with the cross entropy for entire input tokens,
            just like a normal transformer (GPT/BERT)
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
    position_embeddings = self.pos_emb[:, :t] # each position maps to a (learnable) vector
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
        t = torch.tile(t, [a.shape[0], 1, 1]) # proper batch-ise
        # print(a.shape, t.shape) # [b, s, s]

        mse_loss += F.mse_loss(a, t)
      loss = ce_loss + mse_loss

    if not output_dict:
      return logits, loss
    return Response(logits, loss, all_attn)

# ------ model function ------ #

def get_model(**kwargs):
  config = Config(**kwargs)
  return FullTransformer(config)
