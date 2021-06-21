# Since building simple primitives is the primary task of this language,
# training the model should have first class support.
# now you can directly load a primitive as follows:
#
# >>> from rasp import Primitive
# >>> reverse = Primitive("reverse")
# >>> reverse("hey")
# ... "yeh"

import json
import numpy as np
from tqdm import trange

from rasp.model import *
from rasp.manual import ivocab, vocab, tokens
from rasp.daily import Hashlib

def set_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Primitive:
  # primtive class is a Transformer neural network whose objective
  # is to perform that particular task.
  def __init__(self, name, code = None, seed = 4, **model_kwargs):
    set_seed(4)
    if code is not None:
      raise NotImplementedError("code parsing is still not implemented, hold your horses!")

    self.model = get_model(**model_kwargs)
    self.name = name

    str_ = f"{name}-" + json.dumps(self.model.config.get_json())
    self._hash = Hashlib.sha256(str_)

  def get_parameters(self):
    return self.model.parameters()

  def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)

  def viz(self, x):
    # this is not the best visualisation of attention since the values are
    # in float. But this is good enough to see what's up
    print("-+-" + "-" * len(x) * 2)
    print(" | " + " ".join(x))
    print("-+-" + "-" * len(x) * 2)
    r = self(x, output_dict = True)
    a = r.attns[0] * 10
    a = a.long()
    a = a.tolist()[0]
    for i in range(len(a)):
      print(f"{x[i]}|", " ".join([str(b) for b in a[i]]))
    print("-+-" + "-" * len(x) * 2)

  def train(self, ds, man_fn, optim_name = "Adam", n_epochs = 5, pbar = False, **optimiser_params):
    """training any primitive has first class support since this is what each primitive is"""
    optim = getattr(torch.optim, optim_name)(self.get_parameters(), **optimiser_params)
    for i in range(n_epochs):
      bar = trange(len(ds)) if pbar else range(len(ds))
      for x, j in zip(ds, bar):
        t = man_fn(x)
        out, loss = self(idx = x, targets = t)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        if j % 50 == 0:
          print(loss)
    self.viz(ds[0])

def get_vocab():
  return vocab, ivocab
