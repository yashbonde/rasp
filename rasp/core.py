# Since building simple primitives is the primary task of this language,
# training the model should have first class support.
# now you can directly load a primitive as follows:
#
# >>> from rasp import Primitive
# >>> reverse = Primitive("reverse")
# >>> reverse("hey")
# ... "yeh"

from rasp.model import *
from rasp.manual import ivocab, vocab, tokens

class Trainer:
  # base object to train primitives
  def __init__(self):
    pass


class Primitive:
  # primtive class is a Transformer neural network whose objective
  # is to perform that particular task.
  def __init__(self, code = None, **kwargs):
    if code is not None:
      raise NotImplementedError("code parsing is still not implemented, hold your horses!")

    config = Config(**kwargs)
    self.model = FullTransformer(config)

  def get_parameters(self):
    return self.model.parameters()

  def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)



def get_vocab():
  return vocab, ivocab

if __name__ == "__main__":
  import os, sys
  import random
  import torch

  import numpy as np
  from rasp.daily import folder
  sys.path.append(os.path.join(folder(folder(__file__)), "primitives"))
  from primitives import functional as F

  # check if the loading is working
  # print(F.identity("foo"))
  vocab, ivocab = get_vocab()

  def identity_dataset(n = 200, m = 32):
    # since our manual primitives take care of the input output
    # we can batch the dataset into buckets of similar lengths
    ds = []
    for _ in range(n): # generate samples
      x = "".join([
        ivocab[_i] for _i in np.random.randint(0, len(vocab) - 1, size = (np.random.randint(m) + 1,))
      ])
      ds.append(x)

    m = max([len(x) for x in ds])
    for i,s in enumerate(ds):
      s = s[:m]
      if np.random.random() > 0.6:
        _i = np.random.randint(len(s))
        _j = _i + np.random.randint(5)
        _v = ivocab[np.random.randint(25)]
        s = s[_i] + "".join([_v for _ in range(_i, _j, 1)]) + s[_j:]
      s = s  + "".join(["$" for _ in range(m - len(s))])
      ds[i] = s[:m]
    return ds

  ds = identity_dataset()
  p = Primitive()
  print("Test 1D:", tokens(p(ds[0])[0].argmax(-1)))
  print("Test (batch):", tokens(p(ds[:2])[0].argmax(-1)))

  # optim = torch.optim.Adam(p.get_parameters())

  # from tqdm import trange
  # pbar = trange(1000)



