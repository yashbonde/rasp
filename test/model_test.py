import unittest
from rasp.model import *
from rasp.core import Primitive, get_vocab

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TestTransformer(unittest.TestCase):

  def test_model_string_input(self):
    set_seed(4)
    model = get_model()
    out, loss = model("hello")
    self.assertEqual(tokens(out.argmax(-1)), ['etffn'])
    out, loss = model(["hello", "world"])
    self.assertEqual(tokens(out.argmax(-1)), ['etffn', 'zkofk'])
    out, loss = model(["hello", "wd", "sdfg"])
    self.assertEqual(tokens(out.argmax(-1)), ['etffn', 'zkxxx', 'jkkdx'])

  def test_train(self):
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
      set_seed(4)
      ds = []
      for _ in range(n): # generate samples
        x = "".join([
          ivocab[_i] for _i in np.random.randint(0, len(vocab) - 1, size = (np.random.randint(m) + 1,))
        ])
        ds.append(x)

      # create the dataset
      m = max([len(x) for x in ds])
      for i,s in enumerate(ds):
        s = s[:m]
        if np.random.random() > 0.6:
          _i = np.random.randint(len(s))
          _j = _i + np.random.randint(5)
          _v = ivocab[np.random.randint(25)]
          s = s[_i] + "".join([_v for _ in range(_i, _j, 1)]) + s[_j:]
        ds[i] = s[:m]
      return ds

    # create dataset
    ds = identity_dataset()

    # define the primitive
    p = Primitive("identity")
    # print("Test 1D:", tokens(p(ds[0])[0].argmax(-1)))
    # print("Test (batch):", tokens(p(ds[:2])[0].argmax(-1)))

    # train the network things
    p.train(ds, F.identity)

if __name__ == '__main__':
  unittest.main()
