import unittest
from rasp.model import *

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
    

if __name__ == '__main__':
  unittest.main()
