import unittest
import numpy as np
from rasp.transformer.model import *

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestTransformer(unittest.TestCase):

  def test_initialize(self):
    # test if the model is even initialized correctly
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    self.assertEqual(model.num_parameters, 10368)

  def test_forward(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    x = torch.randint(0, config.vocab_size, size = (2, 6))
    logits, loss = model(x)
    self.assertEqual(
      logits.argmax(-1).tolist(),
      [[ 26,  38,  18, 104, 124, 116],
       [ 70,  13,  60,  80, 104,  62]]
    )
    self.assertEqual(loss, None)

  def test_forward_with_loss(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    x = torch.randint(0, config.vocab_size, size = (2, 6))
    target = torch.randint(0, config.vocab_size, size = (2, 6))
    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).tolist(),
      [[ 26,  38,  18, 104, 124, 116],
       [ 70,  13,  60,  80, 104,  62]]
    )
    out = np.isclose(loss.item(), 4.84674)
    self.assertTrue(out)

  def test_backward(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    optim = torch.optim.Adam(model.parameters())

    x = torch.randint(0, config.vocab_size, size = (2, 6))
    target = torch.randint(0, config.vocab_size, size = (2, 6))
    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).tolist(),
      [[ 26,  38,  18, 104, 124, 116],
       [ 70,  13,  60,  80, 104,  62]]
    )
    out = np.isclose(loss.item(), 4.84674)
    self.assertTrue(out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).tolist(),
      [[9, 60, 18, 22, 5, 106], [94, 13, 60, 80, 104, 46]]
    )
    out = np.isclose(loss.item(), 4.7731)
    self.assertTrue(out)

  # NOTE: this assumes that the entire model resides on a single card, ie. there is
  # no model distributed.

  @unittest.skipIf(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_initialize(self):
    # test if the model is even initialized correctly
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    self.assertEqual(model.num_parameters, 10368)
    del model

  @unittest.skipIf(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    logits, loss = model(x)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[ 26,  38,  18, 104, 124, 116],
       [ 70,  13,  60,  80, 104,  62]]
    )
    self.assertEqual(loss, None)

  @unittest.skipIf(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_with_loss(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[ 26,  38,  18, 104, 124, 116],
       [ 70,  13,  60,  80, 104,  62]]
    )
    out = np.isclose(loss.item(), 4.84674)
    self.assertTrue(out)

  @unittest.skipIf(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_backward(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    optim = torch.optim.Adam(model.parameters())

    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[ 26,  38,  18, 104, 124, 116],
       [ 70,  13,  60,  80, 104,  62]]
    )
    out = np.isclose(loss.item(), 4.84674)
    self.assertTrue(out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[9, 60, 18, 22, 5, 106], [94, 13, 60, 80, 104, 46]]
    )
    out = np.isclose(loss.item(), 4.7731)
    self.assertTrue(out)
