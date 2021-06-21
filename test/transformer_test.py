import unittest
import numpy as np
from rasp.model import *
from rasp.manual import tokens

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# assertion values

# for numpy and tensor
FIRST_PASS_TARGET_TENSOR = [[13, 23, 3, 10, 14, 13], [18, 5, 25, 3, 4, 10]]
FIRST_PASS_LOSS_TENSOR = 4.4479
SECOND_PASS_TARGET_TENSOR = [[19, 23, 3, 10, 14, 13], [18, 5, 25, 14, 4, 10]]
SECOND_PASS_LOSS_TENSOR = 4.3977

# for string computation
STRING_PREDICTION = ['fme']
FIRST_PASS_LOSS_STRING = 4.5970
SECOND_PASS_LOSS_STRING = 4.5050

# test class

class TestTransformer(unittest.TestCase):

  # test if the model is even initialized correctly
  def test_initialize(self):
    config = TinyConfig()
    model = FullTransformer(config)
    self.assertEqual(model.num_parameters, 5706)

  # forward + backward testing with tensors
  def test_forward(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    x = torch.randint(0, config.vocab_size, size = (2, 6))
    logits, loss = model(x)
    self.assertEqual( logits.argmax(-1).tolist(), FIRST_PASS_TARGET_TENSOR )
    self.assertEqual(loss, None)

  def test_forward_with_loss(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    x = torch.randint(0, config.vocab_size, size = (2, 6))
    target = torch.randint(0, config.vocab_size, size = (2, 6))
    target = (target, [[torch.randn(6, 6)]])
    
    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).tolist(), FIRST_PASS_TARGET_TENSOR )
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_TENSOR)
    self.assertTrue(out)

  def test_backward(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    optim = torch.optim.Adam(model.parameters())

    x = torch.randint(0, config.vocab_size, size = (2, 6))
    target = torch.randint(0, config.vocab_size, size = (2, 6))
    target = (target, [[torch.randn(6, 6)]])

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).tolist(), FIRST_PASS_TARGET_TENSOR )
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_TENSOR)
    self.assertTrue(out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).tolist(), SECOND_PASS_TARGET_TENSOR)
    out = np.isclose(loss.item(), SECOND_PASS_LOSS_TENSOR)
    self.assertTrue(out)

  # forward + backward testing with strings
  def test_forward_str(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    x = "hey"
    logits, loss = model(x)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    self.assertEqual(loss, None)

  def test_forward_with_loss_str(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    x = "hey"; target = "hey"
    target = (target, [[torch.randn(3, 3)]])

    logits, loss = model(x, target)
    
    p = [tokens(x) for x in logits.argmax(-1)]
    
    self.assertEqual(p, STRING_PREDICTION)
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_STRING)
    self.assertTrue(out)

  def test_backward_str(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    optim = torch.optim.Adam(model.parameters())

    # first pass
    x = "hey"; target = "hey"
    target = (target, [[torch.randn(3, 3)]])

    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_STRING)
    self.assertTrue(out)

    # backprop
    optim.zero_grad()
    loss.backward()
    optim.step()

    # second pass
    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    out = np.isclose(loss.item(), SECOND_PASS_LOSS_STRING)
    self.assertTrue(out)

  # NOTE: this assumes that the entire model resides on a single card, ie. there is
  # no model distributed.
  
  # forward + backward testing with tensors CUDA
  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_initialize_cuda(self):
    # test if the model is even initialized correctly
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    self.assertEqual(model.num_parameters, 5706)
    del model

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_cuda(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    logits, loss = model(x)
    self.assertEqual( logits.argmax(-1).detach().cpu().tolist(), FIRST_PASS_TARGET_TENSOR )
    self.assertEqual(loss, None)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_with_loss_cuda(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = (target, [[torch.randn(6, 6).cuda()]])

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).detach().cpu().tolist(), FIRST_PASS_TARGET_TENSOR )
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_TENSOR)
    self.assertTrue(out)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_backward_cuda(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    optim = torch.optim.Adam(model.parameters())

    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = (target, [[torch.randn(6, 6).cuda()]])

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).detach().cpu().tolist(), FIRST_PASS_TARGET_TENSOR )
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_TENSOR)
    self.assertTrue(out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).detach().cpu().tolist(), SECOND_PASS_TARGET_TENSOR)
    out = np.isclose(loss.item(), SECOND_PASS_LOSS_TENSOR)
    self.assertTrue(out)

  # forward + backward testing with strings CUDA
  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_str_cuda(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    x = "hey"
    logits, loss = model(x)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    self.assertEqual(loss, None)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_with_loss_str_cuda(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    x = "hey"; target = "hey"
    target = (target, [[torch.randn(3, 3).cuda()]])

    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_STRING)
    self.assertTrue(out)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_backward_str_cuda(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config).cuda()
    optim = torch.optim.Adam(model.parameters())

    # first pass
    x = "hey"; target = "hey"
    target = (target, [[torch.randn(3, 3).cuda()]])

    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    out = np.isclose(loss.item(), FIRST_PASS_LOSS_STRING)
    self.assertTrue(out)

    # backprop
    optim.zero_grad()
    loss.backward()
    optim.step()

    # second pass
    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, STRING_PREDICTION)
    out = np.isclose(loss.item(), SECOND_PASS_LOSS_STRING)
    self.assertTrue(out)

if __name__ == "__main__":
  unittest.main()
