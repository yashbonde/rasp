import unittest
import numpy as np
from rasp.transformer.model import *
from rasp.manual import tokens

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestTransformer(unittest.TestCase):

  # test if the model is even initialized correctly
  def test_initialize(self):
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    self.assertEqual(model.num_parameters, 9342)

  # forward + backward testing with tensors
  def test_forward(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    x = torch.randint(0, config.vocab_size, size = (2, 6))
    logits, loss = model(x)
    self.assertEqual(
      logits.argmax(-1).tolist(),
      [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]]
    )
    self.assertEqual(loss, None)

  def test_forward_with_loss(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    x = torch.randint(0, config.vocab_size, size = (2, 6))
    target = torch.randint(0, config.vocab_size, size = (2, 6))
    target = (target, [[torch.randn(6, 6)]])
    
    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).tolist(), [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]] )
    out = np.isclose(loss.item(), 5.8412)
    self.assertTrue(out)

  def test_backward(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config)
    optim = torch.optim.Adam(model.parameters())

    x = torch.randint(0, config.vocab_size, size = (2, 6))
    target = torch.randint(0, config.vocab_size, size = (2, 6))
    target = (target, [[torch.randn(6, 6)]])

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).tolist(), [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]] )
    out = np.isclose(loss.item(), 5.8412)
    self.assertTrue(out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    logits, loss = model(x, target)
    self.assertEqual( logits.argmax(-1).tolist(), [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]] )
    out = np.isclose(loss.item(), 5.7863)
    self.assertTrue(out)

  # forward + backward testing with strings
  def test_forward_str(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    x = "hey"
    logits, loss = model(x)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, ['fmo'])
    self.assertEqual(loss, None)

  def test_forward_with_loss_str(self):
    set_seed(4)
    config = TinyConfig()
    model = FullTransformer(config)
    x = "hey"; target = "hey"
    target = (target, [[torch.randn(3, 3)]])

    logits, loss = model(x, target)
    
    p = [tokens(x) for x in logits.argmax(-1)]
    
    self.assertEqual(p, ['fmo'])
    out = np.isclose(loss.item(), 4.6525)
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
    self.assertEqual(p, ['fmo'])
    out = np.isclose(loss.item(), 4.6525)
    self.assertTrue(out)

    # backprop
    optim.zero_grad()
    loss.backward()
    optim.step()

    # second pass
    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, ['fmo'])
    out = np.isclose(loss.item(), 4.5600)
    self.assertTrue(out)


  # forward + backward testing with tensors CUDA

  # NOTE: this assumes that the entire model resides on a single card, ie. there is
  # no model distributed.

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_initialize_cuda(self):
    # test if the model is even initialized correctly
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    self.assertEqual(model.num_parameters, 9342)
    del model

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_cuda(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    logits, loss = model(x)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]]
    )
    self.assertEqual(loss, None)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_forward_with_loss_cuda(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = (target, [[torch.randn(6, 6).cuda()]])

    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]]
    )
    out = np.isclose(loss.item(), 5.8412)
    self.assertTrue(out)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA not found, skipping these tests")
  def test_backward_cuda(self):
    set_seed(4)
    config = TinyConfig()
    config.vocab_size = 128
    model = FullTransformer(config).cuda()
    optim = torch.optim.Adam(model.parameters())

    x = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = torch.randint(0, config.vocab_size, size = (2, 6)).cuda()
    target = (target, [[torch.randn(6, 6).cuda()]])

    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]]
    )
    out = np.isclose(loss.item(), 5.8412)
    self.assertTrue(out)

    optim.zero_grad()
    loss.backward()
    optim.step()

    logits, loss = model(x, target)
    self.assertEqual(
      logits.argmax(-1).detach().cpu().tolist(),
      [[50, 15, 117, 52, 89, 95], [32, 13, 89, 9, 9, 7]]
    )
    out = np.isclose(loss.item(), 5.7863)
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
    self.assertEqual(p, ['fmo'])
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
    self.assertEqual(p, ['fmo'])
    out = np.isclose(loss.item(), 4.6525)
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
    self.assertEqual(p, ['fmo'])
    out = np.isclose(loss.item(), 4.6525)
    self.assertTrue(out)

    # backprop
    optim.zero_grad()
    loss.backward()
    optim.step()

    # second pass
    logits, loss = model(x, target)
    p = [tokens(x) for x in logits.argmax(-1)]
    self.assertEqual(p, ['fmo'])
    out = np.isclose(loss.item(), 4.5600)
    self.assertTrue(out)

if __name__ == "__main__":
  unittest.main()
