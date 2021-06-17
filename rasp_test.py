import unittest
import numpy as np
from rasp import *

F = False
T = True

def set_seed(seed):
  if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestPrimaryOps(unittest.TestCase):

  def test_tokens(self):
    self.assertEqual(tokens("hello").tolist(), [ 7.,  4., 11., 11., 14.])
    self.assertEqual(tokens(tokens("hey")), "hey")

  def test_logical_1d(self):
    x = torch.Tensor([F, F, T])
    y = torch.Tensor([F, T, T])
    self.assertEqual(logical(x, "not", y).tolist(), [T, T, F])
    self.assertEqual(logical(x, "or", y).tolist(), [F, T, T])
    self.assertEqual(logical(x, "and", y).tolist(), [F, F, T])
    self.assertEqual(logical(x, "xor", y).tolist(), [F, T, F])

  def test_logical_2d(self):
    x = torch.Tensor([[F, F, T], [T, T, F], [T, F, T]]).bool()
    y = torch.Tensor([[T, F, F], [T, T, F], [T, T, T]]).bool()

    self.assertEqual(logical(x, "not", y).tolist(), [[T, T, F], [F, F, T], [F, T, F]])
    self.assertEqual(logical(x, "or", y).tolist(),  [[T, F, T], [T, T, F], [T, T, T]])
    self.assertEqual(logical(x, "and", y).tolist(), [[F, F, F], [T, T, F], [T, F, T]])
    self.assertEqual(logical(x, "xor", y).tolist(), [[T, F, T], [F, F, F], [F, T, F]])

  def test_elementwise(self):
    x = torch.Tensor([[0, 0, 2], [2, 2, 0], [2, 0, 2]])
    y = torch.Tensor([[3, 0, 0], [3, 3, 0], [3, 3, 3]])

    self.assertEqual(elementwise(x, "+", y).tolist(), [[ 3, 0, 2], [ 5,  5, 0], [ 5,  3,  5]])
    self.assertEqual(elementwise(x, "-", y).tolist(), [[-3, 0, 2], [-1, -1, 0], [-1, -3, -1]])
    self.assertEqual(elementwise(x, "*", y).tolist(), [[ 0, 0, 0], [ 6,  6, 0], [ 6,  0,  6]])
    self.assertTrue(np.allclose(
      np.array(elementwise(x, "/", y).tolist()).reshape(-1),
      np.array([[ 0, 0, 0], [2/3,2/3,0], [2/3, 0, 2/3]]).reshape(-1)
    ))

  def test_select(self):
    x = torch.Tensor([1, 2, 2])
    y = torch.Tensor([0, 1, 2])
    s = select(x, y, "==")
    self.assertEqual(s.tolist(), [[F, F, F], [T, F, F], [F, T, T]])

  def test_aggregate(self):
    x = torch.Tensor([4, 6, 8])
    s = torch.Tensor([[F, F, F], [T, F, F], [F, T, T]]).bool()
    self.assertEqual(aggregate(s, x).tolist(), [0, 4, 7])

  def test_indices(self):
    self.assertEqual(indices("hi").tolist(), [0, 1])

  def test_length(self):
    self.assertEqual(length("yoo").tolist(), [3, 3, 3])
    self.assertEqual(length("hi").tolist(), [2, 2])

  def test_flip(self):
    self.assertEqual(flip("hey").tolist(), [[F, F, T], [F, T, F], [T, F, F]])

class PaperBuiltOps(unittest.TestCase):
  
  def test_reverse(self):
    x = "hey"
    reverse = tokens(aggregate(flip(x), tokens(x)))
    self.assertEqual(reverse, "yeh")

  def test_select_1(self):
    i = indices("hey")
    out = select(i, i, "<").tolist()
    self.assertEqual(out, [[F,F,F],[T,F,F],[T,T,F]])

  def test_select_aggregate_1(self):
    def a(x):
      return select(indices(x), indices(x), "<")
    i = indices("hey"); a = a("hey")
    aggregate(a, i + 1, "mean")

  def  test_select_aggregate_2(self):
    def load1(x):
      return select(indices(x), 1, "==")
    out = logical(load1("hey"), "or", flip("hey")).tolist()
    self.assertEqual(out, [[F,T,T],[F,T,F],[T,T,F]])


if __name__ == "__main__":
  unittest.main()
