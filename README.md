# rasp

Implementing Restricted Access Sequence Processing (RASP) transformer language from ["Thinking like Transformers"](https://arxiv.org/pdf/2106.06981.pdf) paper. From the paper:

> RASP can be used to program solutions to tasks that could conceivably be learned by a Transformer, and how a Transformer can be trained to mimic a RASP solution.

## Installation

Simple pip install. `pip install git+https://github.com/yashbonde/rasp`

### What is RASP?

- `select` (creating selection matrices called selectors): This is similar to attention creation `k x q.T => attn`.
- `aggregate`  (collapsing selectors and s-ops into a new s-ops): This is like value multiplication `attn x v`.
- `selector_width` (creating an s-op from a selector): returns number of activated tensors `selector_width(select(tokens,tokens,==))("hello")=[1,1,2,2,1]`.

### Structure

- `rasp/`: main code library
- `test/`: tests
- `primitives`: neural functions and primitives built using manual codes that can be used for training

## Code

### Example: Reverse [WIP]

So you can built complex flows directly in terms of architecture ex. building reverse function:
```python
from rasp import RaspModule

reverse = RaspModule('''
def reverse(tokens):
  opp_idx = length - indices - 1;
  flip = select (indices ,opp_index ,==) ;
  return aggregate (flip, tokens);
''')
assert reverse("hey") == "yeh"
```

This would create a neural network as follows:
```python
class Flip(nn.Module):
  # flip = select (indices ,opp_index ,==) ;
  def __init__(self):
    self.n_head = 1;

  def forward(self):
    pass
```

### Tests

All the code for tests are given in `test/`, run `pytest -v`.
## Experiments

1. Reverse e.g.: `reverse("abc")="cba"`
2. Histograms, with a unique beginning-of-sequence (BOS) token `$` (e.g., `hist_bos("$aba")=[$,2,1,2]`) and without it (e.g., `hist_nobos("aba")=[2,1,2]`)
3. Double-Histograms, with BOS: for each token, the number of unique tokens with same histogram value as itself. E.g.: `hist2("$abbc")=[§,2,1,1,2]`
4. Sort, with BOS: ordering the input tokens lexicographically. e.g.: `sort("$cba")="$abc".`
5. Most-Freq, with BOS: returning the unique input tokens in order of decreasing frequency, with original position as a tie-breaker and the BOS token for padding. E.g.: `most_freq("$abbccddd")="$dbca$$$$"`
6. Dyck-i PTF, for `i = 1, 2`: the task of returning, at each output position, whether the input prefix up to and including that position is a legal Dyck-i sequence (`T`), and if not, whether it can (`P`) or cannot (`F`) be continued into a legal Dyck-i sequence. E.g: `Dyck1_ptf("()())")="PTPTF"`

### References:

- [Tokens List](https://github.com/tech-srl/RASP/blob/main/RASP_support/zzantlr/RASP.tokens)
- [RASP Cheatsheet](https://github.com/tech-srl/RASP/blob/main/cheat_sheet.pdf)
