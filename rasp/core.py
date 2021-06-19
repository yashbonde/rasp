from rasp import manual as man
from rasp.transformer.model import *

'''How this works

```
aggregate(indentity, [9, 12, 15])
```

# during init
_m1, _ = indentity()
_m2, _ = aggregate()

# during runtime
_m2(_m1([9, 12, 15])) == [3, 4, 5]

'''

def identity(cache_location = ".rasp-cache", **arch_kwargs):
  def identity_loss_fn(tokens):
    attn = man.select(man.indices(tokens), man.indices(tokens), "==")
    return [[attn]] # shape: [number of blocks, number of heads in each block]

  config = TinyConfig()
  for k, v in arch_kwargs.items():
    setattr(config, k, v)
  
  # create the model and return it
  model = FullTransformer(config)

  # load the model from `cache_location`

  return model, identity_loss_fn
