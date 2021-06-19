# this file has the support for loss functions
# This particular transformer was trained using both target and attention supervision,
# - the standard cross entropy loss on the target output,
# - An MSE-loss between its attention heatmaps and those expected by the RASP solution

import rasp.manual as man

def identity(tokens) -> tuple:
  identity_selector = man.select(tokens, tokens, "==")
  return identity_selector.float()
