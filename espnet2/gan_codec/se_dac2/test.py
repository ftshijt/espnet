import torch
from se_dac import SEDAC
c = torch.randn(1, 1, 20160)
model = SEDAC()
b = model.forward(c)
