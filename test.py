
import torch
from transformer._utils import make_causal_mask


a = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 8]])
print(a)
mask = make_causal_mask(a, prefix_len=2)
print(mask)

