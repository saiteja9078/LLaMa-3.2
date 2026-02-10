import torch
import torch.nn as nn
#For text models
class RMSNorm(nn.Module):
    def __init__(self,dim, eps = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim,dtype=torch.float32),requires_grad=True)
        self.eps = eps
    def forward(self, x: torch.Tensor):
        _,_,d = x.shape
        y = x
        x = x**2
        den = torch.sqrt(
            self.eps + (x.sum(dim=-1,keepdim=True)/d)
        )
        y = y / den
        return self.scale * y 