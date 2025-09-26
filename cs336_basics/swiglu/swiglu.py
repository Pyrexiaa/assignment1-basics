import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x: torch.tensor): return 1 / (1 + torch.exp(-x)) # sigmoid activation function

def silu(x: torch.tensor): return x * torch.sigmoid(x) # SiLU activation function

def glu(a:torch.tensor, b:torch.tensor): return a * b # element-wise multiplication

def swiglu_fn(a: torch.tensor, b: torch.tensor): return glu(silu(a), b) # SwiGLU activation function

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Projections: one for content
        self.w1 = nn.Linear(d_model, d_ff, **factory_kwargs, bias=False)
        # Output projection back to model dimension
        self.w2 = nn.Linear(d_ff, d_model, **factory_kwargs, bias=False)
        # Projections: one for gate
        self.w3 = nn.Linear(d_model, d_ff, **factory_kwargs, bias=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Content branch
        content1 = self.w1(x)
        content2 = self.w3(x)
        # Gate branch with SwiGLU activation
        gate_result = swiglu_fn(content1, content2)
        # Final projection back
        return self.w2(gate_result)