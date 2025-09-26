import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.device: torch.device | None = device
        self.dtype : torch.dtype | None = dtype
        
        # Initialize weights to 1
        self.weight = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Prevent overflow in mean/sqrt calculations
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute the mean square value along the last dimension
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        
        # Compute the root mean square (RMS) value
        rms = torch.sqrt(mean_square + self.eps)
        
        # Normalize the input tensor
        normalized_x = x / rms
        
        return normalized_x * self.weight