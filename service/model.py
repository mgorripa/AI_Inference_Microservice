#tiny PyTorch MLP
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# Instantiate and set to evaluation mode
_model = TinyMLP()
_model.eval()


def predict(x: torch.Tensor) -> torch.Tensor:
    """Run inference on input tensor using TinyMLP."""
    with torch.no_grad():
        return _model(x)
