# models/PredictionHead.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# FNN 模型 64 -> 128 -> 64 -> 1
class PredictionHead_FNNMini(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, plan_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(plan_emb)


class ResBlock(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, d)
        self.fc2  = nn.Linear(d, d)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        # Use simple addition for residual connection without norm first to debug
        # Or ensure x shape is compatible
        residual = x
        out = self.norm(x)
        out = F.gelu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        return residual + out

class PredictionHead_V2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dims[0])
        self.blocks = nn.ModuleList([ResBlock(hidden_dims[i], dropout) for i in range(len(hidden_dims))])
        self.norm = nn.LayerNorm(hidden_dims[-1])
        self.out = nn.Linear(hidden_dims[-1], out_dim)
    def forward(self, plan_emb: torch.Tensor) -> torch.Tensor:
        # ResBlock expects input of shape [*, hidden_dim]
        # Input plan_emb might be [batch_size, in_dim]
        # We first project it to hidden_dim
        h = F.gelu(self.inp(plan_emb))
        
        # Apply LayerNorm carefully
        # h shape is [batch_size, hidden_dim]
        # ResBlock.norm is LayerNorm(hidden_dim)
        for b in self.blocks:
            h = b(h)
            
        # Final norm before output
        h = self.norm(h)
        return self.out(h)