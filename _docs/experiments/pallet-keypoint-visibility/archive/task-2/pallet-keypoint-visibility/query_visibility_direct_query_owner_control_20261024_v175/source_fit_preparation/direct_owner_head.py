"""Independent V175 owner branch; no geometry, native slot, q or labels in forward."""
import torch


class DirectOwnerHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(3464, 64), torch.nn.ReLU(),
                                        torch.nn.Linear(64, 3))
        torch.nn.init.zeros_(self.head[2].weight)
        torch.nn.init.zeros_(self.head[2].bias)

    @staticmethod
    def visual(features):
        assert features.ndim == 2 and features.shape[1] == 1928
        assert features.dtype == torch.float32
        T, Q = features[:, :384], features[:, 384:768]
        C, P = features[:, 1152:1536], features[:, 1536:1920]
        return torch.cat((features, (T-C).abs(), T*C, (Q-P).abs(), Q*P), 1)

    def forward(self, features):
        return self.head(self.visual(features))
