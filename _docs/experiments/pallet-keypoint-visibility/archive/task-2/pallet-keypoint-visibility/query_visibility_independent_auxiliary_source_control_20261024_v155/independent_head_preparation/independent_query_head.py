"""Source-control head: original visibility and two disjoint auxiliary branches.

This file loads only the pinned original class's source AST. It reads no data,
checkpoint or feature bodies and does not instantiate a head at import time.
"""
from pathlib import Path
import ast
import hashlib
import torch
from torch.nn import functional as F

P = Path(__file__).resolve().parents[2]
ORIGINAL_SOURCE = P / 'query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py'
ORIGINAL_SHA = 'bce3255068fddc91400c8c575060989ca927fd70d68ff8be1928dee75cf159b5'
assert hashlib.sha256(ORIGINAL_SOURCE.read_bytes()).hexdigest() == ORIGINAL_SHA
_nodes = [n for n in ast.parse(ORIGINAL_SOURCE.read_text()).body
          if isinstance(n, ast.ClassDef) and n.name == 'GeometryQueryHead']
assert len(_nodes) == 1
_scope = {'torch': torch}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), str(ORIGINAL_SOURCE), 'exec'), _scope)
OriginalHead = _scope['GeometryQueryHead']


class IndependentAuxiliaryQueryHead(OriginalHead):
    """224513 original visibility +24705 entity +221825 point quality params.

    The common frozen input is not a shared trainable hidden representation.
    Use one AdamW/clip5 for visibility_parameters and a separate AdamW/clip5
    for auxiliary_parameters. A global clip/optimizer would change that control.
    """
    def __init__(self):
        super().__init__()
        self.entity_head = torch.nn.Sequential(
            torch.nn.Linear(384, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        self.quality_head = torch.nn.Sequential(
            torch.nn.Linear(3464, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        for last in (self.entity_head[2], self.quality_head[2]):
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)

    def load_visibility_state(self, state):
        """Load exactly original C32-style visibility keys; leave aux untouched."""
        auxiliary_keys = {k for k in self.state_dict()
                          if k.startswith(('entity_head.', 'quality_head.'))}
        expected = set(self.state_dict()) - auxiliary_keys
        if set(state) != expected:
            raise ValueError('Expected original visibility state keys only')
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert set(missing) == auxiliary_keys and not unexpected

    def visibility_parameters(self):
        return list(self.head.parameters()) + list(self.geometry.parameters())

    def entity_parameters(self):
        return list(self.entity_head.parameters())

    def quality_parameters(self):
        return list(self.quality_head.parameters())

    def auxiliary_parameters(self):
        return self.entity_parameters() + self.quality_parameters()

    def forward(self, features, geometry42):
        # The parent's exact original expression retains head/geometry state keys.
        visibility = OriginalHead.forward(self, features, geometry42)
        T, Q = features[:, :384], features[:, 384:768]
        C, Pv = features[:, 1152:1536], features[:, 1536:1920]
        visual = torch.cat((features, torch.abs(T-C), T*C,
                            torch.abs(Q-Pv), Q*Pv), 1)
        return dict(external_occlusion=visibility,
                    target_entity=self.entity_head(T).squeeze(-1),
                    coarse_point_quality=self.quality_head(visual).squeeze(-1))


def per_output_loss(outputs, labels, available, training_rows):
    """Independent known-label masks; real quality unknown never filters D/E."""
    losses = {}
    for key, logits in outputs.items():
        target = labels[key]
        known = available & training_rows & ((target == 0) | (target == 1))
        parts = [F.binary_cross_entropy_with_logits(logits[known & (target == cls)],
                                                   target[known & (target == cls)])
                 for cls in (0, 1) if (known & (target == cls)).any()]
        losses[key] = sum(parts) / len(parts) if parts else logits.sum() * 0
    return losses
