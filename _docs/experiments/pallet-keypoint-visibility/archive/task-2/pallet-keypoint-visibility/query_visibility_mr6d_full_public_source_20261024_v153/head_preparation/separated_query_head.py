"""Small query head with separate visibility, entity and coarse point quality.

Prototype only: no checkpoint, data loader, coordinate prediction or calibration.
The original visibility computation/state keys are preserved for source trials.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import torch
from torch.nn import functional as F

H = Path(__file__).resolve().parent
P = H.parents[1]
PARENT = P / 'query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py'
PARENT_SHA = 'bce3255068fddc91400c8c575060989ca927fd70d68ff8be1928dee75cf159b5'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
nodes = [n for n in ast.parse(PARENT.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GeometryQueryHead']
assert len(nodes) == 1
scope = {'torch': torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARENT), 'exec'), scope)
OriginalHead = scope['GeometryQueryHead']


class SeparatedQueryHead(OriginalHead):
    def __init__(self):
        super().__init__()
        self.entity = torch.nn.Linear(64, 1)
        self.point_quality = torch.nn.Linear(64, 1)
        for layer in (self.entity, self.point_quality):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def load_visibility_state(self, state):
        expected = set(OriginalHead().state_dict())
        if set(state) != expected:
            raise ValueError('Expected original visibility head state keys only')
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert set(missing) == {'entity.weight', 'entity.bias', 'point_quality.weight', 'point_quality.bias'}
        assert not unexpected

    def forward(self, features, geometry42):
        assert features.ndim == geometry42.ndim == 2
        assert features.shape[1] == 1928 and geometry42.shape == (len(features), 42)
        assert features.dtype == geometry42.dtype == torch.float32
        T, Q = features[:, :384], features[:, 384:768]
        C, Pv = features[:, 1152:1536], features[:, 1536:1920]
        visual = torch.cat((features, torch.abs(T-C), T*C, torch.abs(Q-Pv), Q*Pv), 1)
        hidden = self.head[1](self.head[0](visual) + self.geometry(geometry42))
        return dict(external_occlusion=self.head[2](hidden).squeeze(-1),
            target_entity=self.entity(hidden).squeeze(-1),
            coarse_point_quality=self.point_quality(hidden).squeeze(-1))


def per_output_loss(outputs, labels, available, training_rows):
    """Independent loss masks; unknown quality never filters visibility.

    Labels are -1 unknown or binary. Class balancing is within each known output.
    The future experiment must freeze task weights/sampling before actual fit.
    """
    result = {}
    for key, logits in outputs.items():
        target = labels[key]
        eligible = available & training_rows & ((target == 0) | (target == 1))
        losses = [F.binary_cross_entropy_with_logits(logits[eligible & (target == cls)],
                   target[eligible & (target == cls)])
                  for cls in (0, 1) if (eligible & (target == cls)).any()]
        result[key] = sum(losses) / len(losses) if losses else logits.sum() * 0
    return result


def generated_checks():
    torch.set_num_threads(1)
    torch.manual_seed(1530)
    base = OriginalHead().float()
    prototype = SeparatedQueryHead().float()
    prototype.load_visibility_state(copy.deepcopy(base.state_dict()))
    x, G = torch.randn(12, 1928), torch.zeros(12, 42)
    original, output = base(x, G), prototype(x, G)
    assert torch.equal(original, output['external_occlusion'])
    assert not output['target_entity'].any() and not output['coarse_point_quality'].any()
    assert sum(t.numel() for t in prototype.parameters()) == 224643
    # Visibility loss with auxiliary weight0 has identical original gradients.
    labels = {key: torch.full((12,), -1.) for key in output}
    labels['external_occlusion'][:4] = torch.tensor([0., 1., 0., 1.])
    labels['target_entity'][4:8] = torch.tensor([0., 1., 0., 1.])
    labels['coarse_point_quality'][8:] = torch.tensor([0., 1., 0., 1.])
    available, train = torch.ones(12, dtype=torch.bool), torch.ones(12, dtype=torch.bool)
    terms = per_output_loss(output, labels, available, train)
    (terms['external_occlusion'] + 0 * terms['target_entity'] + 0 * terms['coarse_point_quality']).backward()
    F.binary_cross_entropy_with_logits(original[:4], labels['external_occlusion'][:4]).backward()
    for name, param in base.named_parameters():
        assert torch.equal(param.grad, dict(prototype.named_parameters())[name].grad)
    assert not prototype.entity.weight.grad.any() and not prototype.point_quality.weight.grad.any()
    # Verify raw output gradients for unknown, unavailable and held-validation rows.
    raw = {key: torch.randn(12, requires_grad=True) for key in labels}
    available[2] = False
    train[3] = False
    losses = per_output_loss(raw, labels, available, train)
    sum(losses.values()).backward()
    for key, value in raw.items():
        eligible = available & train & (labels[key] >= 0)
        assert not value.grad[~eligible].any()
        assert value.grad[eligible].abs().min() > 0
    # No valid labels still preserves a differentiable zero with exactly zero grad.
    z = {key: torch.randn(3, requires_grad=True) for key in labels}
    empty = {key: torch.full((3,), -1.) for key in labels}
    sum(per_output_loss(z, empty, torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool)).values()).backward()
    assert all(not value.grad.any() for value in z.values())
    report = dict(completed=True, prototype_only=True,
        code=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        parent_code=dict(path=str(PARENT), sha256=PARENT_SHA), parameters=224643,
        visibility_random_state_output_bitexact_original=True,
        original_visibility_gradients_bitexact_when_auxiliary_weights_zero=True,
        per_output_unknown_unavailable_and_validation_gradient_exact_zero=True,
        neutral_new_auxiliary_initial_logits=0, parent_checkpoint_or_actual_features_read=False,
        polarities=dict(external_occlusion='E positive / D negative', target_entity='supported target positive / wrong target negative',
            coarse_point_quality='synthetic same-native error<=3 positive / >20 negative; middle unknown'),
        inference_fusion_or_abstention_policy='NOT_DEFINED_OR_CALIBRATED; future source protocol must freeze before fit/evaluation',
        real_coordinate_accuracy_gold_created=0, actual_training_updates=0,
        private_data_or_PnP_or_coordinate_correction=0)
    path = H / 'SEPARATED_HEAD_PROTOTYPE_CHECKS.json'
    with path.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())))


if __name__ == '__main__':
    generated_checks()
