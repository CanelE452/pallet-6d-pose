"""V1-identical neural forward with isolated loss balance / PCGrad integration."""
from __future__ import annotations

import torch
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from scripts.research.pallet_dht_joint_v1 import integration as V1
from scripts.research.pallet_dht_joint_v1.line_targets import auxiliary_line_loss
from .gradient_surgery import parameter_dependencies

ARMS = ('balanced', 'pcgrad', 'balanced_pcgrad', 'incidence')
DIAGNOSTIC_STEPS = (2, 16, 256, 1750, 3499, 3500, 5249, 6998)
R0_PATH, R0_SHA256 = V1.R0_PATH, V1.R0_SHA256
snapshot, changes_since = V1.snapshot, V1.changes_since


class AssignedPoseLoss(PoseLoss26):
    """Observe unchanged TAL outputs once; no second assignment or GT alteration."""
    def get_assigned_targets_and_loss(self, predictions, batch):
        result = super().get_assigned_targets_and_loss(predictions, batch)
        fg_mask, target_gt_idx, _, anchor_points, stride_tensor = result[0]
        self.incidence_assignment = (fg_mask, target_gt_idx, anchor_points, stride_tensor)
        return result


class CouplingCriterion(V1.JointCriterion):
    def __init__(self, model):
        super().__init__(model)
        self.arm = model.coupling_arm
        if self.arm == 'incidence':
            self.stock = E2ELoss(model, AssignedPoseLoss)
        self.incidence_weight = float(model.incidence_weight)
        named = [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and (not n.startswith('model.23.') or '.hough.' in n)]
        # EMA parameters have requires_grad=False during validation.
        self.names = tuple(n for n, _ in named)
        self.parameters = tuple(p for _, p in named)
        self.pending = None
        self.capture_gradients = False
        self.next_step = 0
        self.last_scalars = {}
        self.head = model.model[-1]

    def __call__(self, predictions, batch):
        raw = predictions[1] if isinstance(predictions, (tuple, list)) else predictions
        base, items = self.stock(predictions, batch)
        logits = raw['line_logits']
        aux, diagnostics = auxiliary_line_loss(logits, batch['keypoints'], batch['batch_idx'],
            tuple(int(v) for v in batch['img'].shape[-2:]), raw['line_lattice'])
        coefficient = self.weight * (self.stock.o2m / .8) if self.arm in ('balanced', 'balanced_pcgrad') else self.weight
        weighted = aux * coefficient
        incidence = logits.float().sum() * 0.
        if self.arm == 'incidence':
            from .incidence import incidence_loss, decode_points_px
            branch_losses = []
            for branch, coefficient_stock in [('one2many', self.stock.o2m), ('one2one', self.stock.o2o)]:
                branch_loss = getattr(self.stock, branch)
                fg, target, anchors, strides = branch_loss.incidence_assignment
                points = decode_points_px(raw[branch], anchors, strides)
                value, detail = incidence_loss(logits, raw['line_lattice'], points, fg, target,
                    batch['keypoints'], batch['batch_idx'], batch['img'].shape[-2:])
                branch_losses.append(value * coefficient_stock)
                diagnostics.update({f'{branch}_{key}': val for key, val in detail.items()})
                branch_loss.incidence_assignment = None
            incidence = sum(branch_losses)
        weighted_incidence = incidence * self.incidence_weight
        count = batch['img'].shape[0]
        loss = torch.cat((base, (weighted * count).reshape(1), (weighted_incidence * count).reshape(1)))
        self.latest_line_diagnostics = diagnostics
        self.last_scalars = dict(one2many=float(self.stock.o2m), one2one=float(self.stock.o2o),
            line_coefficient=float(coefficient), incidence_coefficient=self.incidence_weight if self.arm == 'incidence' else 0.)
        if self.capture_gradients:
            if self.pending is not None:
                raise RuntimeError('Unconsumed gradient capture; accumulation is unsupported')
            task_a, task_b = base.sum(), weighted * count
            main_ids = parameter_dependencies(task_a)
            auxiliary_gradients = torch.autograd.grad(task_b, self.parameters, retain_graph=True, allow_unused=True)
            pose_gradients = None
            main_gradient = None
            if self.next_step in DIAGNOSTIC_STEPS:
                pose_gradients = torch.autograd.grad(base[1] + base[5], self.parameters,
                    retain_graph=True, allow_unused=True)
                main_gradient = torch.autograd.grad(task_a, self.parameters,
                    retain_graph=True, allow_unused=True)
            self.pending = dict(auxiliary=auxiliary_gradients,
                main_present=[id(p) in main_ids for p in self.parameters], pose=pose_gradients,
                main_gradient=main_gradient, task_a=task_a, task_b=task_b,
                scalars=self.last_scalars.copy(), step=self.next_step)
        return loss, torch.cat((items, weighted.detach().reshape(1), weighted_incidence.detach().reshape(1)))


class CouplingPoseModel(V1.DHTPoseModel):
    def init_criterion(self):
        return CouplingCriterion(self)


def build_model(arm, weights=R0_PATH, line_weight=.1, incidence_weight=.1, verbose=False):
    if arm not in ARMS:
        raise ValueError('Unsupported coupling arm')
    model = V1.build_model('hough_joint', weights, line_weight, verbose=False)
    model.__class__ = CouplingPoseModel
    model.coupling_arm = arm
    model.incidence_weight = float(incidence_weight)
    model.criterion = None
    if verbose:
        print(f'{arm}: {sum(p.numel() for p in model.parameters()):,} trainable parameters; unchanged v1 forward')
    return model
