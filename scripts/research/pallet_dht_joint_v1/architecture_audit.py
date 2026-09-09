"""Independent CPU geometry/gradient audit; no data or GPU training.

Generated tensors and labels check plumbing, not accuracy. Only this audit's
JSON is written. Existing checkpoint, implementation and result files are read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import torch
from ultralytics.utils.torch_utils import ModelEMA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import hough_block as H
import integration as I
from scripts.research.deep_hough_side_v1.dht import SparseDHT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grad_l1(model):
    groups = dict(backbone=0., hough_reduce=0., hough_line_head=0., hough_outputs=0.)
    for name, p in model.named_parameters():
        key = ('backbone' if name.startswith('model.0.') else
               'hough_reduce' if '.hough.reduce.' in name else
               'hough_line_head' if '.hough.line_head.' in name else
               'hough_outputs' if '.hough.outputs.' in name else None)
        if key and p.grad is not None:
            groups[key] += float(p.grad.detach().abs().sum())
    return groups


def geometry_checks():
    rows = []
    for height, width in [(40, 40), (32, 40), (24, 40)]:
        g = SparseDHT(height, width, 90, .5, False, 28.)
        a = g.vote_matrix.double()
        x = torch.randn(1, 1, height, width, dtype=torch.float64)
        y = torch.randn(1, 1, 90, 113, dtype=torch.float64)
        ax = torch.sparse.mm(a, x.flatten().reshape(-1, 1)).reshape_as(y)
        effective_ax = H.smooth_rho(ax)
        aty = torch.sparse.mm(a.t(), H.smooth_rho(y).flatten().reshape(-1, 1)).reshape_as(x)
        adjoint_error = abs(float((effective_ax * y).sum() - (x * aty).sum()))
        assert adjoint_error < 1e-9
        ht_ones = H.normalized_hough(torch.ones_like(x), g)
        positive_mass = H.smooth_rho(g.mass) > 1e-7
        ht_constant_error = float((ht_ones[0, 0][positive_mass] - 1).abs().max())
        bp_constant_error = float((H.normalized_backprojection(torch.ones_like(y), g) - 1).abs().max())
        # Geometry weights/masses are stored in FP32 even for this FP64 check.
        assert ht_constant_error < 3e-6 and bp_constant_error < 3e-6
        isolated = torch.zeros_like(y); isolated[0, 0, 0, 68] = 1  # theta0, rho6.
        assert not bool(g.valid[0, 68])
        before = torch.sparse.mm(a.t(), isolated.flatten().reshape(-1, 1))
        after = H.normalized_backprojection(isolated, g)
        assert float(before.abs().sum()) == 0 and float(after.abs().sum()) > 0
        # The smooth operator respects the angular seam's rho sign reversal.
        seam_error = float((H.smooth_rho(y.flip(-1)) - H.smooth_rho(y).flip(-1)).abs().max())
        assert seam_error < 1e-12
        rows.append(dict(feature_hw=[height, width],
                         unnormalized_effective_adjoint_absolute_error=adjoint_error,
                         ht_constant_max_error=ht_constant_error,
                         bp_constant_max_error=bp_constant_error,
                         rho_flip_commutation_max_error=seam_error,
                         zero_vote_axis_peak_original_bp_l1=float(before.abs().sum()),
                         zero_vote_axis_peak_effective_bp_l1=float(after.abs().sum()),
                         zero_vote_axis_peak_feedback_cells=int((after.abs() > 0).sum())))
    # Autograd through both directions, including the actual normalizations.
    tiny = SparseDHT(3, 4, 6, .5, False, 3.)
    x = torch.randn(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: H.normalized_backprojection(
        H.normalized_hough(z, tiny), tiny), (x,), eps=1e-6, atol=1e-5, rtol=1e-4)
    return dict(PASS=True, cases=rows, double_autograd_gradcheck=True,
                normalization_note='KA and A^T K are adjoints before separate row/column mean normalization; normalized BP is not the adjoint of normalized HT.')


def point_gradient_checks():
    model = I.build_model('hough_features').train()
    model.args.epochs = 2
    batch = dict(img=torch.randn(2, 3, 96, 128), batch_idx=torch.tensor([0., 1.]),
                 cls=torch.zeros(2, 1), bboxes=torch.tensor([[.5, .5, .5, .5]] * 2),
                 keypoints=torch.tensor([[[.25,.25,2],[.75,.25,2],[.75,.75,2],[.25,.75,2],
                                          [.35,.35,2],[.65,.35,2],[.65,.65,2],[.35,.65,2],[.5,.5,2]]] * 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    history = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(copy.deepcopy(batch))
        # Coordinate objective only: stock point-location plus stock RLE.
        point = loss[1] + loss[5]
        assert float(point) > 0
        point.backward()
        gradients = grad_l1(model)
        assert gradients['backbone'] > 0 and gradients['hough_outputs'] > 0
        if step == 1:
            assert gradients['hough_reduce'] > 0 and gradients['hough_line_head'] > 0
        history.append(dict(generated_tensor_step=step + 1,
                            point_location_plus_RLE_loss=float(point), gradient_l1=gradients))
        optimizer.step()
    # The deployed one2one branch uses the stock detached feature path.
    model.zero_grad(set_to_none=True)
    raw = model(batch['img'])
    loss, _ = model.criterion.stock.one2one.loss(raw['one2one'], copy.deepcopy(batch))
    (loss[1] + loss[5]).backward()
    detached = grad_l1(model)
    assert all(v == 0 for v in detached.values())
    assert model.model[-1].hough.line_logits is None and model.model[-1].hough.lattice is None
    # EMA includes new parameters and retains a serializable complete module.
    ema = ModelEMA(model)
    name = next(n for n, _ in model.named_parameters() if '.hough.outputs.0.weight' in n)
    before = dict(ema.ema.named_parameters())[name].detach().clone()
    with torch.no_grad():
        dict(model.named_parameters())[name].add_(.01)
    expected = before * ema.decay(1) + dict(model.named_parameters())[name].detach() * (1 - ema.decay(1))
    ema.update(model)
    ema_error = float((dict(ema.ema.named_parameters())[name] - expected).abs().max())
    assert ema_error < 1e-7
    clone = copy.deepcopy(ema.ema).eval()
    clone.criterion = None
    stream = io.BytesIO(); torch.save(clone, stream); stream.seek(0)
    loaded = torch.load(stream, map_location='cpu').eval()
    with torch.no_grad():
        output_a = clone(batch['img'])[0]
        output_b = loaded(batch['img'])[0]
    assert torch.equal(output_a, output_b)
    return dict(PASS=True, auxiliary_weight=0., generated_tensor_only=True,
                point_loss_gradients=history, stock_one2one_detached_gradient_l1=detached,
                ema_new_parameter_max_error=ema_error, serialized_cpu_forward_exact=True,
                inference_forward_uses_GT=False, actual_accuracy_evaluation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--training-batch', type=int, required=True,
                        help='Explicit proposed main batch for the read-only budget review.')
    parser.add_argument('--training-epochs', type=int, required=True,
                        help='Explicit proposed main epochs for the read-only budget review.')
    args = parser.parse_args()
    if args.training_batch <= 0 or args.training_epochs <= 0:
        raise ValueError('Positive reviewed training budget required')
    torch.set_num_threads(2); torch.manual_seed(308)
    paths = [HERE / n for n in ['hough_block.py', 'integration.py', 'train.py', 'line_targets.py']]
    hashes = {str(p): sha(p) for p in paths}
    geometry = geometry_checks()
    gradients = point_gradient_checks()
    unchanged = all(sha(p) == hashes[str(p)] for p in paths)
    assert unchanged, 'Implementation changed during CPU audit; rerun on final code.'
    result = dict(schema='pallet_dht_joint_architecture_audit_v1', complete=True, PASS=True,
                  source_sha256=hashes, audit_source_sha256=sha(__file__), device='cpu',
                  geometry=geometry, gradients=gradients,
                  read_only_training_review=dict(
                      reviewed_proposal_epochs=args.training_epochs, train_frames_per_epoch=55980,
                      batch=args.training_batch,
                      optimizer_updates=math.ceil(55980 / args.training_batch) * args.training_epochs,
                      nbs=args.training_batch, accumulation=1,
                      budget_source='Explicit CLI proposal; actual completed budget is verified by TRAINING_AUDIT.',
                      source_labels='source file SHA and float32 manifest equality; stock post-transform normalized coordinates',
                      role_visibility='12 amodal structural roles; visibility>0 is coordinate supervision, not physical edge evidence',
                      baseline='point_only must receive same data, augmentation, optimizer and epoch budget',
                      auxiliary_ablation='hough_joint versus hough_features; R0-only comparisons confound additional training',
                      validation='4020 synthetic validation frames are evaluated each epoch; final epoch is fixed, no independent unseen validation claim',
                      inference='Full model refinement can change detections and confidences; negatives need fresh inference',
                      resume='Completed epoch model/EMA/optimizer/scaler/RNG restored; LambdaLR reconstructed from fixed config and epoch; no GPU bit-determinism claim',
                      scope='Code and CPU checks only; actual GPU backward, image augmentation parity, budget and accuracy require run artifacts'))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    target = args.run_dir / 'ARCHITECTURE_AUDIT.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(PASS=True, artifact=str(target), artifact_sha256=sha(target))))


if __name__ == '__main__':
    main()
