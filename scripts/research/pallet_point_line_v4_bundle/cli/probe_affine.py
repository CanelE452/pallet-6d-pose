"""G1: measure the real input-pixel <-> P3/P4 feature-index relation.

Impulse fixtures are pushed through the SAME verified predictor module stack
that the export hooks. Nothing here is guessed from YOLO convention.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from cli.common import ROOT, RESULTS, write, backbone_binding  # noqa: E402
sys.path.insert(0, str(ROOT))
from pointline_v4.adapter_helpers import find_unique_hough_module, capture_native_features  # noqa: E402


def main():
    from scripts.research.pallet_dht_joint_v1.evaluate import CanonicalPredictor
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    binding = backbone_binding()
    predictor = CanonicalPredictor(binding['checkpoint'], '0')
    module = find_unique_hough_module(predictor.model.model)
    device = next(predictor.model.model.parameters()).device
    size = 640
    positions = [160, 208, 256, 304, 352, 400, 448]

    def features_of(image):
        with capture_native_features(module, detach=True, cpu=True) as capture:
            with torch.inference_mode():
                predictor.model.model(image)
        return capture.features

    background = torch.full((1, 3, size, size), .45, dtype=torch.float32, device=device)
    reference = features_of(background)
    rows = {8: [], 16: []}
    for pos in positions:
        image = background.clone()
        # Strong, small, high-contrast patch centred exactly on input index `pos`.
        image[:, :, pos - 2:pos + 3, pos - 2:pos + 3] = 1.0
        planes = features_of(image)
        for stride, feature, base in ((8, planes[0], reference[0]), (16, planes[1], reference[1])):
            # Difference isolates the fixture; the static background response cancels.
            weight = (feature[0] - base[0]).abs().mean(0).double().numpy()
            peak = np.unravel_index(int(weight.argmax()), weight.shape)
            r0, r1 = max(0, peak[0] - 2), peak[0] + 3
            c0, c1 = max(0, peak[1] - 2), peak[1] + 3
            patch = weight[r0:r1, c0:c1]
            ys, xs = np.mgrid[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
            total = patch.sum()
            rows[stride].append(dict(input_px=pos, peak_row=int(peak[0]), peak_col=int(peak[1]),
                                     peak_weight=float(weight[peak]), patch_mass=float(total),
                                     centroid_row=float((patch * ys).sum() / total),
                                     centroid_col=float((patch * xs).sum() / total)))
    result = dict(schema='pointline_v4_affine_probe_1', checkpoint_sha256=binding['sha256'],
                  input_size=size, fixture='3x3 bright square on 0.45 grey, centred at the listed input index',
                  measurements={str(k): v for k, v in rows.items()})
    for stride, values in rows.items():
        x = np.array([v['input_px'] for v in values], float)
        for axis in ('row', 'col'):
            y = np.array([v['centroid_' + axis] for v in values], float)
            slope, intercept = np.polyfit(x, y, 1)
            residual = float(np.abs(y - (slope * x + intercept)).max())
            result[f'fit_stride{stride}_{axis}'] = dict(
                slope=float(slope), inverse_slope=float(1 / slope), intercept=float(intercept),
                max_abs_residual_cells=residual,
                offset_if_slope_is_exactly_1_over_stride=float(np.mean(y - x / stride)),
                candidate_offset_minus_half=-0.5,
                candidate_offset_true_receptive_centre=-(stride / 2 - .5) / stride)
    write(RESULTS / 'AFFINE_PROBE.json', result)
    for key in sorted(k for k in result if k.startswith('fit_')):
        f = result[key]
        print(key, 'inv_slope=%.4f' % f['inverse_slope'], 'offset=%.4f' % f['offset_if_slope_is_exactly_1_over_stride'],
              'resid=%.4f' % f['max_abs_residual_cells'],
              'half=%.4f rf=%.4f' % (f['candidate_offset_minus_half'], f['candidate_offset_true_receptive_centre']))


if __name__ == '__main__':
    main()
