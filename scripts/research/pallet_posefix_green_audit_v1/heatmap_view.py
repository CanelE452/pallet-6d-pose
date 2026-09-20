"""CPU render of two distinct frozen-PoseFix failure mechanisms.

Run with ``python -m scripts.research.pallet_posefix_green_audit_v1.heatmap_view``.
No inference, model loading, training, or metric selection is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_posefix_green_audit_v1'
OUT = ROOT / 'outputs/pallet_posefix_green_audit_v1'
BG = '#111b26'
CASES = ('worst', 'nontruncated')
MODELS = ('PRIOR', 'REPLAY')


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    raw = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def verify(binding):
    assert bound(ROOT / binding['path']) == binding, binding['path']


def transform(points, matrix):
    points = np.asarray(points, float)
    return np.c_[points, np.ones(len(points))] @ matrix[:2].T


def main():
    diagnostic_path = DOC / 'HEATMAP_DIAGNOSTIC.json'
    diagnostic = read(diagnostic_path)
    assert diagnostic['complete'] and not diagnostic['new_training']
    verify(diagnostic['heatmaps'])
    verify(diagnostic['code'])
    arrays = np.load(ROOT / diagnostic['heatmaps']['path'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), facecolor=BG)
    fig.subplots_adjust(left=.035, right=.895, top=.815, bottom=.13, hspace=.52, wspace=.10)
    fig.suptitle('Why GREEN150 corners move incorrectly: two heatmap mechanisms',
                 color='white', fontsize=17, y=.98)
    fig.text(.465, .922, 'A. Worst case: the wrong corner becomes the dominant peak',
             color='white', ha='center', fontsize=14)
    fig.text(.465, .482, 'B. Non-truncated case: a correct peak remains, but remote mass pulls the mean',
             color='white', ha='center', fontsize=14)
    manifest = []
    for row_index, tag in enumerate(CASES):
        case = diagnostic['cases'][tag]
        verify(case['image'])
        verify(case['annotation'])
        image = np.asarray(Image.open(ROOT / case['image']['path']).convert('RGB'))
        h, w = image.shape[:2]
        matrix = np.asarray(case['crop_matrix'])
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix[[0, 1], [1, 0]], 0, atol=1e-12)
        inverse = np.linalg.inv(matrix)
        gt = np.asarray(case['manual_xy'])
        locations = [gt]
        for label in MODELS:
            model = case['models'][label]
            locations.append(model['expectation_xy'])
            locations.extend(p['image_xy'] for p in model['peaks'])
        locations = np.asarray(locations)
        low, high = locations.min(0) - 25, locations.max(0) + 25
        center = (low + high) / 2
        half = np.maximum((high - low) / 2, [110, 85])
        low, high = np.maximum(center-half, [0, 0]), np.minimum(center+half, [w-1, h-1])
        crop = [float(low[0]), float(high[0]), float(low[1]), float(high[1])]
        for column, label in enumerate(MODELS):
            ax = axes[row_index, column]
            model = case['models'][label]
            prob = arrays[f'{tag}__{label}']
            assert prob.shape == (96, 72) and np.isfinite(prob).all() and (prob >= 0).all()
            assert abs(float(prob.sum()) - 1) < 1e-5
            yy, xx = np.indices(prob.shape)
            computed_mean = transform([[np.sum(xx*prob)*4, np.sum(yy*prob)*4]], inverse)[0]
            np.testing.assert_allclose(computed_mean, model['expectation_xy'], atol=.003, rtol=0)
            peak_y, peak_x = np.unravel_index(np.argmax(prob), prob.shape)
            computed_peak = transform([[peak_x*4, peak_y*4]], inverse)[0]
            np.testing.assert_allclose(computed_peak, model['peaks'][0]['image_xy'], atol=1e-8, rtol=0)
            mean = np.asarray(model['expectation_xy'])
            peak = computed_peak
            # A heatmap cell center (x, y) is input-crop pixel (4*x, 4*y).
            # Extend the image by half a cell so imshow aligns exact centers.
            corners = transform([[-2, -2], [286, 382]], inverse)
            extent = [corners[0, 0], corners[1, 0], corners[1, 1], corners[0, 1]]
            relative = prob / prob.max()
            scaled = np.clip((np.log10(np.maximum(relative, 1e-30)) + 3) / 3, 0, 1)
            rgba = plt.get_cmap('inferno')(scaled)
            rgba[:, :, 3] = .78 * scaled ** .7
            rgba[relative < 1e-3, 3] = 0
            ax.imshow(image, interpolation='nearest')
            ax.imshow(rgba, extent=extent, origin='upper', interpolation='nearest', zorder=2)
            ax.scatter(*mean, s=150, facecolors='none', edgecolors='#fff266', linewidths=2.1, zorder=7)
            ax.scatter(*peak, s=100, marker='^', facecolors='none', edgecolors='#59edff', linewidths=2, zorder=8)
            ax.scatter(*gt, s=105, marker='x', color='#ff68f1', linewidths=2.4, zorder=9)
            if np.linalg.norm(mean-peak) > 4:
                ax.plot([peak[0], mean[0]], [peak[1], mean[1]], color='white',
                        linestyle='--', linewidth=1, alpha=.8, zorder=5)
            ax.annotate(f'GT G{case["canonical_corner"]}', xy=gt, xytext=(7, -19),
                        textcoords='offset points', color='#ff68f1', fontsize=10,
                        bbox=dict(facecolor=BG, alpha=.85, edgecolor='none', pad=2), zorder=10)
            heading = 'Original synthetic PoseFix (PRIOR)' if label == 'PRIOR' else 'Real + synthetic replay PoseFix'
            ax.set_title(f'{heading}\nmean error {model["expectation_error_px"]:.2f}px | '
                         f'peak error {model["peaks"][0]["error_to_manual_px"]:.2f}px',
                         color='white', fontsize=12, pad=12)
            ax.text(.015, .015, f'channel P{model["native"]} / whole-C4 branch b{model["branch"]}',
                    transform=ax.transAxes, color='white', fontsize=9,
                    bbox=dict(facecolor=BG, alpha=.8, edgecolor='none', pad=3))
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(high[1], low[1])
            ax.axis('off')
            manifest.append(dict(tag=tag, id=case['id'], model=label, image=case['image'], annotation=case['annotation'],
                native_channel=model['native'], whole_C4_branch=model['branch'], canonical_corner=case['canonical_corner'],
                manual_xy=gt.tolist(), mean_xy=mean.tolist(), peak_xy=peak.tolist(),
                mean_error_px=model['expectation_error_px'], peak_error_px=model['peaks'][0]['error_to_manual_px'],
                probability_shape=list(prob.shape), probability_sum=float(prob.sum()), panel_max_probability=float(prob.max()),
                crop_matrix=case['crop_matrix'], displayed_original_image_crop=crop,
                probability_overlay_original_extent=list(map(float, extent))))
    cbax = fig.add_axes([.925, .27, .018, .42])
    cbar = fig.colorbar(ScalarMappable(norm=LogNorm(vmin=1e-3, vmax=1), cmap='inferno'), cax=cbax)
    cbar.ax.tick_params(colors='white', labelsize=9)
    cbar.set_label('p / panel maximum (log scale)', color='white', fontsize=10)
    fig.text(.465, .04,
             'Magenta x = manual GT; yellow circle = predicted expectation; cyan triangle = highest-probability cell.\n'
             'Heatmap independently normalized per panel; color/opacity is NOT a confidence comparison across models.\n'
             'Unchanged RGB context and frozen saved probabilities. Diagnostic only: argmax is not a proposed or validated replacement.',
             ha='center', va='bottom', color='white', fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'heatmap_mechanisms.png'
    fig.savefig(output, dpi=145, facecolor=BG)
    plt.close(fig)
    provenance = dict(figure=bound(output), source_diagnostic=bound(diagnostic_path),
        source_heatmaps=diagnostic['heatmaps'], source_code=bound(Path(__file__)), panels=manifest,
        image_generation=False, inference_rerun=False, GPU_used=False, new_training=False,
        RGB_retouch=False, crop_changed_for_model=False,
        visualization='Aligned original-image RGB with native-grid probability overlay; source coordinates exact inverse crop matrix',
        normalization='Each panel divides by its own maximum; log10 ratio clipped [-3,0]; alpha .78*scaled**.7; ratio<1e-3 transparent',
        marker_verification='Expectation reproduced from probability grid within .003px; maximum coordinate exact to 1e-8px',
        clinical_or_causal_claim=False)
    manifest_path = OUT / 'HEATMAP_FIGURE.json'
    if manifest_path.exists():
        previous = read(manifest_path)
        assert previous['source_diagnostic'] == provenance['source_diagnostic']
        assert previous['source_heatmaps'] == provenance['source_heatmaps']
    manifest_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(image=bound(output), manifest=bound(manifest_path)), indent=2))


if __name__ == '__main__':
    main()
