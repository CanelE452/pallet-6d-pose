"""Run bound public-only RGB-D/mesh projection diagnostics, without fitting."""
from pathlib import Path
import hashlib
import io
import json
import stat
import sys
import types


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for a in reversed([p, *p.parents]):
        try:
            assert not stat.S_ISLNK(a.lstat().st_mode), str(a)
        except FileNotFoundError:
            pass
    return p


def bound(binding):
    b = safe(Path(binding['path'])).read_bytes()
    assert len(b) == binding['size_bytes']
    assert hashlib.sha256(b).hexdigest() == binding['sha256']
    return b


def run(entry_path):
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eb = safe(entry_path).read_bytes()
    e = json.loads(eb)
    assert e['approved_public_geometry_decode'] is True
    assert e['training_or_model_inference'] is False
    assert e['capture_frame'] == '0.55m_hall/0011'
    assert e['driver_sha256'] == hashlib.sha256(safe(Path(__file__)).read_bytes()).hexdigest()
    identity = json.loads(bound(e['identity_QA']))
    assert identity['completed'] and identity['PASS']
    assert identity['permits_this_public_frame_geometry_diagnostic'] is True
    assert identity['permits_training_or_claims_full_capture_disjointness'] is False
    toy = json.loads(bound(e['projection_toy_QA']))
    assert toy['PASS'] is True
    raw = {k: bound(v) for k, v in e['inputs'].items()}
    assert e['inputs']['RGB']['sha256'] == identity['public_RGB_binding']['sha256']
    code = bound(e['projection_code'])
    mod = types.ModuleType('v258_closed_public_projection')
    mod.__file__ = e['projection_code']['path']
    sys.modules[mod.__name__] = mod
    exec(compile(code, mod.__file__, 'exec'), mod.__dict__)
    out = safe(Path(e['output_directory']))
    assert not out.exists()
    out.mkdir()

    def put(name, value):
        with safe(out / name).open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')

    put('STARTED.json', {'entry_sha256': hashlib.sha256(eb).hexdigest(),
                         'entry': e, 'training_or_model_inference': False})
    try:
        rgb_image = Image.open(io.BytesIO(raw['RGB']))
        rgb_original_mode = rgb_image.mode
        assert rgb_image.mode in ('RGB', 'RGBA')
        if rgb_image.mode == 'RGBA':
            assert np.all(np.asarray(rgb_image)[:, :, 3] == 255)
        rgb = np.asarray(rgb_image.convert('RGB'))
        depth = np.asarray(Image.open(io.BytesIO(raw['DEPTH'])))
        sam = np.asarray(Image.open(io.BytesIO(raw['SAM'])))
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        assert depth.shape == sam.shape == (480, 640)
        assert np.issubdtype(depth.dtype, np.integer) and depth.min() >= 0 and depth.max() <= 65535
        assert set(np.unique(sam).tolist()) <= {0, 255}
        observed = depth.astype(np.float64) / 1000.0
        observed[depth == 0] = np.nan
        mask = sam == 255
        K = np.loadtxt(io.StringIO(raw['K'].decode()))
        T = np.loadtxt(io.StringIO(raw['POSE'].decode()))
        assert T.shape == (4, 4) and K.shape == (3, 3)
        assert np.isfinite(T).all() and np.isfinite(K).all()
        assert np.allclose(T[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        verts, faces, parser_report = mod.parse_obj(e['inputs']['OBJ']['path'], binding=e['inputs']['OBJ'])
        rows = []
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        for row_index, (name, transform) in enumerate((('T', T), ('inverse_T', np.linalg.inv(T)))):
            rendered, support, report = mod.render_depth(verts, faces, transform, K, (480, 640), near=1e-9)
            ax = axes[row_index, 0]
            ax.imshow(rgb)
            if np.any(mask) and not np.all(mask):
                ax.contour(mask.astype(float), levels=[0.5], colors=['lime'], linewidths=0.8)
            stats = None
            if rendered is not None:
                if np.any(support) and not np.all(support):
                    ax.contour(support.astype(float), levels=[0.5], colors=['magenta'], linewidths=0.8)
                stats, delta = mod.describe_observed_depth(observed, rendered, mask)
                with safe(out / (name + '_RENDER_DEPTH_M.npy')).open('xb') as f:
                    np.save(f, rendered, allow_pickle=False)
                with safe(out / (name + '_OBSERVED_MINUS_RENDER_M.npy')).open('xb') as f:
                    np.save(f, delta, allow_pickle=False)
                im = axes[row_index, 1].imshow(rendered, cmap='viridis', vmin=0, vmax=8)
                fig.colorbar(im, ax=axes[row_index, 1], label='rendered camera Z (m)')
                im = axes[row_index, 2].imshow(delta, cmap='coolwarm', vmin=-0.1, vmax=0.1)
                fig.colorbar(im, ax=axes[row_index, 2], label='observed minus render (m), display clipped')
            else:
                for col in (1, 2):
                    axes[row_index, col].text(0.5, 0.5, 'UNSUPPORTED: see report', ha='center')
            ax.set_title(name + ': RGB / SAM green / mesh magenta')
            axes[row_index, 1].set_title(name + ': mesh depth')
            axes[row_index, 2].set_title(name + ': raw depth residual')
            for col in range(3):
                axes[row_index, col].set_axis_off()
            rows.append({'pose_interpretation': name, 'render': report, 'depth_and_mask': stats})
            print(json.dumps({'pose_interpretation': name, 'render': report,
                              'depth_and_mask': stats}, allow_nan=False), flush=True)
        fig.suptitle('Pallox public 0.55m_hall/0011: geometry only, no pose optimization or D/E labels')
        figure = safe(out / 'PUBLIC_GEOMETRY_DIAGNOSTIC.png')
        assert not figure.exists()
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        result = {'completed': True, 'PASS': True,
                  'scope': 'Executed geometry diagnostic, not a registration/visibility accuracy pass or training admission.',
                  'entry_sha256': hashlib.sha256(eb).hexdigest(),
                  'RGB_original_mode': rgb_original_mode, 'image_shape': list(rgb.shape),
                  'depth_integer_dtype': str(depth.dtype), 'depth_zero_invalid': int(np.sum(depth == 0)),
                  'SAM_values': np.unique(sam).tolist(), 'SAM_foreground': int(mask.sum()),
                  'pose_matrix_as_published': T.tolist(), 'K_as_published': K.tolist(),
                  'rotation_det': float(np.linalg.det(T[:3, :3])),
                  'rotation_orthogonality_Frobenius': float(np.linalg.norm(T[:3, :3].T @ T[:3, :3] - np.eye(3))),
                  'OBJ_parser_report': parser_report, 'hypotheses': rows,
                  'pose_winner_selected': False, 'new_point_visibility_labels': 0,
                  'model_forward_or_optimizer': 0, 'current223_or_protected_case_reads': 0,
                  'physical_native_point_mapping_required_for_aux_presence': False,
                  'saved_depth_maps_not_binary_D_E_targets': True}
        put('GEOMETRY_DIAGNOSTIC.json', result)
        files = []
        for p in [out / 'STARTED.json', out / 'GEOMETRY_DIAGNOSTIC.json', figure,
                  *[out / (name + suffix) for name in ('T', 'inverse_T')
                    for suffix in ('_RENDER_DEPTH_M.npy', '_OBSERVED_MINUS_RENDER_M.npy')]]:
            safe(p)
            if p.exists():
                b = p.read_bytes()
                files.append({'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)})
        put('COMPLETE.json', {'completed': True, 'PASS': True, 'files': files,
                              'new_point_labels': 0, 'training_or_model_inference': False})
        print(json.dumps({'completed': True, 'PASS': True, 'outputs': len(files),
                          'new_point_labels': 0}), flush=True)
    except BaseException as exc:
        put('FAILURE.json', {'completed': False, 'PASS': False, 'error': str(exc),
                             'automatic_retry': False})
        raise


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', type=Path, required=True)
    run(ap.parse_args().entry)
