"""Fixed C3 plots and six original illustrative overlays; no pose fitting."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from . import common as C

CASE_PROTOCOL = C.REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json'


def display_row(row):
    solver = row.get('solver') or {}
    new = row['new_pose_estimated']
    return dict(id=row['id'], method=row['method'], native_points=row['native_points'], pose=row['pose'],
        output_status=row['output_status'], new_pose_estimated=new,
        input_points=row.get('input_points'), reprojected_ids=row.get('reprojected_ids', []),
        selected_corner_ids=row.get('selected_corner_ids', []),
        evaluation_reference=row.get('evaluation_reference', {}),
        head_arm=row.get('head_arm', 'FIXED_N3_CONTROL'),
        actual_used_U=solver.get('used', []), applied_H=row.get('hidden_initial', []),
        fallback_used=row['fallback_used'], pose_available=row['pose_available'],
        accepted_fit_ids=solver.get('fit_input_ids', []) if new else [],
        accepted_inlier_ids=solver.get('final_inliers', []) if new else [],
        raw_candidate_diagnostic_inliers=solver.get('final_inliers', []),
        original_H=row.get('original_self_hidden_initial', []))


def _render(args):
    C.verify_protocol(args)
    C.protect(args)
    folder = Path(args.output)
    scoring = C.read(folder / 'SCORING_RECEIPT.json')
    C.require(scoring['complete'] is True, 'complete scoring required for figures')
    C.bound(args.protocol, scoring['protocol'], 'scoring protocol')
    C.bound(folder / 'PREDICTIONS.jsonl.gz', scoring['predictions'], 'scored methods')
    C.bound(folder / 'FIXED_PREDICTIONS.jsonl.gz', scoring['fixed_predictions'], 'scored controls')
    for name in ('FIGURE_BINDINGS.json', 'FIGURE_CASE_ROWS.jsonl.gz'):
        C.output_path(args, name)
    figures = folder / 'figures'
    C.require(not figures.exists(), 'preserve previous figures')
    figures.mkdir()
    records = [display_row(row) for filename in ('PREDICTIONS.jsonl.gz', 'FIXED_PREDICTIONS.jsonl.gz')
               for row in C.rows(folder / filename)]
    data = {(row['method'], row['id']): row for row in records}
    C.require(len(data) == len(records) == 1225, 'five complete same245 methods required')
    ids = C.cohort_ids(args)
    shown = ('BASE', 'N3_SUBPIX', *C.METHODS)
    C.require(all((method, identity) in data for identity in ids for method in shown), 'figure cohort differs')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, field, unit in zip(axes, ('translation_cm', 'rotation_deg'), ('cm', 'degrees')):
        for index, method in enumerate(shown):
            values = np.array([data[method, identity]['pose'][field] for identity in ids
                               if data[method, identity]['pose']['available']])
            axis.scatter(values, np.full(len(values), index), s=8, alpha=.24)
            if len(values):
                low, high = np.quantile(values, [.1, .9])
                axis.plot([low, high], [index, index], linewidth=3)
                axis.scatter([values.mean()], [index], marker='D', facecolor='white', edgecolor='black', zorder=5)
        axis.set_yticks(range(len(shown)), shown if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xscale('symlog', linthresh=1)
        axis.set_xlabel(field + ' (' + unit + ')')
        axis.grid(alpha=.2)
    fig.suptitle('Fixed same-observation C3: Clean153 + Moderate92\nAll operational outputs including fallback; diamond=mean, segment=P10-P90; geometric proxy reference')
    fig.tight_layout()
    fig.savefig(figures / '01_all_operational.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    for row_index, method in enumerate(C.METHODS):
        for axis, field in zip(axes[row_index], ('translation_cm', 'rotation_deg')):
            pairs = np.array([(data['N3_SUBPIX', identity]['pose'][field], data[method, identity]['pose'][field])
                              for identity in ids if data['N3_SUBPIX', identity]['pose']['available'] and
                              data[method, identity]['pose']['available']], dtype=float).reshape(-1, 2)
            if len(pairs):
                axis.scatter(pairs[:, 0], pairs[:, 1], s=12, alpha=.5)
                upper = max(1., pairs.max())
            else:
                upper = 1.
            axis.plot([0, upper], [0, upper], 'k--')
            axis.set_xscale('symlog', linthresh=1)
            axis.set_yscale('symlog', linthresh=1)
            axis.set_xlabel('Fixed N3 ' + field)
            axis.set_ylabel(method + '\n' + field)
            axis.grid(alpha=.2)
    fig.suptitle('Same245 paired operational outputs: below diagonal favors C3 method\nNEW and fallback counts remain separate in the report')
    fig.tight_layout()
    fig.savefig(figures / '02_same_frame_pairs.png', dpi=140)
    plt.close(fig)
    frozen = C.read(args.protocol)['inputs']['visual_case_protocol']
    C.bound(CASE_PROTOCOL, frozen, 'original visual cases')
    case_protocol = C.read(CASE_PROTOCOL)
    C.require(len(case_protocol['cases']) == 6, 'same six prespecified cases required')
    case_rows = []
    for index, case in enumerate(case_protocol['cases']):
        path = Path(args.source_root) / case['image']['path']
        C.bound(path, case['image'], 'unchanged illustrative RGB')
        rgb = np.asarray(Image.open(path).convert('RGB'))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for axis, method in zip(axes.flat, ('N3_SUBPIX', *C.METHODS)):
            row = data[method, case['id']]
            axis.imshow(rgb)
            output = np.asarray(row['native_points'], dtype=float)
            valid = np.isfinite(output[:8]).all(axis=1) & (output[:8] != [-1, -1]).any(axis=1)
            axis.scatter(output[:8][valid, 0], output[:8][valid, 1], s=32,
                         facecolor='none', edgecolor='#00dce6', linewidth=1.4)
            for corner in np.flatnonzero(valid):
                axis.text(output[corner, 0] + 3, output[corner, 1] + 3, str(corner), color='cyan', fontsize=8)
            for corner in row['reprojected_ids']:
                axis.scatter(*output[corner], s=68, marker='s', facecolor='none', edgecolor='#ff8c24')
            if method != 'N3_SUBPIX':
                observations = np.asarray(row['input_points'], dtype=float)
                active = np.isfinite(observations[:8]).all(axis=1) & (observations[:8] != [-1, -1]).any(axis=1)
                axis.scatter(observations[:8][active, 0], observations[:8][active, 1], s=35, marker='+', color='#42ff50')
            reference = row['evaluation_reference']
            gt = np.asarray(reference.get('native_points_px', []), dtype=float)
            if len(gt):
                valid_ids = reference.get('valid_native_ids', [])
                axis.scatter(gt[valid_ids, 0], gt[valid_ids, 1], s=19, marker='.', color='white', alpha=.65)
            score = row['pose']
            T, R = score.get('translation_cm'), score.get('rotation_deg')
            axis.set_title(method + '\n' + row['output_status'] + ' / T=' + str(round(T, 4) if T is not None else None) +
                'cm / R=' + str(round(R, 4) if R is not None else None) + 'deg\naccepted fit=' + str(row['accepted_fit_ids']) +
                '; 8px diagnostic inliers=' + str(row['accepted_inlier_ids']), fontsize=8)
            axis.axis('off')
            case_rows.append({key: row[key] for key in ('id', 'method', 'pose', 'output_status', 'new_pose_estimated',
                'selected_corner_ids', 'accepted_fit_ids', 'accepted_inlier_ids', 'raw_candidate_diagnostic_inliers',
                'reprojected_ids', 'original_H', 'head_arm', 'actual_used_U', 'applied_H',
                'fallback_used', 'pose_available')})
        fig.suptitle(case['label'] + ' / ' + case['id'] + '\ncyan=output; green+=same sparse observation; orange square=H projection; white=proxy\n'
                      'Missing native display coordinates are not fit inputs. STANDARD fits whole U; 8px inliers are diagnostic.', fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / ('case_%02d.png' % (index + 1)), dpi=140, bbox_inches='tight')
        plt.close(fig)
    C.require(len(case_rows) == 24, 'six times four panels required')
    # Saved display evidence is a derived artifact, not a replacement of raw witnesses.
    import gzip
    import json
    target = C.output_path(args, 'FIGURE_CASE_ROWS.jsonl.gz')
    with target.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        for row in case_rows:
            zipped.write((json.dumps(C.finite(row), ensure_ascii=False, allow_nan=False) + '\n').encode())
    C.verify_protocol(args)
    C.protect(args)
    C.require(len(list(figures.glob('*.png'))) == 8, 'eight actual PNG files required')
    C.write_new(C.output_path(args, 'FIGURE_BINDINGS.json'), dict(complete=True,
        figures=[C.binding(path) for path in sorted(figures.glob('*.png'))],
        fixed_case_protocol=C.binding(CASE_PROTOCOL), case_rows=C.binding(target), renderer_code=C.binding(__file__),
        scored_methods=C.binding(folder / 'PREDICTIONS.jsonl.gz'), fixed_controls=C.binding(folder / 'FIXED_PREDICTIONS.jsonl.gz'),
        scoring_receipt=C.binding(folder / 'SCORING_RECEIPT.json'), protocol=C.binding(args.protocol),
        panels=24, PNG_files=8, selection_uses_new_scores=False,
        only_display_fields_retained_in_memory=True, original_raw_witnesses_unchanged=True,
        new_detector_head_PnP_GT_score_training_RGB_calls=0, physical_truth_certified=False))
    print('SAME_OBSERVATION_FIGURES', 8, flush=True)


def run(args):
    C.verify_protocol(args)
    C.protect(args)
    for name in ('FIGURES_STARTED.json', 'FIGURES_FAILURE.json'):
        C.output_path(args, name)
    folder = Path(args.output)
    C.write_new(C.output_path(args, 'FIGURES_STARTED.json'), dict(
        protocol=C.binding(args.protocol), renderer_code=C.binding(__file__),
        scoring_receipt=C.binding(folder / 'SCORING_RECEIPT.json'),
        configured_PNG_files=8, configured_case_panels=24, automatic_retry=False))
    try:
        _render(args)
    except BaseException as error:
        figures = folder / 'figures'
        case_rows = folder / 'FIGURE_CASE_ROWS.jsonl.gz'
        C.write_new(C.output_path(args, 'FIGURES_FAILURE.json'), dict(complete=False,
            error=dict(type=type(error).__name__, message=str(error)),
            preserved_PNG_prefix=[C.binding(path) for path in sorted(figures.glob('*.png'))]
                if figures.is_dir() else [],
            preserved_case_rows=C.binding(case_rows) if case_rows.is_file() else None,
            new_detector_head_PnP_GT_score_training_RGB_calls=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    run(C.parser(__doc__, stages=None).parse_args())
