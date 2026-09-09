"""Descriptive post-hoc diagnostics from saved A/B CSV, GT and 12 attention maps.

Does not train, perform model inference, or modify existing experiment artifacts.
All main aggregates average separately computed seed statistics. Attention-map
comparisons describe the 12 previously selected seed-1 examples only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
METRICS = ['angle_deg', 'distance_px', 'distance_diagonal', 'distance_cuboid_diagonal',
           'attention_entropy', 'attention_effective_tokens', 'oracle_midpoint_shift_px']


def summary(frame):
    per_seed = []
    for seed, sub in frame.groupby('seed'):
        row = {'seed': int(seed), 'n_frames': int(sub.id.nunique()), 'n_roles': len(sub)}
        for key in METRICS:
            values = sub[key].to_numpy()
            row[key + '_median'] = float(np.median(values))
            row[key + '_p90'] = float(np.quantile(values, .9))
            row[key + '_mean'] = float(np.mean(values))
        for key in ['position_dominated', 'angle_good_position_bad', 'angle_good', 'joint_good']:
            row[key + '_fraction'] = float(sub[key].mean())
        per_seed.append(row)
    return {'seed_mean': {k: float(np.mean([r[k] for r in per_seed]))
                          for k in per_seed[0] if k != 'seed'}, 'per_seed': per_seed}


def grouped_summary(frame, keys):
    result = {}
    for group, sub in frame.groupby(keys):
        group = group if isinstance(group, tuple) else (group,)
        cursor = result
        for value in group[:-1]:
            cursor = cursor.setdefault(str(value), {})
        cursor[str(group[-1])] = summary(sub)
    return result


def attention_diagnostics(run, records):
    manifest = json.loads((run / 'visualization_manifest.json').read_text())
    output = []
    for sample in manifest['samples']:
        artifact = np.load(run / sample['artifact'])
        for arm in ('A', 'B'):
            maps = artifact[f'{arm}_attention'].reshape(12, -1).astype(float)
            maps /= maps.sum(axis=1, keepdims=True)
            normalized = maps / np.linalg.norm(maps, axis=1, keepdims=True)
            cosine = normalized @ normalized.T
            pairs = cosine[np.triu_indices(12, 1)]
            peaks = maps.argmax(axis=1)
            pair_peaks = (peaks[:, None] == peaks[None, :])[np.triu_indices(12, 1)]
            entropy = -(maps * np.log(maps.clip(1e-30))).sum(axis=1)
            row = {'id': sample['id'], 'population': sample['population'],
                   'group': records[sample['id']]['group'], 'arm': arm,
                   'role_pair_cosine_mean': float(pairs.mean()),
                   'role_pair_cosine_median': float(np.median(pairs)),
                   'role_pair_cosine_gt_095_fraction': float((pairs > .95).mean()),
                   'role_pair_peak_same_fraction': float(pair_peaks.mean()),
                   'unique_peak_cells': int(len(np.unique(peaks))),
                   'effective_tokens_role_median': float(np.median(np.exp(entropy))),
                   'peak_attention_mass_role_mean': float(maps.max(axis=1).mean())}
            output.append(row)
    table = pd.DataFrame(output)
    aggregate = table.groupby(['population', 'arm']).mean(numeric_only=True).reset_index()
    return output, aggregate.to_dict('records')


def paired_diagnostics(frame):
    # Average roles within frame, then pair identical frame and seed. Report
    # frame-mean errors rather than pretending 12 lines are independent samples.
    table = frame.groupby(['population', 'group', 'id', 'seed', 'arm'])[
        ['distance_px', 'distance_cuboid_diagonal', 'angle_deg']].mean().unstack('arm')
    rows = []
    for index, pair in table.iterrows():
        population, group, sample_id, seed = index
        row = dict(population=population, group=group, id=sample_id, seed=int(seed))
        for metric in ['distance_px', 'distance_cuboid_diagonal', 'angle_deg']:
            row[metric + '_A'] = float(pair[(metric, 'A')])
            row[metric + '_B'] = float(pair[(metric, 'B')])
            row[metric + '_delta_B_minus_A'] = row[metric + '_B'] - row[metric + '_A']
        rows.append(row)
    pairs = pd.DataFrame(rows)
    aggregate = {}
    for population, sub in pairs.groupby('population'):
        per_seed = []
        for seed, seed_rows in sub.groupby('seed'):
            delta = seed_rows.distance_px_delta_B_minus_A.to_numpy()
            per_seed.append({'seed': int(seed), 'n_frames': len(delta),
                             'B_better_frame_fraction': float((delta < 0).mean()),
                             'mean_distance_delta_px': float(delta.mean()),
                             'median_distance_delta_px': float(np.median(delta))})
        # Descriptive frame-cluster bootstrap after averaging the three paired
        # seeds for each image. Captures sampled-image variation only; sessions
        # are not resampled and this is not an independent final-test estimate.
        frame_delta = sub.groupby('id').distance_px_delta_B_minus_A.mean().to_numpy()
        rng = np.random.default_rng(417)
        boot = rng.choice(frame_delta, size=(10000, len(frame_delta)), replace=True).mean(axis=1)
        aggregate[population] = {'per_seed': per_seed,
                                'mean_paired_frame_delta_px': float(frame_delta.mean()),
                                'frame_bootstrap_95pct_interval_px': np.quantile(boot, [.025, .975]).tolist()}
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('data/pallet/results/hough_attention_transfer_v1'))
    args = parser.parse_args()
    run = args.run_dir
    out = run / 'diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / 'manifest.json').read_text())
    records = {record['id']: record for pop in manifest['populations'].values() for record in pop}
    frame = pd.read_csv(run / 'PER_ROLE.csv')
    geometry = []
    for sample_id, record in records.items():
        points = np.asarray(record['gt_points'], float)
        valid = np.isfinite(points).all(axis=1) & np.asarray(record.get('gt_valid', [True] * 8))
        span = np.ptp(points[valid], axis=0)
        cuboid_diagonal = float(np.linalg.norm(span))
        for role, (a, b) in enumerate(EDGES):
            geometry.append({'id': sample_id, 'role': role, 'cuboid_diagonal': cuboid_diagonal,
                             'segment_length': float(np.linalg.norm(points[b] - points[a])),
                             'edge_family': 'near_face' if role < 4 else 'far_face' if role < 8 else 'depth',
                             'axis_family': 'height' if role in [1, 3, 5, 7] else 'width' if role in [0, 2, 4, 6] else 'depth'})
    frame = frame.merge(pd.DataFrame(geometry), on=['id', 'role'], validate='many_to_one')
    frame['distance_cuboid_diagonal'] = frame.distance_px / frame.cuboid_diagonal
    frame['attention_effective_tokens'] = np.exp(frame.attention_entropy * np.log(2500))
    frame['oracle_midpoint_shift_px'] = frame.segment_length * np.sin(np.deg2rad(frame.angle_deg)) / 2
    # Exact identity: mean(abs(endpoint signed distances)) =
    # max(abs(midpoint signed distance), length/2 * abs(sin(angle error))).
    residual = frame.distance_px - frame.oracle_midpoint_shift_px
    if residual.min() < -1e-5:
        raise ValueError(f'Endpoint-distance identity violated: {residual.min()}')
    frame['position_dominated'] = residual > 1e-5
    frame['angle_good'] = frame.angle_deg <= 5
    frame['angle_good_position_bad'] = frame.angle_good & (frame.distance_diagonal > .01)
    frame['joint_good'] = frame.angle_good & (frame.distance_diagonal <= .01)
    real = frame[frame.population == 'real_dev'].copy()
    sizes = real[['id', 'cuboid_diagonal']].drop_duplicates().copy()
    sizes['size_tertile'] = pd.qcut(sizes.cuboid_diagonal, q=3, labels=['small', 'medium', 'large']).astype(str)
    real = real.merge(sizes[['id', 'size_tertile']], on='id', validate='many_to_one')
    size_ranges = sizes.groupby('size_tertile').cuboid_diagonal.agg(['min', 'max', 'count']).to_dict('index')
    attention_rows, attention_aggregate = attention_diagnostics(run, records)
    paired_rows, paired_aggregate = paired_diagnostics(frame)
    report = {
        'scope': 'Saved CSV: all four evaluation populations and three paired seeds. Attention maps: only 12 predefined seed-1 examples (3 synth, 3 cross, 6 real). No retraining.',
        'aggregation': 'Each statistic computed within one seed, then averaged across seeds unless otherwise named. Pixels refer to original 640x480 images.',
        'thresholds': {'angle_good_deg': 5, 'position_good_fraction_image_diagonal': .01,
                       'position_good_px_for_640x480': 8},
        'oracle_midpoint_shift': 'GT-only non-deployable diagnostic: retain predicted angle and translate line through GT segment midpoint; residual is segment_length/2 * sin(angle error).',
        'cuboid_normalization': 'distance divided by diagonal of axis-aligned 2D bounds of valid projected GT cuboid corners; not visible mask diagonal.',
        'visibility_limit': 'geometry.supported means GT valid, segment intersects image, length>=2px; does not mean physically visible material edge. Visible-edge and hidden-edge accuracy cannot be separated from these labels.',
        'bootstrap_limit': 'Paired frame bootstrap averages three seed differences per frame; reflects sampled-frame uncertainty only, not sessions, seed population, or final test.',
        'population_summary': grouped_summary(frame, ['population', 'arm']),
        'real_by_role': grouped_summary(real, ['role', 'arm']),
        'real_by_edge_family': grouped_summary(real, ['edge_family', 'arm']),
        'real_by_axis_family': grouped_summary(real, ['axis_family', 'arm']),
        'real_by_group': grouped_summary(real, ['group', 'arm']),
        'real_by_size_tertile': grouped_summary(real, ['size_tertile', 'arm']),
        'real_size_ranges_px': size_ranges,
        'paired_frames': paired_aggregate,
        'attention_selected_examples': {'scope': '12 seed-1 samples only; head-averaged token mixing weights, not causal importance.',
                                        'rows': attention_rows, 'group_mean': attention_aggregate},
        'endpoint_distance_identity_min_residual_px': float(residual.min()),
    }
    (out / 'METRIC_DIAGNOSIS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    frame.to_csv(out / 'DIAGNOSTIC_PER_ROLE.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(out / 'PAIRED_FRAMES.csv', index=False)
    pd.DataFrame(attention_rows).to_csv(out / 'ATTENTION_SELECTED_EXAMPLES.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {'A': '#4267ac', 'B': '#db8043'}
    population_names = ['synth_test', 'cross_v4', 'real_dev']
    labels = ['Held-out synthetic', 'Other synthetic', 'Real DEV52']
    for arm, offset in [('A', -.17), ('B', .17)]:
        vals = [report['population_summary'][p][arm]['seed_mean']['distance_px_median'] for p in population_names]
        axes[0].bar(np.arange(3) + offset, vals, width=.32, color=colors[arm], label=arm)
        for x, y in zip(np.arange(3) + offset, vals): axes[0].text(x, y + .7, f'{y:.1f}', ha='center', fontsize=9)
        role_vals = [report['real_by_role'][str(r)][arm]['seed_mean']['distance_px_median'] for r in range(12)]
        axes[1].plot(np.arange(12), role_vals, 'o-', color=colors[arm], label=arm)
        att = pd.DataFrame(attention_rows)
        arm_att = att[att.arm == arm]
        axes[2].scatter(np.arange(len(arm_att)), arm_att.role_pair_cosine_mean, color=colors[arm], label=arm)
    axes[0].set_xticks(np.arange(3), labels, rotation=12)
    axes[0].set_ylabel('Endpoint-to-line distance median (px)')
    axes[0].set_title('Mean of three seed medians')
    axes[1].set_xticks(np.arange(12), [f'{a}-{b}' for a, b in EDGES], rotation=45)
    axes[1].set_ylabel('Real distance median (px)')
    axes[1].set_title('12 cuboid roles: near / far / depth')
    axes[1].axvline(3.5, color='.8'); axes[1].axvline(7.5, color='.8')
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel('Mean pairwise attention cosine')
    axes[2].set_xticks([1, 4, 8.5], ['3 synth', '3 cross', '6 real'])
    axes[2].axvline(2.5, color='.8'); axes[2].axvline(5.5, color='.8')
    axes[2].set_title('12 selected seed-1 samples only')
    for ax in axes:
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(out / 'metric_diagnosis.png', dpi=160)
    plt.close(fig)
    print(out / 'METRIC_DIAGNOSIS.json')
    for population in population_names:
        for arm in ('A', 'B'):
            row = report['population_summary'][population][arm]['seed_mean']
            print(population, arm, {k: round(row[k], 4) for k in ['distance_px_median', 'distance_cuboid_diagonal_median', 'position_dominated_fraction', 'angle_good_position_bad_fraction', 'joint_good_fraction', 'oracle_midpoint_shift_px_median', 'attention_effective_tokens_median']})
    print('Selected attention', attention_aggregate)
    print('Paired real', paired_aggregate['real_dev'])


if __name__ == '__main__':
    main()
