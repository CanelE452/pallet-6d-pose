"""Render only saved whole-U closure results and the original six RGB cases.

ROOT alone freezes/runs/views. Independent eight-method statistics PASS is
required before existing RGB decode. No pose, reference error, moment, CI,
observation, model or latency is recomputed. Proxy truth is not physical truth.
The V19 ready renderer supplies the fixed readable presentation/source lineage.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap
import time
import traceback

PRIVATE = Path('/tmp/pallet-whole-u-closure-private-20261011-v21')
HISTORY = Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
PRIMARY = 'HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
OLD = 'HYBRID_POINT_CONSENSUS_RESTORED'
HISTORY_NEW = ('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS', 'HYBRID_OBSERVED_FRAME_POINT_ONLY')
METHODS = ('BASE', 'N3_SUBPIX', 'HYBRID_STANDARD_PNP', 'NATIVE_STANDARD_PNP', OLD, *HISTORY_NEW, PRIMARY)
CONTRASTS = tuple((PRIMARY, m) for m in METHODS[:-1])
DISPLAY = ('N3_SUBPIX', HISTORY_NEW[1], PRIMARY)
LABELS = dict(zip(METHODS, ('Base', 'N3→cornerSubPix', 'Hybrid ordinary', 'Native ordinary',
    'V18 point restored', 'V19 observed-frame point + line', 'V19 same-q point only', 'V21 whole-U point closure')))
FIELDS = ('translation_cm', 'rotation_deg', 'ADDsym_cm'); UNITS = ('cm', 'degree', 'cm')
MASKS = ('MATCHES_ON_KNOWN', 'WRONG_ON_KNOWN', 'NO_KNOWN_VISIBILITY', 'MASK_NOT_APPLIED')
OUTCOMES = ('BOTH_BETTER', 'BOTH_WORSE', 'MIXED_OR_UNCHANGED', 'POSE_UNAVAILABLE')
STATS_SHA = '9ea0be750a7f921bb572801b48953b9530f6a34799a368dc0507c0e34f9cf0d2'
LAYOUT_SHA = '1f4430bc7ed2906a59b2cbdc42a733508b7abad3a6863c27a537d74aced55715'
NAMES = ('FIGURE_PROTOCOL.json', 'FIGURES_STARTED.json', 'FIGURE_BINDINGS.json', 'FIGURES_FAILURE.json',
    'FIGURE_CASE_ROWS.jsonl.gz', 'FIGURE_DIAGNOSTIC_COUNTS.json')
LIMITS = dict(new_pose_GT_error_moment_CI_model_training_calls=0, cases_selected_from_V21_results=False,
    reference='saved N3-phase GEOMETRIC_PROXY; not independently measured physical R/t',
    physical_correspondence_ownership='UNKNOWN', old_full_V16_564_FAIL_remains_FAIL=True,
    fixed_N3_H_and_fit_metadata_unrecorded=True, fallback_preserves_original_full_N3=True,
    visible_damage_is_saved_proxy_error_increase=True, diagnostic_moments_copied_not_independently_recomputed=True,
    latency_not_plotted_or_inferred_from_historical_V19=True, ROOT_visual_review_claimed=False)

def require(value, message):
    if not value: raise ValueError(message)

def read(path): return json.loads(Path(path).read_bytes())

def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())

def same(a, b): return all(a.get(k) == b.get(k) for k in ('bytes', 'sha256'))

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream: require(bool(line.strip()), 'blank saved row'); yield json.loads(line)

def metadata_helpers(path):
    require(binding(path)['sha256'] == STATS_SHA, 'exact immutable independent statistics checker')
    tree = ast.parse(path.read_bytes()); canonical = ast.dump(tree, include_attributes=False)
    env = dict(__name__='_v21_figure_metadata_only', __file__=str(path), __package__=None)
    exec(compile(tree, str(path) + ':metadata-only', 'exec'), env)
    require(ast.dump(tree, include_attributes=False) == canonical, 'whole checker AST unchanged')
    return env

def inputs(a):
    checker = a.stats_checker.absolute(); h = metadata_helpers(checker); f = h['paths'](a)
    b = {k: binding(p) for k, p in f.items()}; m = h['metadata'](f, b)
    sp, ss, sc = [a.stats_input / n for n in ('STATISTICS_CHECK_PROTOCOL.json', 'STATISTICS_CHECK_STARTED.json', 'STATISTICS_CHECKS.json')]
    p, s, c = [read(v) for v in (sp, ss, sc)]
    require(p['schema'] == 'independent_whole_U_closure_statistics_protocol_v21'
        and c['schema'] == 'independent_whole_U_closure_statistics_checks_v21'
        and c['complete'] is True and c['passed'] is True and c['error'] is None
        and c['inputs'] == p['inputs'] == s['inputs'] == b and c['old_full_V16_validation_passed'] is False
        and c['new_production_calls'] == s['new_production_calls'] == 0, 'actual same-input independent statistics PASS')
    for item in (c['own_protocol'], s['own_protocol']): require(same(binding(sp), item), 'actual statistics protocol bytes')
    for item in (p['checker'], s['checker'], c['checker']): require(same(binding(checker), item), 'current statistics checker')
    require(c['source_witness'] == p['source_witness'] == s['source_witness'] == h['extraction'](f), 'unchanged isolated statistics source proof')
    expected = dict(frames=245, methods=8, saved_error_rows_streamed=2205, saved_error_rows_selected=1960,
        moment_distributions_checked=783, moment_scalars_checked=6264, paired_groups_checked=63, metric_CI_slots_checked=189)
    require(all(c['actual_counts'][k] == v for k, v in expected.items()), 'full eight-method statistics coverage')
    f.update(renderer=Path(__file__), statistics_checker=checker, statistics_protocol=sp, statistics_started=ss, statistics_checks=sc,
        readable_layout_source=HISTORY / 'render_results_ready.py',
        case_protocol=a.repo / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json',
        posthoc=a.input / 'POSTHOC_ROWS.jsonl.gz')
    require(binding(f['readable_layout_source'])['sha256'] == LAYOUT_SHA, 'unchanged V19 readable layout lineage')
    require(same(binding(f['posthoc']), m['posthoc_rows']), 'actual new245 posthoc bytes')
    case, cohort = read(f['case_protocol']), read(f['cohort'])
    require(same(binding(f['cohort']), case['cohort']), 'original six-case cohort bytes')
    cases = case['cases']; by_id = {r['id']: r for r in cohort['frames']}
    require(len(cases) == len({r['id'] for r in cases}) == 6 and len(cohort['ids']) == len(by_id) == 245
        and Counter(r['label'] for r in cases) == Counter(clean=3, moderate=3), 'fixed three Clean/three Moderate cases; full245')
    for i, r in enumerate(cases):
        require(r['id'] in by_id and all(r[k] == by_id[r['id']][k] for k in ('session', 'label', 'image')), 'original case metadata')
        relative = Path(r['image']['path']); require(not relative.is_absolute() and '..' not in relative.parts, 'safe existing RGB path')
        f['image:' + str(i)] = a.source_root / relative
        require(same(binding(f['image:' + str(i)]), r['image']), 'fixed original RGB bytes')
    return f, {k: binding(p) for k, p in f.items()}, m, cohort, cases

def point_valid(p): return isinstance(p, list) and len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p) and p != [-1, -1]

def display(row):
    solver = row.get('solver') or {}; new = bool(row['new_pose_estimated']); fixed = row['method'] in METHODS[:2]
    return dict(id=row['id'], method=row['method'], native_points=row['native_points'], input_points=row.get('input_points'),
        pose=row['pose'], new_pose_estimated=new, fallback_used=row['fallback_used'], output_status=row['output_status'],
        applied_H=None if fixed else row.get('hidden_initial'), reprojected_ids=row.get('reprojected_ids', []),
        solver_state=solver.get('state'), accepted_fit_ids=None if fixed else solver.get('fit_input_ids', []) if new else [],
        actual_final_inliers=None if fixed else solver.get('final_inliers', []) if new else [],
        selected_corner_ids=row.get('selected_corner_ids'), accepted_fit_edges=None if fixed else solver.get('fit_line_edges', []) if new else [],
        fit_metadata_scope='UNRECORDED_FIXED_CONTROL_METADATA' if fixed else 'ACCEPTED_NEW_ONLY_NOT_INITIAL_FALLBACK_FIT',
        evaluation_reference=row.get('evaluation_reference', {}), physical_ownership='UNKNOWN')

def saved_data(f, ids, cases):
    groups = {m: {} for m in METHODS}; views = {}; wanted = {r['id'] for r in cases}; total = 0
    streams = (('fixed_scores', METHODS[:2]), ('ordinary_scores', METHODS[2:4]),
        ('point_scores', ('HYBRID_POINT_CONSENSUS_ORIGINAL', OLD)), ('history_scores', HISTORY_NEW), ('new_scores', (PRIMARY,)))
    for key, allowed in streams:
        seen = set()
        for row in rows(f[key]):
            identity = (row['method'], row['id']); total += 1
            require(row['method'] in allowed and row['id'] in ids and identity not in seen, 'unique source method/ID'); seen.add(identity)
            if row['method'] not in groups: continue
            pose = row['pose']; available = pose['available']
            values = {k: float(pose['ADDsym_m']) * 100 if k == 'ADDsym_cm' else float(pose[k]) for k in FIELDS} if available else {}
            require(all(math.isfinite(v) for v in values.values()), 'finite saved errors')
            groups[row['method']][row['id']] = dict(values=values, available=available, new=row['new_pose_estimated'], fallback=row['fallback_used'], status=row['output_status'])
            if row['method'] in DISPLAY and row['id'] in wanted: views[identity] = display(row)
        require(seen == {(m, i) for m in allowed for i in ids}, 'full fixed source population: ' + key)
    require(total == 2205 and all(set(g) == set(ids) for g in groups.values()) and len(views) == 18, 'eight full245 methods; original245 excluded;18 panels')
    post = {}
    for r in rows(f['posthoc']):
        require(r['method'] == PRIMARY and r['id'] in ids and r['id'] not in post, 'unique new245 posthoc'); post[r['id']] = r
    require(set(post) == set(ids), 'all245 new posthoc frames')
    return groups, views, post

def diagnostic_counts(post, m, ids):
    groups = {}; saved = m['strata']['combined']['diagnostics'][PRIMARY]
    for mask in MASKS:
        for outcome in OUTCOMES:
            chosen = [i for i in ids if post[i]['mask_state'] == mask and post[i]['pose_outcome_vs_N3'] == outcome]
            value = dict(n=len(chosen), ids=chosen, new_pose=sum(post[i]['new_pose_estimated'] for i in chosen), fallback=sum(post[i]['fallback_used'] for i in chosen))
            require(value == saved['mask_and_pose_groups_vs_fixed_N3'][mask + '__' + outcome], 'saved exact mask/outcome group')
            groups[mask + '__' + outcome] = value
    require(sum(v['n'] for v in groups.values()) == 245, 'full mask/outcome partition')
    direct = [(i, c['id']) for i in ids for c in post[i]['corners'] if c['human_state'] == 'DIRECT_VISIBLE']
    damage = [(i, c['id']) for i in ids for c in post[i]['corners'] if c['directly_visible_damage_5_to_10']]
    actual = [(i, c['id']) for i in ids for c in post[i]['corners'] if c['actually_reprojected']]
    require(len(direct) == saved['DIRECT_VISIBLE']['corners'] and len(damage) == saved['DIRECT_VISIBLE']['damage_5px_to_over10px']
        and len(actual) == saved['ACTUALLY_REPROJECTED']['corners'], 'saved direct damage and actual H counts')
    return dict(mask_and_pose_groups_vs_fixed_N3=groups, directly_visible_frame_corner_ids=direct,
        direct_damage_frame_corner_ids=damage, actually_reprojected_frame_corner_ids=actual,
        saved_corner_error_moments={k: saved[k] for k in ('DIRECT_VISIBLE', 'SELF_OCCLUDED', 'ACTUALLY_REPROJECTED')},
        reference='saved GEOMETRIC_PROXY visibility/error; physical ownership UNKNOWN')

def run(a, f, b, m, cohort, cases, out):
    lock = read(out / NAMES[0]); require(lock['inputs'] == b and lock['limits'] == LIMITS
        and lock['methods'] == list(METHODS) and lock['contrasts'] == [list(pair) for pair in CONTRASTS]
        and lock['case_ids'] == [r['id'] for r in cases], 'own frozen render inputs/methods/cases')
    for n in NAMES[1:]: require(not (out / n).exists(), 'preserve ' + n)
    folder = out / 'figures'; require(not folder.exists(), 'preserve figures')
    write(out / NAMES[1], dict(protocol=binding(out / NAMES[0]), inputs=b, configured_PNGs=9, configured_case_panels=18, new_production_calls=0))
    began = time.monotonic(); case_rows = []
    try:
        data, views, post = saved_data(f, cohort['ids'], cases); diagnostic = diagnostic_counts(post, m, cohort['ids']); ids = cohort['ids']; folder.mkdir()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        fig, axes = plt.subplots(1, 3, figsize=(22, 7), layout='constrained')
        for ax, field, unit in zip(axes, FIELDS, UNITS):
            for j, method in enumerate(METHODS):
                values = [data[method][i]['values'][field] for i in ids if data[method][i]['available']]
                saved = m['strata']['combined']['methods'][method]; moment = saved['metrics']['operational'][field]
                require(len(values) == moment['n'] == saved['operational'], 'full operational denominator')
                ax.scatter(values, [j] * len(values), s=6, alpha=.25)
                if values: ax.plot([moment['median'], moment['P90']], [j, j], linewidth=3); ax.scatter([moment['mean']], [j], marker='D', facecolor='white', edgecolor='black')
            ax.set_yticks(range(8), [LABELS[k] + ' (op ' + str(m['strata']['combined']['methods'][k]['operational']) + '/245, NEW ' + str(m['strata']['combined']['methods'][k]['new_pose']) + ')' for k in METHODS])
            ax.invert_yaxis(); ax.set_xscale('symlog', linthresh=1); ax.set_xlabel(field + ' [' + unit + ']'); ax.grid(alpha=.2)
        fig.suptitle('All245 operational outputs including original N3 fallback · saved GEOMETRIC_PROXY\nDiamond=saved mean; segment=saved median–P90. Numerical NEW does not certify physical accuracy.')
        fig.savefig(folder / '01_all245_operational.png', dpi=140); plt.close(fig)
        fig, axes = plt.subplots(4, 2, figsize=(21, 23), layout='constrained')
        for ax, (candidate, comparator) in zip(axes.flat, CONTRASTS):
            comparison = m['strata']['combined']['contrasts'][candidate + '_minus_' + comparator]; common = comparison['common_operational']; pairs = common['pair_ids']
            for j, (field, unit) in enumerate(zip(FIELDS, UNITS)):
                delta = [data[candidate][i]['values'][field] - data[comparator][i]['values'][field] for i in pairs]; saved = common['metrics'][field]
                require(len(delta) == saved['n'], 'same-ID paired denominator'); ax.scatter(delta, [j] * len(delta), s=8, alpha=.3)
                if delta: ax.plot(saved['CI95'], [j, j], linewidth=4); ax.scatter([saved['mean_delta']], [j], marker='D', edgecolor='black', color='white')
            ax.axvline(0, color='black', linestyle='--'); ax.set_xscale('symlog', linthresh=1); ax.set_yticks(range(3), [k + ' [' + u + ']' for k, u in zip(FIELDS, UNITS)]); ax.grid(alpha=.2)
            ax.set_title(LABELS[candidate] + ' − ' + LABELS[comparator] + '\ncommon n=' + str(len(pairs)) + '; candidate NEW n=' + str(comparison['candidate_new_pose']['common_frames']) + '\n' + comparison['both_new_pose']['scope'] + ' n=' + str(comparison['both_new_pose']['common_frames']), fontsize=10)
            ax.set_xlabel('saved-error difference; below zero favors closure; diamond=mean, segment=existing session CI')
        axes.flat[-1].axis('off'); fig.suptitle('Seven fixed same-frame contrasts · cm / degree as marked\nSaved paired errors and CIs; no new draw, fit or outcome-selected case')
        fig.savefig(folder / '02_same_ID_paired_errors.png', dpi=140); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(22, 7), layout='constrained'); ax = axes[0]; bottom = [0] * 4
        for outcome, color in zip(OUTCOMES, ('seagreen', 'firebrick', 'goldenrod', 'gray')):
            values = [diagnostic['mask_and_pose_groups_vs_fixed_N3'][mask + '__' + outcome]['n'] for mask in MASKS]
            ax.bar(range(4), values, bottom=bottom, color=color, label=outcome); bottom = [x + y for x, y in zip(bottom, values)]
        ax.set_xticks(range(4), [k.replace('_', '\n') for k in MASKS], fontsize=8); ax.set_ylabel('all245 frames'); ax.set_title('Mask declaration × T/R outcome vs N3'); ax.legend(fontsize=7)
        for ax, category in zip(axes[1:], ('DIRECT_VISIBLE', 'ACTUALLY_REPROJECTED')):
            saved = diagnostic['saved_corner_error_moments'][category]
            for j, source in enumerate(('N3', 'output')):
                value = saved['errors'][source]
                if value['n']: ax.bar(j, value['mean'], color=('gray', 'royalblue')[j]); ax.plot([j, j], [value['median'], value['P90']], color='black', linewidth=3)
                ax.text(j, value['mean'] or 0, 'n=' + str(value['n']), ha='center', fontsize=8)
            ax.set_xticks(range(2), ('original N3', 'output')); ax.set_ylabel('saved proxy corner error [px]'); ax.set_title(category + '\nbar=saved mean; segment=median–P90')
        axes[1].text(.02, .98, 'visible error ≤5px → >10px flags: ' + str(len(diagnostic['direct_damage_frame_corner_ids'])), transform=axes[1].transAxes, va='top', fontsize=8)
        fig.suptitle('Mask disagreement is separate from actual saved pose errors\nDirectly visible damage and actual H reprojection error are separate; ownership/physical truth remain UNKNOWN.')
        fig.savefig(folder / '03_mask_and_visible_error.png', dpi=140); plt.close(fig)
        for ci, case in enumerate(cases):
            with Image.open(f['image:' + str(ci)]) as image: rgb = image.convert('RGB').copy()
            fig, axes = plt.subplots(1, 3, figsize=(24, 11)); fig.subplots_adjust(wspace=.12, top=.62, bottom=.04, left=.015, right=.985)
            for ax, method in zip(axes, DISPLAY):
                r = views[method, case['id']]; points = r['native_points']; ax.imshow(rgb)
                for k, p in enumerate(points[:8]):
                    if point_valid(p): ax.scatter(*p, s=34, facecolor='none', edgecolor='cyan'); ax.text(p[0] + 3, p[1] + 3, str(k), fontsize=8, color='cyan')
                for k in r['reprojected_ids']:
                    require(r['new_pose_estimated'] and k in (r['applied_H'] or []), 'approved H-only display projection'); ax.scatter(*points[k], s=80, marker='s', facecolor='none', edgecolor='orange')
                for p in (r['input_points'] or [])[:8]:
                    if point_valid(p): ax.scatter(*p, s=32, marker='+', color='#43ff50')
                ref = r['evaluation_reference']; xy = ref.get('native_points_px', [])
                for k in ref.get('valid_native_ids', []):
                    if k < len(xy) and point_valid(xy[k]): ax.scatter(*xy[k], s=14, marker='.', color='white')
                if method == PRIMARY:
                    for c in post[case['id']]['corners']:
                        if c['directly_visible_damage_5_to_10']: ax.scatter(*points[c['id']], s=85, marker='x', color='red')
                pose = r['pose']; shown = lambda v: 'None' if v is None else format(v, '.3f')
                recorded = lambda v: 'unrecorded' if v is None else '[' + ','.join(str(k) for k in v) + ']'
                wrap = lambda v: textwrap.fill(v, width=42, break_long_words=True, break_on_hyphens=False)
                text = LABELS[method] + '\n' + wrap('status: ' + r['output_status']) + '\n' + wrap('state: ' + str(r['solver_state']))
                text += '\nT=' + shown(pose.get('translation_cm')) + ' cm | R=' + shown(pose.get('rotation_deg')) + ' deg\nADDsym=' + shown(data[method][case['id']]['values'].get('ADDsym_cm')) + ' cm'
                text += '\nfit U=' + recorded(r['accepted_fit_ids']) + '\ninliers8px=' + recorded(r['actual_final_inliers']) + '\nfit edges=' + recorded(r['accepted_fit_edges'])
                text += '\n' + wrap('H(pred)=' + recorded(r['applied_H']) + '; adopted=' + recorded(r['selected_corner_ids']))
                if method == PRIMARY: text += '\n' + wrap(post[case['id']]['mask_state']) + '\n' + wrap(post[case['id']]['pose_outcome_vs_N3'])
                ax.set_title(text, fontsize=8.5, pad=10, linespacing=1.15); ax.axis('off')
                case_rows.append(dict(case_index=ci, id=case['id'], method=method, display=r, posthoc=post[case['id']] if method == PRIMARY else None))
            fig.suptitle(case['label'] + ' / ' + case['id'] + ' · original fixed-case ID\ncyan=output; green+=supplied input q; orange square=one approved H reprojection; white=saved proxy XY; red×=saved visible proxy-error increase\nPoint fits exclude initial H and center; H is reprojected once after NEW and never refitted. Fallback preserves full original N3.\nN3 H/fit are unrecorded fixed metadata. Physical correspondence ownership and global root uniqueness remain UNKNOWN.', fontsize=11)
            fig.savefig(folder / ('case_%02d.png' % (ci + 1)), dpi=140); plt.close(fig)
        require(len(case_rows) == 18 and len(list(folder.glob('*.png'))) == 9, 'actual planned9 PNG/18 panels')
        with (out / NAMES[4]).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as z:
                for row in case_rows: z.write((json.dumps(row, allow_nan=False) + '\n').encode())
            raw.flush(); os.fsync(raw.fileno())
        write(out / NAMES[5], dict(complete=True, method=PRIMARY, diagnostics=diagnostic, limits=LIMITS, saved_moments_or_CIs_recomputed=False))
        require(inputs(a)[1] == b, 'render inputs changed')
        write(out / NAMES[2], dict(schema='whole_U_closure_results_figure_bindings_v21', complete=True, inputs=b,
            own_protocol=binding(out / NAMES[0]), renderer=b['renderer'], figures=[binding(p) for p in sorted(folder.glob('*.png'))],
            case_rows=binding(out / NAMES[4]), diagnostic_counts=binding(out / NAMES[5]),
            actual_counts=dict(PNG_files=9, case_panels=18, RGB_images_read=6, full_selected_score_rows=1960, source_score_rows=2205, posthoc_rows=245),
            elapsed_seconds=time.monotonic() - began, elapsed_is_inference_latency=False, limits=LIMITS))
    except BaseException as exc:
        write(out / NAMES[3], dict(complete=False, error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()),
            preserved_PNG_prefix=[binding(p) for p in sorted(folder.glob('*.png'))], completed_case_panels=len(case_rows), automatic_retry=False, limits=LIMITS)); raise

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('freeze', 'run'))
    for name in ('repo', 'input', 'source-root', 'stats-input', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--stats-checker', type=Path, default=Path(__file__).with_name('stats_check.py')); a = p.parse_args()
    for name in ('repo', 'input', 'source_root', 'stats_input', 'stats_checker', 'output'): setattr(a, name, getattr(a, name).absolute())
    out = a.output; require(out != a.input and out != PRIVATE and out.is_relative_to(PRIVATE)
        and not any(q.is_symlink() for q in (out, *out.parents)), 'new nonsymlink private V21 output')
    out.mkdir(parents=True, exist_ok=True)
    for n in NAMES if a.stage == 'freeze' else NAMES[1:]: require(not (out / n).exists(), 'preserve ' + n)
    f, b, m, cohort, cases = inputs(a); require(not any(v.absolute().is_relative_to(out) for v in f.values()), 'output overlaps inputs')
    if a.stage == 'freeze':
        write(out / NAMES[0], dict(schema='prespecified_whole_U_closure_results_figures_v21', inputs=b, limits=LIMITS,
            case_ids=[r['id'] for r in cases], methods=list(METHODS), contrasts=[list(pair) for pair in CONTRASTS], display_methods=list(DISPLAY),
            planned_PNGs=9, planned_case_panels=18, frames=245, full_selected_score_rows=1960, source_score_rows=2205,
            presentation=dict(source=b['readable_layout_source'], numeric_display_decimal_places=3, display_ADDsym_unit='cm',
                panel_size_inches=[24, 11], panel_top=.62, caption_wrap_characters=42, stored_full_precision_case_rows_preserved=True),
            frozen_after_own_full_geometry_and_statistics_PASS_before_image_decode=True))
    else: run(a, f, b, m, cohort, cases, out)

if __name__ == '__main__': main()
