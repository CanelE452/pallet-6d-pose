"""Actual saved stages after prediction/selection/scoring seals; no inference.

This module reads sealed numeric metrics, never annotation or reference files.
Only image permission inventory, font and cuboid edges are reused from the
previous gallery. Model projections always use the registered CF basis.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.research.pallet_minimal_bi_ablation_v1.gallery import (
    EDGES, FONT, public_image_inventory,
)


SCHEMA = 'GUARDED_BI_ACTUAL_STAGE_GALLERY_V1'
PUBLIC_INVENTORY = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/'
    'publication_followup_20261015_v11/repro_inventory_current_files_v2.jsonl.gz')
PUBLIC_INVENTORY_SHA256 = '75c5aaa33d534faefc028c3358d9be6b5cbe688599488a29d34478502fac47b5'
BRANCH = {'R': 'RAW_YOLO', 'N': 'N3_SUBPIX', 'E': 'E_FROZEN'}
COLOR = {'B': '#00a3c7', 'I': '#8847ce'}
RAY_SOURCES = {'OBSERVED_TARGET_BACKGROUND_TRANSITION',
               'OBSERVED_INTERNAL_RGB_GRADIENT',
               'OBSERVED_EXTERIOR_MASK_AND_RGB_TRANSITION'}
STYLE = ('body{font-family:sans-serif;max-width:1600px;margin:20px auto;padding:18px;'
         'line-height:1.65;background:#eef2f6;color:#172633}article,section{background:white;'
         'padding:18px;margin:14px 0;border-radius:10px}.tiles{display:grid;'
         'grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}img{width:100%;height:auto}'
         'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:6px;'
         'border-bottom:1px solid #ccc;text-align:left;overflow-wrap:anywhere}'
         '@media(max-width:850px){.tiles{grid-template-columns:1fr}}')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt') as stream:
        yield from map(json.loads, stream)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _cube(extents):
    a, b, c = np.asarray(extents, float) / 2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def projection(frame, pose):
    """Physical vertex IDs are already remapped in each own input frame."""
    if not isinstance(pose, dict) or pose.get('available', True) is False:
        return None
    try:
        X = _cube(pose['cf_extents'])
        declared = np.asarray(frame['xyz'], float)
        R, t, K = np.asarray(pose['R_cf'], float), np.asarray(pose['centroid'], float), np.asarray(frame['K'], float)
        # Frame xyz is the fixed physical dimensions. CF dimensions may be
        # W/D-swapped; authenticate the physical-vertex set through its own Q.
        physical_match = True
        if declared.shape == (3,):
            Q = R.T @ np.asarray(pose['R_physical'], float)
            distances = np.linalg.norm((X @ Q)[:,None,:] - _cube(declared)[None,:,:], axis=2)
            match = distances <= 1e-8  # Existing physical-vertex registration tolerance.
            physical_match = bool(np.all(match.sum(axis=0) == 1) and np.all(match.sum(axis=1) == 1))
        if ((declared.shape == (3,) and not physical_match)
                or (declared.shape == (8,3) and not np.array_equal(X, declared))
                or declared.shape not in ((3,), (8,3))):
            raise ValueError('Gallery native physical vertex/CF basis changed')
        if R.shape != (3, 3) or t.shape != (3,) or K.shape != (3, 3):
            raise ValueError('Invalid saved pose geometry')
        camera = X @ R.T + t
        homogeneous = camera @ K.T
        if not np.isfinite(homogeneous).all() or np.any(camera[:, 2] <= 0) or np.any(homogeneous[:, 2] <= 0):
            return None
        projected = homogeneous[:, :2] / homogeneous[:, 2, None]
        return projected if np.isfinite(projected).all() else None
    except KeyError:
        return None


def clip_segment(first, second, width, height):
    """Liang-Barsky clipping; never expand the actual RGB canvas for a pose."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != (2,) or second.shape != (2,) or not np.isfinite([first, second]).all():
        return None
    direction = second - first
    low, high = 0., 1.
    for p, q in ((-direction[0], first[0]), (direction[0], width - 1. - first[0]),
                 (-direction[1], first[1]), (direction[1], height - 1. - first[1])):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            low = max(low, q / p)
        else:
            high = min(high, q / p)
        if low > high:
            return None
    return np.array([first + low * direction, first + high * direction])


def select_cases(scored_rows, selected_arm):
    primary = sorted((r for r in scored_rows if r['partition'] != 'CAL'), key=lambda r: r['id'])
    chosen, categories = {}, []

    def add(category, fid, prefix):
        categories.append(dict(category=category, id=fid, prefix=prefix, absent=fid is None))
        if fid is not None:
            entry = chosen.setdefault(fid, dict(id=fid, prefix=prefix, categories=[]))
            entry['categories'].append(category)

    target = next((r['id'] for r in primary if r['id'] == 'wood_night_01:029428'), None)
    add('명시된 P3P 실패 사례', target, selected_arm[0])
    available = [r for r in primary if r['metrics'][selected_arm]['translation_cm'] is not None]
    maximum = min(available, key=lambda r: (-r['metrics'][selected_arm]['translation_cm'], r['id'])) if available else None
    add('CAL 선택 경로의 PRIMARY 최대 위치오차', None if maximum is None else maximum['id'], selected_arm[0])
    for arm, baseline in (('EA', 'E0'), ('EB', 'E0'), ('NB', 'N0')):
        for category, before, after in (('회복', False, True), ('손상', True, False)):
            fid = next((r['id'] for r in primary
                        if r['metrics'][baseline]['success_le5'] is before
                        and r['metrics'][arm]['success_le5'] is after), None)
            add(f'{arm} vs {baseline} {category} · ID순 첫 사례', fid, arm[0])
    return dict(schema=SCHEMA, selected_arm=selected_arm, maximum_cases=8,
                rule='fixed requested failure, selected maximum T, first sorted EA/EB/NB recovery and damage; deduplicated',
                categories=categories, cases=list(chosen.values()),
                absent_categories=[r['category'] for r in categories if r['absent']],
                selection_is_postscore_diagnostic_only=True, inference_effect=0)


def _input_binding(name, folder, prior):
    path = Path(name).resolve()
    for prefix, base in (('FOLLOWUP', folder), ('PRIOR', prior)):
        if path.is_relative_to(base.resolve()):
            return prefix + '/' + str(path.relative_to(base.resolve()))
    raise ValueError('Gallery input binding outside the two authenticated run folders')


def _metric_text(metric):
    if metric['translation_cm'] is None:
        return 'pose 없음 · 전체 분모 실패'
    success = '성공' if metric['success_le5'] is True else '실패' if metric['success_le5'] is False else '회전 참조 미확정'
    rotation = 'NA' if metric['rotation_deg'] is None else f'{metric["rotation_deg"]:.3f}'
    return f'T {metric["translation_cm"]:.3f}cm / R {rotation}° · {success}'


def _supported_rays(obs):
    accepted, excluded = [], []
    X = np.asarray(obs['initial_geometry']['native_xyz'], float)
    for record in obs['visible_correction']['records']:
        k = record['native_corner']
        for index, ray in enumerate(record.get('rays', [])):
            audit = dict(native_corner=k, index=index, native_neighbor=ray.get('native_neighbor'), source=ray.get('measurement_source'))
            try:
                j = ray['native_neighbor']
                points = np.asarray(ray.get('observed_points', []), float)
                normal = np.asarray(ray.get('normal', []), float)
                good = (obs['states'][k] in ('B', 'I') and record.get('pixel_queried') is True
                    and ray.get('supported') is True and ray.get('measurement_source') in RAY_SOURCES
                    and isinstance(j, int) and not isinstance(j, bool) and 0 <= j < 8
                    and ray.get('native_corner') == k and ray.get('endpoints') == [k, j]
                    and np.count_nonzero(X[k] != X[j]) == 1
                    and points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2
                    and normal.shape == (2,) and np.isfinite(points).all() and np.isfinite(normal).all()
                    and np.linalg.norm(normal) > 0 and np.isfinite(ray.get('offset', np.nan)))
            except (KeyError, TypeError, ValueError, IndexError):
                good = False
            if good:
                accepted.append((k, deepcopy(ray)))
            else:
                audit['reason'] = 'unsupported_nonimage_nonBI_or_invalid_native_line_not_drawn'
                excluded.append(audit)
    return accepted, excluded


def _draw(frame, prediction, obs, solved, guard_audits, score, case, folder, source_root):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager, patheffects
    from matplotlib.lines import Line2D
    from PIL import Image

    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                         'axes.unicode_minus': False})
    source = frame['arms'][BRANCH[case['prefix']]]
    prefix = case['prefix']
    states = obs['states']
    path = Path(source_root) / frame['image']
    if sha(path) != frame['image_sha256']:
        raise ValueError('Actual gallery source RGB changed')
    image = np.asarray(Image.open(path).convert('RGB'))
    if list(image.shape[:2]) != frame['raw_hw']:
        raise ValueError('Original RGB canvas changed')
    height, width = image.shape[:2]
    folder.mkdir()
    shutil.copyfile(path, folder / ('source_rgb' + path.suffix.lower()))
    projections = {suffix: projection(frame, prediction['predictions'][prefix + suffix]['actual_pose'])
                   for suffix in ('0', '1', 'A', 'B')}
    own_initial = projection(frame, source['coarse_pose'])
    if (own_initial is None) != (projections['0'] is None) or (own_initial is not None
            and not np.array_equal(own_initial, projections['0'])):
        raise ValueError('Own baseline projection does not match the original input')
    raw = np.asarray(source['native_points'], float)
    support = np.asarray(source['prediction_support'], bool)
    measurements = obs['frozen_visible_measurements']
    if any(states[m['native_corner']] not in ('B', 'I') for m in measurements):
        raise ValueError('S/O/T/U supplied as actual image measurement')
    rays, excluded_rays = _supported_rays(obs)
    proof = dict(id=frame['id'], prefix=prefix, categories=case['categories'],
        image_sha256=frame['image_sha256'], viewport=[0, width - 1, 0, height - 1],
        same_full_RGB_canvas_every_panel=True, model_basis='own CF native cuboid + R_cf + centroid',
        physical_vertex_mapping=source.get('native_vertex_mapping'),
        initial_model_projection_xy=None if own_initial is None else own_initial.tolist(),
        model_projection_xy={k: None if v is None else v.tolist() for k, v in projections.items()},
        states=states, displayed_native_predictions=[dict(native_corner=k, xy=raw[k].tolist())
            for k, state in enumerate(states) if state in ('B','I') and support[k]],
        actual_BI_measurements=deepcopy(measurements),
        supported_observed_rays=[dict(r, native_corner=k) for k, r in rays],
        excluded_rays=excluded_rays, metrics={prefix+s: score['metrics'][prefix+s] for s in ('0','1','A','B')},
        guard_decisions={m: {key:guard_audits[prefix+m]['result'][key]
            for key in ('accepted','fallback','status','reason','decision_audit')} for m in ('A','B')},
        original_solver=dict(accepted=solved.get('accepted'), fallback=solved.get('fallback'),
            status=(solved.get('solution') or {}).get('status'),
            fallback_reason=(solved.get('solution') or {}).get('fallback_reason')),
        S_O_T_U_actual_measurement_markers=0, GT_wire_drawn=False,
        intermediate_pose_trajectory_created=False, new_NN_PnP_or_image_measurement_calls=0,
        model_clipping={}, tiles=[])

    def canvas(title):
        fig, ax = plt.subplots(figsize=(10.4, 8.2), layout='constrained')
        ax.imshow(image, interpolation='nearest')
        ax.set_xlim(-.5, width - .5)
        ax.set_ylim(height - .5, -.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('원영상 x [px]')
        ax.set_ylabel('원영상 y [px]')
        return fig, ax

    def point(ax, xy, label, color, marker='o', size=40):
        if not np.isfinite(xy).all() or not (0 <= xy[0] < width and 0 <= xy[1] < height):
            return
        ax.scatter(*xy, color=color, marker=marker, s=size, zorder=7)
        tag = ax.annotate(label, xy, xytext=(5, -12 if int(label.split(':')[0]) % 2 else 9),
            textcoords='offset points', color=color, fontsize=10, weight='bold', clip_on=True)
        tag.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])

    def wire(ax, coordinates, color, label, token, linestyle='-', linewidth=1.7):
        info = dict(outside_native_vertices=0, clipped_edges=0, invisible_edges=0,
                    disclosure='Full original canvas; off-canvas model segments are clipped, never rescaled')
        proof['model_clipping'][token] = info
        if coordinates is None:
            info['unprojectable_model'] = True
            return
        info['outside_native_vertices'] = int(np.sum((coordinates[:,0] < 0) | (coordinates[:,0] >= width)
            | (coordinates[:,1] < 0) | (coordinates[:,1] >= height)))
        labelled = False
        for first, second in EDGES:
            original = coordinates[[first, second]]
            clipped = clip_segment(*original, width, height)
            if clipped is None:
                info['invisible_edges'] += 1
                continue
            if not np.array_equal(clipped, original):
                info['clipped_edges'] += 1
            ax.plot(clipped[:,0], clipped[:,1], color=color, lw=linewidth, ls=linestyle,
                    label=label if not labelled else None)
            labelled = True

    def save(fig, ax, name):
        handles, _ = ax.get_legend_handles_labels()
        if handles and ax.get_legend() is None:
            ax.legend(fontsize=9, loc='upper right')
        if any(v.get('outside_native_vertices', 0) for v in proof['model_clipping'].values()):
            ax.text(.01, .01, '화면 밖 모델 선은 원영상 경계에서 잘라 표시', transform=ax.transAxes,
                    fontsize=9, color='#222', bbox=dict(facecolor='white', alpha=.85))
        fig.savefig(folder / name, dpi=130)
        plt.close(fig)
        proof['tiles'].append(name)

    fig, ax = canvas('① ' + prefix + '0 · 자기 초기 입력과 모델\n' + _metric_text(score['metrics'][prefix+'0']))
    wire(ax, own_initial, '#8055b3', '초기 모델 투영', 'panel1_initial')
    for k, state in enumerate(states):
        if state in ('B', 'I') and support[k] and np.isfinite(raw[k]).all() and not np.all(raw[k] == -1):
            point(ax, raw[k], str(k), '#de663c', 'x')
    ax.legend(handles=[Line2D([], [], color='#8055b3', label='초기 모델 투영'),
        Line2D([], [], color='#de663c', marker='x', ls='', label='보정 전 native 예측(B/I만 표시)')], fontsize=9)
    save(fig, ax, '01_own_initial.png')

    fig, ax = canvas('② 자기 상태 · 실제 B/I 측정과 관측한 선\nS/O/T/U는 2D 측정 없음 · S는 모델 투영으로만 표시')
    wire(ax, own_initial, '#b4a9c1', '초기 모델 투영', 'panel2_initial', linewidth=1.)
    for k, ray in rays:
        points = np.asarray(ray['observed_points'], float)
        normal = np.asarray(ray['normal'], float)
        normal_norm = np.linalg.norm(normal)
        normal /= normal_norm
        tangent = np.array([-normal[1], normal[0]])
        support_interval = points @ tangent
        fit = -float(ray['offset']) / normal_norm * normal + np.array([support_interval.min(), support_interval.max()])[:,None] * tangent
        ax.scatter(points[:,0], points[:,1], color='#f3c52b', s=5, alpha=.85, zorder=4)
        ax.plot(fit[:,0], fit[:,1], color='#d1a900', lw=1.2, zorder=4)
    actual = []
    for m in measurements:
        k = m['native_corner']
        xy = np.asarray(m['xy'], float)
        fallback = m['source'] == 'VISIBLE_BACKBONE_FALLBACK'
        point(ax, xy, f'{k}:{states[k]}', '#666666' if fallback else COLOR[states[k]], 's' if fallback else 'o')
        movement = float(np.linalg.norm(xy - raw[k]))
        actual.append(dict(native_corner=k, state=states[k], source=m['source'], xy=m['xy'],
                           displacement_px=movement, fallback_original_prediction=fallback))
        if not fallback and movement > 1e-8:
            ax.annotate('', xy, raw[k], arrowprops=dict(arrowstyle='->', color=COLOR[states[k]], lw=1.5))
    if own_initial is not None:
        for k, state in enumerate(states):
            if state not in ('B', 'I') and (0 <= own_initial[k,0] < width and 0 <= own_initial[k,1] < height):
                ax.text(*own_initial[k], f'모델{k}:{state}', color='#69616f', fontsize=8, clip_on=True)
    proof['measurement_display'] = actual
    ax.legend(handles=[Line2D([], [], color='#f3c52b', marker='.', label='실제 관측 선 표본/지지구간'),
        Line2D([], [], color=COLOR['B'], marker='o', ls='', label='B 영상 측정'),
        Line2D([], [], color=COLOR['I'], marker='o', ls='', label='I 영상 측정'),
        Line2D([], [], color='#666666', marker='s', ls='', label='원예측 유지')], fontsize=9)
    save(fig, ax, '02_states_BI_rays.png')

    fig, ax = canvas('③ ' + prefix + '1 · 보류 규칙 적용 전 B/I PnP 결과\n' + _metric_text(score['metrics'][prefix+'1']))
    wire(ax, own_initial, '#a7a2ac', '초기 모델', 'panel3_initial', '--', 1.1)
    wire(ax, projections['1'], '#ef861a', 'B/I PnP 결과 모델', 'panel3_proposal')
    for m in measurements:
        point(ax, np.asarray(m['xy'], float), str(m['native_corner']), '#00a3c7', 'o', 22)
    if not solved.get('accepted'):
        ax.text(.01, .97, 'PnP 미수락: 기존 fallback 출력', transform=ax.transAxes, va='top',
                color='#813b2c', bbox=dict(facecolor='white', alpha=.9), fontsize=10)
    save(fig, ax, '03_unguarded_BI_proposal.png')

    fig, ax = canvas('④ 자기 초기/규칙 A/규칙 B 최종 모델\n측정점이 아닌 실제 최종 자세 투영')
    wire(ax, projections['0'], '#666666', prefix+'0 초기', 'panel4_baseline', '--', 2.1)
    wire(ax, projections['A'], '#119d65', prefix+'A 최종', 'panel4_A', '-', 1.7)
    wire(ax, projections['B'], '#d347a6', prefix+'B 최종', 'panel4_B', ':', 2.)
    lines = [prefix+'0: '+_metric_text(score['metrics'][prefix+'0'])]
    for mode in ('A','B'):
        decision = guard_audits[prefix+mode]['result']
        lines += [prefix+mode+': '+_metric_text(score['metrics'][prefix+mode])+' · '
            +('교체' if decision['accepted'] else '초기 그대로'), '  '+decision['reason']]
    ax.text(.01, .99, '\n'.join(lines), transform=ax.transAxes, va='top', fontsize=8.5,
            color='#222', bbox=dict(facecolor='white', alpha=.9))
    save(fig, ax, '04_guarded_final.png')
    return proof


def render(folder, prior, source_root, output):
    folder, prior, source_root, output = map(Path, (folder, prior, source_root, output))
    if output.exists():
        raise FileExistsError('Preserve every previous gallery attempt')
    pseal, sseal = read(folder/'PREDICTION_SEAL.json'), read(folder/'score/SCORE_SEAL.json')
    selection = read(folder/'SELECTION_SEAL.json')
    for file, expected in (('PREDICTIONS.jsonl.gz', pseal['predictions_sha256']),
            ('GUARD_AUDITS.jsonl.gz', pseal['guard_audits_sha256']),
            ('INDEPENDENT_BINDINGS.json', pseal['independent_bindings_sha256'])):
        if sha(folder/file) != expected:
            raise ValueError('Saved gallery input changed: ' + file)
    if (sseal['prediction_seal_sha256'] != sha(folder/'PREDICTION_SEAL.json')
            or sseal['pre_PRIMARY_selection_seal_sha256'] != sha(folder/'SELECTION_SEAL.json')
            or selection['prediction_seal_sha256'] != sha(folder/'PREDICTION_SEAL.json')):
        raise ValueError('Prediction/CAL selection/score seal chain changed')
    for name, digest in sseal['artifacts'].items():
        if sha(folder/'score'/name) != digest:
            raise ValueError('Sealed numeric score artifact changed')
    scored = read(folder/'score/SCORED_ROWS.json')['rows']
    predictions = {r['id']:r for r in rows(folder/'PREDICTIONS.jsonl.gz')}
    if len(scored) != 238 or len(predictions) != 238 or [r['id'] for r in scored] != pseal['ordered_ids']:
        raise ValueError('Exact238 gallery population required')
    selected = select_cases(scored, selection['selected_arm'])
    wanted = {r['id'] for r in selected['cases']}
    scores = {r['id']:r for r in scored}
    inputs = read(prior/'inputs_v2/INPUTS.json')['frames']
    e_inputs = read(folder/'e_inputs/INPUTS.json')['frames']
    if [f['id'] for f in inputs] != pseal['ordered_ids'] or [f['id'] for f in e_inputs] != pseal['ordered_ids']:
        raise ValueError('Own input physical-vertex population differs')
    permission = public_image_inventory(source_root)
    for frame in inputs:
        if frame['image_sha256'] not in permission:
            raise ValueError('RGB publication permission missing: ' + frame['id'])
        if sha(source_root/frame['image']) != frame['image_sha256']:
            raise ValueError('Original238 RGB hash changed')
    indices = {f['id']:i for i,f in enumerate(inputs)}
    bindings = {r['id']:r for r in read(folder/'INDEPENDENT_BINDINGS.json')['records']}
    audits = {}
    for row in rows(folder/'GUARD_AUDITS.jsonl.gz'):
        if row['id'] in wanted:
            audits.setdefault(row['id'], {})[row['arm']] = row
    output.mkdir(parents=True)
    write(output/'SELECTION.json', selected)
    cases = []
    for index, case in enumerate(selected['cases']):
        fid, prefix = case['id'], case['prefix']
        i = indices[fid]
        frame = e_inputs[i] if prefix == 'E' else inputs[i]
        bound = bindings[fid]['branches'][prefix]
        saved = {}
        for kind in ('observation', 'solution'):
            binding = bound[kind]
            if sha(binding['path']) != binding['sha256']:
                raise ValueError('Bound actual saved stage changed')
            saved[kind] = read(binding['path'])
            if saved[kind]['id'] != fid or saved[kind]['branch'] != BRANCH[prefix]:
                raise ValueError('Foreign branch cached stage')
        name = f'case{index:02d}'
        proof = _draw(frame, predictions[fid], saved['observation']['result']['observation'],
            saved['solution']['result'], audits[fid], scores[fid], case, output/name, source_root)
        proof['input_bindings'] = {kind:dict(path=_input_binding(v['path'], folder, prior), sha256=v['sha256'])
                                    for kind,v in bound.items()}
        proof['public_RGB_proof'] = dict(inventory_path=str(PUBLIC_INVENTORY),
            inventory_sha256=PUBLIC_INVENTORY_SHA256, original_public_path=permission[frame['image_sha256']],
            image_sha256=frame['image_sha256'], whole238_hashes_and_permission_rechecked=True)
        write(output/name/'PROCESS.json', proof)
        table = '<table><tr><th>경로</th><th>실제 저장 오차</th><th>교체/유지 사유</th></tr>'
        for suffix in ('0','1','A','B'):
            arm = prefix+suffix
            state = predictions[fid]['predictions'][arm]
            table += '<tr><td>'+arm+'</td><td>'+html.escape(_metric_text(scores[fid]['metrics'][arm]))+'</td><td>'+html.escape(str(state.get('reason')))+'</td></tr>'
        table += '</table>'
        tilehtml = ''.join(f'<a href="{tile}" target="_blank"><img src="{tile}" alt="{html.escape(tile)}"></a>' for tile in proof['tiles'])
        page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+html.escape(fid)+'</title><style>'+STYLE+'</style>'
            '<a href="../index.html">사례 목록</a><h1>'+html.escape(fid)+'</h1><p>'+html.escape(' / '.join(case['categories']))+'</p>'
            '<p>같은 원영상 전체 범위를 사용합니다. 원/사각형은 B/I 영상 좌표이며, wire는 모델 투영입니다. S/O/T/U는 실제 2D 측정으로 표시하지 않습니다. 화면 밖 모델 선은 원영상 경계에서 잘라 표시하며 PROCESS.json에 기록했습니다. 정답 선과 만들어낸 중간 자세는 없습니다.</p>'
            +table+'<div class="tiles">'+tilehtml+'</div><p><a href="PROCESS.json">실제 좌표·수락 사유·관측 출처</a></p></html>')
        (output/name/'index.html').write_text(page)
        cases.append(dict(**case, path=name+'/index.html', process=name+'/PROCESS.json',
                          tiles=[name+'/'+t for t in proof['tiles']]))
    nav = ''.join('<li><a href="'+c['path']+'">'+html.escape(c['id'])+'</a> · '+html.escape(' / '.join(c['categories']))+'</li>' for c in cases)
    absent = ' / '.join(selected['absent_categories']) or '없음'
    (output/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>상태별 B/I 최종 검증 과정</title><style>'+STYLE+'</style>'
        '<h1>자기 초기 자세 → 상태/영상 보정 → PnP → 검증 후 교체</h1><p>CAL에서 고정한 경로: '+html.escape(selection['selected_arm'])+'. 아래 사례 선택은 점수 봉인 뒤의 진단이며 방법 선택에 영향을 주지 않습니다. 전체192장 분모와 실패는 보고서에 남습니다.</p>'
        '<p>고정 범주에 존재하지 않는 전환: '+html.escape(absent)+'</p><ul>'+nav+'</ul><p><a href="GALLERY_KO.md">Markdown 그림 목록</a></p></html>')
    md = ['# 상태별 B/I 검증 과정 갤러리', '',
        '실제 저장된 초기 입력·자기 상태/영상 측정·B/I PnP 제안·A/B 최종 결과를 같은 전체 원영상 위에 표시했다. 정답 선은 그리지 않았다. 화면 밖 모델은 원영상 경계에서 잘라 표시한다.', '',
        'CAL 선택 경로: **'+selection['selected_arm']+'**. 사례 선택은 사후 진단이며 추론과 경로 선택에 영향이 없다.', '',
        '존재하지 않는 고정 범주: '+absent, '']
    for case in cases:
        prefix = case['prefix']; score = scores[case['id']]
        md += ['## '+case['id'], '', ' / '.join(case['categories']), '',
               '[개별 HTML]('+case['path']+') · [실제 과정/좌표]('+case['process']+')', '',
               '| 경로 | 위치·회전 오차 | 교체·유지 사유 |', '|---|---|---|']
        for suffix in ('0','1','A','B'):
            arm = prefix+suffix
            md += ['| '+arm+' | '+_metric_text(score['metrics'][arm])+' | '+str(predictions[case['id']]['predictions'][arm].get('reason')).replace('|','/')+' |']
        md += [''] + ['!['+Path(t).stem+']('+t+')' for t in case['tiles']] + ['']
    (output/'GALLERY_KO.md').write_text('\n'.join(md))
    write(output/'PROCESS.json', dict(schema=SCHEMA, cases=cases,
        selection=selected, all238_public_RGB_hashes_rechecked=True, new_inference_calls=0,
        annotation_or_reference_file_reads=0, model_projection_not_measurement=True))
    assets = {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'GALLERY_SEAL.json', dict(schema=SCHEMA, cases=len(cases), stage_images=4*len(cases),
        prediction_seal_sha256=sha(folder/'PREDICTION_SEAL.json'),
        score_seal_sha256=sha(folder/'score/SCORE_SEAL.json'),
        CAL_selection_seal_sha256=sha(folder/'SELECTION_SEAL.json'),
        source_sha256=sha(__file__), font_sha256=sha(FONT),
        public_RGB_inventory_sha256=PUBLIC_INVENTORY_SHA256, assets=assets,
        new_NN_PnP_calls=0, GT_wire_or_reference_reads=0))
    return dict(output=str(output), cases=len(cases), stage_images=len(cases)*4,
                gallery_seal_sha256=sha(output/'GALLERY_SEAL.json'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('folder','prior','source-root','output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    print(render(args.folder, args.prior, args.source_root, args.output))
