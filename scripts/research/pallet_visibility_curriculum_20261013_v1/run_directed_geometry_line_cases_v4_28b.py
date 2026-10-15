"""Re-rank the two sealed old-input RGB line sets; never search new pixels.

This diagnostic preserves every measured alternative. A new directed incident
support test can reject inconsistent junction associations. The original image
and a coordinate-preserving ROI are shown together; no KP or pose is adopted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from . import run_joint_geometry_guided_line_cases_v4_28 as OLD

C = OLD.C
INPUT = OLD.B / 'joint_geometry_guided_line_cases_v4_28_v2'
OUTPUT = OLD.B / 'joint_geometry_guided_line_cases_v4_28_v3'


def _directed():
    from . import ordered_geometry_support_v4_28b as module
    return module


def _verify_parent(parent):
    parent = Path(parent)
    inputs = C.read(parent / 'INPUT_SEAL.json')
    observed = C.read(parent / 'OBSERVATION_SEAL.json')
    rendered = C.read(parent / 'RENDER_SEAL.json')
    if (C.sha(parent / 'INPUTS.json') != inputs['inputs_sha256']
            or C.sha(parent / 'PLAN.json') != inputs['plan_sha256']
            or C.sha(parent / 'INPUT_SEAL.json') != observed['input_seal_sha256']
            or C.sha(parent / 'OBSERVATION_SEAL.json') != rendered['observation_seal_sha256']):
        raise ValueError('Sealed historical two-case input/observation/report required')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        OLD._verify(inputs[key])
    for rel, digest in observed['observations'].items():
        if C.sha(parent / rel) != digest:
            raise ValueError('Cached actual RGB chains changed')
    return inputs


def prepare(output=OUTPUT, parent=INPUT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists():
        raise FileExistsError('Preserve every preparation and partial run')
    _verify_parent(parent)
    directed = _directed()
    sources = dict(C.read(parent / 'INPUT_SEAL.json')['source_sha256'])
    for module in (directed, OLD, OLD.VISIBILITY):
        sources[str(Path(module.__file__).resolve().relative_to(C.ROOT))] = C.sha(module.__file__)
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))] = C.sha(__file__)
    parent_paths = [parent / name for name in
                    ('PLAN.json', 'INPUTS.json', 'INPUT_SEAL.json', 'OBSERVATION_SEAL.json', 'RENDER_SEAL.json')]
    parent_paths += [parent / rel for rel in C.read(parent / 'OBSERVATION_SEAL.json')['observations']]
    bound = {str(p.relative_to(C.ROOT)): C.sha(p) for p in parent_paths}
    output.mkdir(parents=True)
    C.write(output / 'PLAN.json', dict(
        schema='CACHED_DIRECTED_INCIDENT_LINE_DIAGNOSTIC_PLAN_V28B',
        created_UTC=datetime.now(timezone.utc).isoformat(), case_ids=list(OLD.CASES),
        parent=str(parent.relative_to(C.ROOT)), parent_sha256=bound, source_sha256=sources,
        inputs='exact sealed v2 historical q, initial geometry, base43 and geometry43 actual chains',
        configuration=deepcopy(directed.RECIPE),
        new_actual_pixel_searches=0, new_NN_pose_CAL_fit_GT_or_metrics=0,
        guide_bank='same initial ORIGINAL/SWAPPED coarse and BI endpoint directions; no numeric e pose',
        operation='directed incident support re-ranking; all measured alternatives archived; low-global-span actual TLS support may remain explicitly graded PARTIAL_LINE_PROPOSAL diagnostics',
        partial_line_scope='cached inlier points only; partial evidence is not a vanilla full17c line or a complete KP',
        no_KP_PnP_or_default_adoption=True, independent_occluder_mask=None,
        display='original full image + four common-coordinate ROI panels',
        crop_rule='model projections + gated archived BI + cached actual support, padding existing43+7',
        S_U_local_queries_or_point_markers=0, U_regions_only=True,
        previous_wrong_associations_preserved_not_claimed_corrections=True))
    for relative, digest in sources.items():
        target = output / 'code_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(C.ROOT / relative, target)
        if C.sha(target) != digest:
            raise ValueError('New source snapshot differs')
    shutil.copyfile(parent / 'INPUTS.json', output / 'INPUTS.json')
    C.write(output / 'INPUT_SEAL.json', dict(
        plan_sha256=C.sha(output / 'PLAN.json'), inputs_sha256=C.sha(output / 'INPUTS.json'),
        parent_sha256=bound, source_sha256=sources,
        protected_user_sha256=C.read(parent / 'INPUT_SEAL.json')['protected_user_sha256'],
        rows=2, before_new_reranking=True, new_pixel_searches=0))
    print(dict(prepared_cases=2, input_seal=C.sha(output / 'INPUT_SEAL.json')), flush=True)


def _verify_run(output):
    output = Path(output)
    seal = C.read(output / 'INPUT_SEAL.json')
    if (C.sha(output / 'PLAN.json') != seal['plan_sha256']
            or C.sha(output / 'INPUTS.json') != seal['inputs_sha256']):
        raise ValueError('Frozen diagnostic plan and input bytes required')
    for key in ('parent_sha256', 'source_sha256', 'protected_user_sha256'):
        OLD._verify(seal[key])
    return seal


def _combined(observed):
    result = observed['result']
    return dict(result['base_observation'], edge_records=[
        *result['base_observation']['edge_records'], *result['additional_guided_edge_records']])


def _expected_visibility(row, graph, context):
    selected = set(graph['selected_line_names']);seen = set();records = [];counts = Counter()
    for k, state in enumerate(row['states']):
        if row['geometry']['initial_self_hidden'][k] or state not in ('B', 'I'):
            continue
        for mate in OLD.JOINT.NEIGHBORS[k]:
            edge = tuple(sorted([k, mate]))
            if (row['geometry']['initial_self_hidden'][mate] or row['states'][mate]=='S' or edge in seen):
                continue
            seen.add(edge)
            actual = [n for n in graph['line_candidates'] if n['name'] in selected and tuple(n['native_edge'])==edge]
            support = np.vstack([n['actual_inliers'] for n in actual]) if actual else np.empty((0, 2))
            # A cached span-only TLS rejection has no full RGB/context arrays.
            # Do not manufacture them or require the full17c context field.
            roles = {n['line_record'].get('mask_context_kind') for n in actual}
            known_role = next(iter(roles)) if len(roles)==1 and None not in roles else None
            record = context.classify_expected_edge(k, mate, observed_chain_points=support,
                                                   mask_context_kind=known_role)
            record['actual_line_context_missing_or_incomplete'] = any(
                n.get('proposal_grade', n['line_record'].get('proposal_grade'))=='PARTIAL_LINE_PROPOSAL'
                or n['line_record'].get('mask_context_kind') is None for n in actual)
            records.append(record);counts.update(record['counts'])
    return dict(records=records, counts=dict(counts), independent_occluder_present=False,
                target_mask_zero_not_promoted_to_occlusion=True,
                expected_model_is_original_N3_not_measured_edge=True,
                partial_line_RGB_context_not_fabricated=True)


def rerank(output=OUTPUT):
    output = Path(output).resolve()
    seal = _verify_run(output)
    if (output / 'OBSERVATION_SEAL.json').exists():
        raise FileExistsError('Preserve completed directed rankings')
    plan = C.read(output / 'PLAN.json');parent = C.ROOT / plan['parent']
    directed = _directed();started = time.monotonic();paths = []
    for row in C.read(output / 'INPUTS.json')['cases']:
        relative = 'frames/' + row['id'].replace(':', '__') + '.json'
        path = output / relative
        if path.exists():
            raise FileExistsError('Preserve every partial case')
        previous = C.read(parent / relative)
        with np.load(C.ROOT / row['mask_binding']['path'], allow_pickle=False) as arrays:
            mask = arrays[row['mask_binding']['key']].copy()
        binding = dict(sha256=row['image_sha256'], image_hw=row['geometry']['image_hw'])
        context = OLD.VISIBILITY.prepare_edge_visibility(mask, row['geometry'], row['states'],
            row['visible_measurements'], image_binding=binding, occluder_mask=None, occluder_binding=None)
        graph = directed.rank_ordered_geometry_line_graph(_combined(previous), row['geometry'], row['states'],
            row['visible_measurements'], image_binding=binding, pose_guides=row['pose_guides'],
            pallet_mask=mask, visibility_context=context)
        result = deepcopy(previous['result']);result['joint_graph'] = graph
        C.write(path, dict(id=row['id'], result=result,
            previous_joint_graph=previous['result']['joint_graph'],
            expected_edge_visibility=_expected_visibility(row, graph, context),
            previous_expected_edge_visibility=previous['expected_edge_visibility'],
            parent_frame_sha256=C.sha(parent / relative), cached_actual_chains_unchanged=True,
            new_pixel_queries=0, new_NN_posefit_GT_or_metrics=0, KP_PnP_adoptions=0))
        paths.append(path)
        print(dict(reranked=row['id'], elapsed_seconds=time.monotonic()-started), flush=True)
    _verify_run(output)
    C.write(output / 'OBSERVATION_SEAL.json', dict(
        input_seal_sha256=C.sha(output / 'INPUT_SEAL.json'), rows=2,
        observations={str(p.relative_to(output)): C.sha(p) for p in paths},
        source_sha256=seal['source_sha256'], new_pixel_searches=0,
        new_NN_posefit_GT_metrics=0, precise_KP_or_PnP_adoption=False,
        elapsed_seconds=time.monotonic()-started))


def _roi(row, graph):
    points = [np.asarray(row['geometry']['initial_projection_xy'], float)]
    for m in row['visible_measurements']:
        k = m['native_corner']
        if row['geometry']['initial_self_hidden'][k] or row['states'][k] not in ('B', 'I'):
            continue
        points.append(np.asarray(m['xy'], float).reshape(1, 2))
    for node in graph['line_candidates']:
        k = node['native_corner']
        if row['geometry']['initial_self_hidden'][k] or row['states'][k] not in ('B', 'I'):
            continue
        points.append(np.asarray(node['actual_inliers'], float).reshape(-1, 2))
    p = np.vstack(points);p = p[np.isfinite(p).all(axis=1)]
    h, w = row['geometry']['image_hw'];pad = 43. + 7.
    return [float(max(0., p[:, 0].min()-pad)), float(min(w, p[:, 0].max()+pad)),
            float(max(0., p[:, 1].min()-pad)), float(min(h, p[:, 1].max()+pad))]


def render(output=OUTPUT):
    output = Path(output).resolve();_verify_run(output)
    if (output / 'RENDER_SEAL.json').exists():
        raise FileExistsError('Preserve completed report')
    observed_seal = C.read(output / 'OBSERVATION_SEAL.json')
    if C.sha(output / 'INPUT_SEAL.json') != observed_seal['input_seal_sha256']:
        raise ValueError('Rerank input binding changed')
    for rel, digest in observed_seal['observations'].items():
        if C.sha(output / rel) != digest:
            raise ValueError('Directed ranking changed')
    from matplotlib.font_manager import fontManager, FontProperties
    font = OLD.DRAW_SOURCE.DRAW._font();fontManager.addfont(str(font))
    plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    summaries = [];assets = []
    for i, row in enumerate(C.read(output / 'INPUTS.json')['cases']):
        observed = C.read(output / 'frames' / (row['id'].replace(':', '__')+'.json'))
        graph = observed['result']['joint_graph'];prior = observed['previous_joint_graph']
        image = cv2.cvtColor(cv2.imread(str(C.ROOT / row['image'])), cv2.COLOR_BGR2RGB)
        roi = _roi(row, graph);left, right, top, bottom = roi
        fig = plt.figure(figsize=(17, 9));grid = fig.add_gridspec(2, 3, width_ratios=[1.1, 1., 1.])
        overview = fig.add_subplot(grid[:, 0]);overview.imshow(image);overview.set_axis_off()
        overview.add_patch(Rectangle((left, top), right-left, bottom-top, fill=False, ec='#efb52f', lw=1.5))
        overview.set_title('원본 사진 · 노랑 상자만 확대', fontsize=12)
        axes = [fig.add_subplot(grid[r, c]) for r, c in [(0, 1), (0, 2), (1, 1), (1, 2)]]
        titles = ['① 초기 KP/PnP 예상 변 (예측)', '② 보관된 실제 RGB 선 후보',
                  '③ 방향 조건을 함께 본 선 후보', '④ 관측 지지 / 미확정 영역']
        for ax, title in zip(axes, titles):
            ax.imshow(image);ax.set_xlim(left, right);ax.set_ylim(bottom, top);ax.set_aspect('equal')
            ax.set_title(title, fontsize=11);ax.tick_params(labelsize=7)
        OLD._wire(axes[0], row['geometry']['initial_projection_xy'])
        for m in row['visible_measurements']:
            k = m['native_corner']
            if row['geometry']['initial_self_hidden'][k] or row['states'][k] not in ('B', 'I'):
                raise ValueError('No S/U point marker')
            q = m['xy'];color = '#ef9a31' if m['source']=='VISIBLE_BACKBONE_FALLBACK' else '#229c75'
            axes[0].scatter(*q, marker='x', c=color, s=35)
            axes[0].text(q[0]+2, q[1], str(k), color=color, fontsize=9)
        nodes = graph['line_candidates'];chosen = set(graph['selected_line_names'])
        OLD._lines(axes[1], [n['line_record'] for n in nodes], '#8e969d', alpha=.65)
        OLD._lines(axes[2], [n['line_record'] for n in nodes], '#8e969d', alpha=.25)
        selected = [n for n in nodes if n['name'] in chosen]
        partial = [n for n in selected if n.get('proposal_grade', n['line_record'].get('proposal_grade'))=='PARTIAL_LINE_PROPOSAL']
        full = [n for n in selected if n not in partial]
        OLD._lines(axes[2], [n['line_record'] for n in full], '#15a59b', alpha=.95, lw=2.)
        OLD._lines(axes[2], [n['line_record'] for n in partial], '#d7a329', alpha=.95, lw=2.)
        shown_junctions = []
        for j in graph.get('rounded_BI_junctions', []):
            k = j['native_corner']
            if row['states'][k] not in ('B', 'I') or row['geometry']['initial_self_hidden'][k]:
                raise ValueError('No U/S intersection display')
            if j.get('available') and j.get('finite_support_11px_guard_would_pass', False):
                q = j['candidate_extrapolated_intersection_xy'];shown_junctions.append(k)
                axes[2].scatter(*q, marker='D', facecolors='none', edgecolors='#d5a22c', s=45)
        for edge in observed['expected_edge_visibility']['records']:
            for a, b in zip(edge['samples'], edge['samples'][1:]):
                label = a['classification'] if a['classification']==b['classification'] else 'unsupported_unknown'
                axes[3].plot([a['xy'][0], b['xy'][0]], [a['xy'][1], b['xy'][1]],
                    c={'observed_support':'#15a59b', 'occluder_supported_gap':'#df8c37', 'unsupported_unknown':'#9c6fc7'}[label],
                    ls='-' if label=='observed_support' else '--', lw=1.5)
        OLD._lines(axes[3], [n['line_record'] for n in full], '#15a59b', alpha=.9, lw=1.7)
        OLD._lines(axes[3], [n['line_record'] for n in partial], '#d7a329', alpha=.9, lw=1.7)
        for region in row['u_regions']:
            k = region['native_corner']
            if row['states'][k]!='U' or row['geometry']['initial_self_hidden'][k]:
                raise ValueError('Only saved U range permitted')
            axes[3].add_patch(Ellipse(region['center_xy'], 2*region['half_axes_px'][0], 2*region['half_axes_px'][1],
                angle=region.get('angle_deg', 0.), facecolor='#d1ab4b', edgecolor='#b68f31', alpha=.12))
        for ax in axes:
            ax.text(.01, .02, '보라 점선=예측 / 청록=실제 선 / 노랑=짧은 부분 선\n선후보 선정 ≠ 정답 · KP/자세 채택 없음',
                transform=ax.transAxes, fontsize=7, va='bottom', bbox=dict(facecolor='white', alpha=.88))
        axes[3].text(.01, .98, '독립 가림 마스크 없음 → 미지원은 원인 미확정\nU는 기존 범위만, 확대창 밖은 생략',
            transform=axes[3].transAxes, fontsize=7, va='top', bbox=dict(facecolor='white', alpha=.9))
        fig.suptitle(row['id']+' · 같은 실제 선을 방향 조건으로 다시 비교', fontsize=14)
        fig.tight_layout();path = output / ('case'+str(i)+'_actual.png')
        fig.savefig(path, dpi=150, bbox_inches='tight');plt.close(fig);assets.append(path)
        summaries.append(dict(id=row['id'], figure=path.name, roi_original_xy=roi,
            image_sha256=row['image_sha256'], mask_binding=row['mask_binding'],
            previous_selected_line_names=prior['selected_line_names'], selected_line_names=graph['selected_line_names'],
            selected_partial_line_names=[n['name'] for n in partial],
            partial_line_scope='saved span-only rejected TLS; missing RGB provenance remains unknown; engineering proposal only, not full17c/KP',
            rounded_BI_junctions=graph.get('rounded_BI_junctions', []), displayed_unadopted_junction_ids=shown_junctions,
            expected_edge_visibility=observed['expected_edge_visibility'],
            directed_ranking_audit={k:v for k,v in graph.items() if 'direct' in k or 'reject' in k},
            actual_chain_count=len(nodes), new_actual_pixel_queries=0, precise_KP_or_pose_adoptions=0))
    intro = ('<p><b>보라 점선은 예측</b>, 회색·청록은 사진에서 측정한 선 지지입니다. 원본과 확대창은 같은 좌표를 씁니다. '
             '아래 모서리에 위쪽 선을 붙이는 모순을 줄이기 위해, 모서리에서 이웃으로 향하는 방향과 실제 지지 구간을 함께 비교합니다.</p>'
             '<p>새 사진 탐색 없이 앞서 보관한 모든 선 후보를 재평가했습니다. 청록 선정선도 정답이 아니며, '
             '노랑 선은 기존 길이 비율 조건을 통과하지 못했던 짧은 실제 지지로, 완전한 선 관측으로 승격하지 않은 부분 후보입니다. '
             '교차 후보는 직선 연장의 결과입니다. KP 보정·자세 추정에는 채택하지 않았습니다. '
             '가림 물체의 독립 마스크가 없어 미지원 구간의 원인은 미확정으로 남습니다. U는 기존 넓은 범위만 표시합니다.</p>')
    sections = ''.join('<section><h2>'+html.escape(s['id'])+'</h2><img src="'+s['figure']+'" alt="원본과 네 단계 확대 비교">'
        '<details><summary>입력과 관측 근거</summary><pre>'+html.escape(json.dumps(s, ensure_ascii=False, indent=2))+'</pre></details></section>' for s in summaries)
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>실제 선과 모서리 방향 비교</title>'
            '<style>body{font:16px/1.6 sans-serif;max-width:1700px;margin:24px auto;padding:0 20px;color:#183042}'
            'img{width:100%;height:auto}pre{white-space:pre-wrap;font-size:12px}section{margin:28px 0}</style>'
            '<h1>실제 선을 방향과 함께 비교하기</h1>'+intro+sections+'<p><a href="PROCESS_DATA.json">전체 표시 근거</a></p></html>')
    (output / 'index.html').write_text(page, encoding='utf-8')
    C.write(output / 'PROCESS_DATA.json', dict(cases=summaries, new_pixel_queries=0, pose_metrics_or_GT_reads=0,
        model_guide_not_measured_normal=True, S_U_point_markers=0, KP_pose_adoptions=0))
    assets += [output / 'index.html', output / 'PROCESS_DATA.json']
    C.write(output / 'RENDER_SEAL.json', dict(observation_seal_sha256=C.sha(output / 'OBSERVATION_SEAL.json'),
        source_sha256=C.sha(__file__), assets={p.name:C.sha(p) for p in assets},
        case_ids=list(OLD.CASES), original_RGB_plus_coordinate_preserving_ROI=True,
        no_GT_wire_S_U_point_or_adopted_KP=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'rerank', 'render'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--parent', type=Path, default=INPUT)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.parent)
    else:
        dict(rerank=rerank, render=render)[args.command](args.output)
