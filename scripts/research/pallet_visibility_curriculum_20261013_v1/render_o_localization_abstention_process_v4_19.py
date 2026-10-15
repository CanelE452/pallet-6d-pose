"""Postseal actual-image O-localization ablation; three predeclared cases.

Uniform e / continuous18c native / continuous19 native comparison. No target
pose or corner reader, new fit, inference, case-best arm or image alteration.
The old O ellipse is a raw-predictor tolerance, never an occluder outline.
"""
from __future__ import annotations

from pathlib import Path
import html
import os
from urllib.parse import quote

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from . import common as C
from . import evaluate_o_localization_abstention_v4_19 as EXP
from . import render_refined_grazing_mixture_process_v4_18c as OLD


DRAW = OLD.DRAW
OUTPUT = EXP.OUTPUT.parent/'o_localization_abstention_process_v4_19'
PLAN = EXP.OUTPUT.parent/'o_localization_abstention_report_plan_v4_19'
CASES = ('plastic_night_01:040923', 'wood_night_01:032781', 'wood_night_01:033821')
BASELINE, PRIMARY = EXP.ARMS[0], EXP.ARMS[1]
PARENT_NATIVE = EXP.PARENT.ARMS[1]
METHODS = (BASELINE, PARENT_NATIVE, PRIMARY)
NAMES = {BASELINE: '기존 e', PARENT_NATIVE: '18c 연속·native (기존 O 원)',
         PRIMARY: '19 연속·native (O 위치제약 보류)'}
COLORS = {BASELINE: DRAW.BLUE, PARENT_NATIVE: DRAW.ORANGE, PRIMARY: DRAW.GREEN}
TITLES = ('① 초기 모델·고정 B/I 기록', '② 동일 실제 마스크·NMS 선 대안',
          '③ 기존 O 원 + U·부분관측 범위', '④ O 위치제약 보류: 기존 전체 캔버스',
          '⑤ 기존 e 저장 자세', '⑥ 18c native 저장 자세', '⑦ 19 native 저장 자세')
CAPTIONS = (
    '회색 점선은 초기 모델투영. 분홍 ×는 완전 B/I 기록, 회색 +는 부분 B/I의 보관 q이며 정밀 RGB 코너가 아님. S/U 원좌표는 읽거나 표시하지 않음.',
    '저장 실제 마스크와 모든 실제 NMS 선 대안. 청록은 현재 부분관측에 연결된 물리 edge 대안. 영상·마스크·선·잡음은 18c와 19에서 동일하며 새 조회 없음.',
    '주황 O 원은 raw N3 예측 중심과 CAL 경험 반경의 허용 영역. 실제 가림 물체 외곽이나 정밀 KP가 아님. 회색은 그대로 둔 U·기존 부분 B/I 접선 범위.',
    'O 상태는 유지하고 원의 위치제약만 보류. 원래도 있던 닫힌 영상 캔버스를 유지하며 이 확대창 밖도 허용됨. 작은 창은 전체 영상과 현재 확대 위치. U·부분관측은 그대로.',
    '파란 wire는 봉인된 기존 e의 모델투영. 영상 측정점이나 참조/GT wire가 아님.',
    '주황 wire는 기존 O 원을 포함한 18c native-only 연속 제어의 저장 선택. 모든 사례에서 같은 방법이며 사례별 최적 결과를 고르지 않음.',
    '초록 wire는 O 원 위치제약만 보류한 19 native-only 저장 선택. 비용·관측·시작점 정책은 동일. 선택된 실제 시작점/해는 바뀔 수 있으며 유일성·전역최적성 보증은 아님.')


def _url(path):
    return quote(os.path.relpath(Path(path).resolve(), OUTPUT), safe='/')


def _binding(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)), C.sha(path)


def _load_plan():
    plan = C.read(PLAN/'PREDECLARATION.json')
    if tuple(plan['case_ids']) != CASES or plan['new_v19_metrics_read_before_declaration']:
        raise ValueError('The predeclared cases must precede new control metrics')
    if C.sha(Path(__file__)) != plan['renderer_source_sha256']:
        raise ValueError('Render the declared source, not an unrecorded revision')
    return plan


def _old_o(ax, domains):
    offsets = ((7, 16), (7, -12), (-60, 13))
    for index, d in enumerate(domains):
        if d['state'] != 'O':
            continue
        if d['shape'] != 'ellipse':
            raise ValueError('These predeclared cases have original localized-O ellipses')
        center = np.asarray(d['center_xy'])
        axes = np.asarray(d['half_axes_px'])
        ax.add_patch(Ellipse(center, 2*axes[0], 2*axes[1], angle=d.get('angle_deg', 0),
            fc='#efa300', ec='#b37500', alpha=.18, lw=1.7))
        ax.plot(*center, '+', c='#b37500', ms=6)
        DRAW._label(ax, center, 'O'+str(d['native_corner'])+' raw 중심', '#946000', offsets[index % 3])


def _new_canvas(ax, image, domains, bounds):
    h, w = image.shape[:2]
    ids = []
    for d in domains:
        if d['state'] != 'O':
            continue
        if d['shape'] != 'box' or d['bounds_xyxy'] != [0., 0., float(w), float(h)] or d['halfplanes']:
            raise ValueError('The actual selected control domains must be the complete original canvas')
        ids.append(d['native_corner'])
    ax.add_patch(Rectangle((0, 0), w, h, fc='#efa300', ec='#ac7200', alpha=.10, lw=2))
    ax.text(.02, .97, 'O '+','.join(map(str, ids))+' : 전체 영상 [0,'+str(w)+']×[0,'+str(h)+']',
        transform=ax.transAxes, va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.86))
    # Actual full photo, not a fabricated occluder or image evidence region.
    inset = ax.inset_axes([.77, .05, .21, .25])
    inset.imshow(image)
    inset.add_patch(Rectangle((0, 0), w, h, fill=False, ec='#c38400', lw=1.8))
    inset.add_patch(Rectangle((bounds[0], bounds[3]), bounds[1]-bounds[0], bounds[2]-bounds[3],
        fill=False, ec='#00a5aa', lw=1.1))
    inset.set_xlim(0, w); inset.set_ylim(h, 0); inset.set_axis_off()
    inset.set_title('전체 영상 / 청록: 확대창', fontsize=7)


def _context(ax, task):
    DRAW._domains(ax, [], task['u_regions'])
    OLD._tangent_ranges(ax, task['boundary_merged'])


def _domain_audit(projected, domains, image_hw):
    if projected is None:
        return []
    h, w = image_hw
    result = []
    for d in domains:
        if d['state'] != 'O':
            continue
        k = d['native_corner']; angle = np.radians(d.get('angle_deg', 0))
        basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        delta = basis.T @ (projected[k]-np.asarray(d['center_xy']))
        radius2 = float(np.sum((delta/np.asarray(d['half_axes_px']))**2))
        result.append(dict(native_corner=k, model_projection_xy=projected[k].tolist(),
            original_ellipse_quadratic=radius2, inside_original_ellipse=radius2 <= 1.+1e-10,
            inside_closed_canvas=bool(0 <= projected[k, 0] <= w and 0 <= projected[k, 1] <= h),
            model_projection_not_image_measurement_or_reference=True))
    return result


def _pose_panel(ax, q, arm, key, metric, bounds, states):
    if q is None:
        ax.text(.03, .05, '저장 자세 없음', transform=ax.transAxes)
        return
    DRAW._wire(ax, q, COLORS[key], width=1.8, ls='-' if arm['accepted'] else '--')
    DRAW._points(ax, {k:q[k] for k, state in enumerate(states) if state == 'S'}, DRAW.PURPLE, '^', prefix='S모델')
    outside = np.any((q < [bounds[0], bounds[3]]) | (q > [bounds[1], bounds[2]]), axis=1)
    text = OLD._metric_text(metric)+' · '+('joint 수용' if arm['accepted'] else 'fallback')
    if outside.any():
        text += '\n모델 꼭짓점 '+str(int(outside.sum()))+'개가 확대창 밖 (전체 좌표 JSON)'
    ax.text(.02, .97, text, transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=.88))


def _summary_table(reports, parent_reports):
    rows = []
    for partition, title in (('TEST', '기존 TEST49'), ('REMAINDER', '추가143'), ('CAL', '별도 CAL46')):
        for key in METHODS:
            report = parent_reports[partition] if key == PARENT_NATIVE else reports[partition]
            value = report['summary'][key]
            rows.append('<tr><td>'+title+'</td><td>'+html.escape(NAMES[key])+'</td><td>'+str(value['translation_le5'])+
                '</td><td>'+str(value['successes_5cm_5deg'])+'</td><td>'+f"{value['translation_cm']['mean']:.3f}"+
                '</td><td>'+f"{value['translation_cm']['max']:.3f}"+'</td><td>'+str(value['fallback_count'])+'</td></tr>')
    return '<table><tr><th>집합</th><th>고정 방법</th><th>T≤5cm</th><th>5cm·5°</th><th>평균 cm</th><th>최대 cm</th><th>fallback</th></tr>'+''.join(rows)+'</table>'


def render():
    if OUTPUT.exists():
        raise FileExistsError('Preserve every completed or partial report')
    _load_plan()
    protocol, prediction, scores, reports = OLD._load_completed(EXP.OUTPUT)
    parent_protocol, parent, parent_scores, parent_reports = OLD._load_completed(EXP.PARENT.OUTPUT)
    if protocol['ids'] != parent_protocol['ids']:
        raise ValueError('All original238, including failed/fallback rows, must remain')
    for fid in protocol['ids']:
        if (C.finite(prediction[fid]['arms'][BASELINE]) != C.finite(parent[fid]['arms'][BASELINE])
                or scores[fid]['metrics'][BASELINE] != parent_scores[fid]['metrics'][BASELINE]):
            raise ValueError('Unchanged e baseline must be exact for all238')
    tasks = {t['id']:t for t in C.rows(EXP.OUTPUT/'INPUTS.jsonl.gz') if t['id'] in CASES}
    old_tasks = {t['id']:t for t in C.rows(EXP.PARENT.OUTPUT/'INPUTS.jsonl.gz') if t['id'] in CASES}
    observations = {r['id']:r for r in C.rows(OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz') if r['id'] in CASES}
    if set(tasks) != set(CASES) or set(old_tasks) != set(CASES) or set(observations) != set(CASES):
        raise ValueError('All three predeclared cases, without replacement, are required')
    paths = [Path(__file__), Path(OLD.__file__), Path(DRAW.__file__), PLAN/'PREDECLARATION.json',
        OLD.OBS.OUTPUT/'OBSERVATION_SEAL.json', OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz',
        EXP.PARENT.OUTPUT/'SCORER_COMPATIBILITY_RECEIPT.json', EXP.PARENT.OUTPUT/'SCORER_COMPATIBILITY_SEAL.json']
    for folder in (EXP.OUTPUT, EXP.PARENT.OUTPUT):
        paths += [folder/name for name in ('PROTOCOL.json', 'INPUT_SEAL.json', 'INPUTS.jsonl.gz',
            'PREDICTION_SEAL.json', 'PREDICTIONS.jsonl.gz', 'SCORE_SEAL.json',
            'CAL_SCORE.json', 'TEST_SCORE.json', 'REMAINDER_SCORE.json')]
    bindings = dict(_binding(p) for p in paths)
    font = DRAW._font()
    if font.is_file():
        bindings[str(font)] = C.sha(font)
    OUTPUT.mkdir(parents=True); assets = OUTPUT/'assets'; assets.mkdir()
    cases, pictures = [], []
    for index, fid in enumerate(CASES):
        task, old_task, observed = tasks[fid], old_tasks[fid], observations[fid]
        audit = task['O_localization_abstention']
        if (task['new_chain_observation'] != old_task['new_chain_observation']
                or task['boundary_merged'] != old_task['boundary_merged']
                or task['spec']['visible'] != old_task['spec']['visible']
                or observed['domains'] != old_task['spec']['domains']
                or observed['visible_measurements'] != task['spec']['visible']
                or observed['states'] != task['states']):
            raise ValueError('Same actual observations and original O provenance required')
        if [r['original_domain'] for r in audit['records']] != [d for d in old_task['spec']['domains'] if d['state'] == 'O']:
            raise ValueError('Original raw-centered O ellipse archive differs')
        image_path = C.ROOT/task['image']; mask_path = C.ROOT/observed['mask_binding']['path']
        if C.sha(image_path) != task['image_sha256'] or C.sha(mask_path) != observed['mask_binding']['sha256']:
            raise ValueError('Exact actual photo and mask required')
        bindings[task['image']] = task['image_sha256']
        bindings[observed['mask_binding']['path']] = observed['mask_binding']['sha256']
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError('Actual image decode failed')
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with np.load(mask_path, allow_pickle=False) as arrays:
            mask = arrays['pallet_mask_clean'].copy()
        geometry = observed['initial_geometry']; K = geometry['K']
        arms = {BASELINE:prediction[fid]['arms'][BASELINE],
            PARENT_NATIVE:parent[fid]['arms'][PARENT_NATIVE], PRIMARY:prediction[fid]['arms'][PRIMARY]}
        projected = {key:OLD._model(arm, K) for key, arm in arms.items()}
        metrics = {BASELINE:scores[fid]['metrics'][BASELINE],
            PARENT_NATIVE:parent_scores[fid]['metrics'][PARENT_NATIVE], PRIMARY:scores[fid]['metrics'][PRIMARY]}
        # Crop from unchanged initial geometry/observations, not GT or a chosen final pose.
        support = [np.asarray(geometry['initial_projection_xy'])]
        support += [np.asarray([m['xy'] for m in task['spec']['visible']])]
        support += [np.asarray([d['center_xy'] for d in old_task['spec']['domains'] if d['state'] == 'O'])]
        xy = np.vstack(support); h, w = image.shape[:2]
        lo = np.maximum(xy.min(0)-35, 0); hi = np.minimum(xy.max(0)+35, [w, h])
        bounds = (float(lo[0]), float(hi[0]), float(hi[1]), float(lo[1]))
        primary_arm = arms[PRIMARY]
        bundle = primary_arm['new_image_line_group_authentication'][primary_arm['WD_family']]
        case = dict(id=fid, partition=task['partition'], image=task['image'], image_sha256=task['image_sha256'],
            states=task['states'], initial_geometry=geometry, frozen_BI=task['spec']['visible'],
            original_O_T_domains=old_task['spec']['domains'], replacement_O_T_domains=task['spec']['domains'],
            O_localization_abstention=audit, U_regions=task['u_regions'], partial_boundary=task['boundary_merged'],
            all_actual_NMS_chain_observation=task['new_chain_observation'], mask_binding=observed['mask_binding'],
            methods={key:dict(actual_pose=arm['actual_pose'], accepted=arm['accepted'], WD_family=arm.get('WD_family'),
                model_projection_xy=None if projected[key] is None else projected[key].tolist(),
                selected_endpoint_origin=arm.get('keypoint_readout', {}).get('selected_endpoint_origin'),
                metric=metrics[key], original_O_projection_audit=_domain_audit(projected[key], old_task['spec']['domains'], [h,w]))
                for key,arm in arms.items()}, crop_bounds=bounds,
            crop_uses_only_unchanged_initial_geometry_and_archived_BI_O_centers=True,
            no_reference_pose_or_corner_read_or_overlay=True, stages=[])
        for stage, title in enumerate(TITLES):
            fig, ax = DRAW._base(image, title); ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:])
            if stage == 0:
                DRAW._wire(ax, np.asarray(geometry['initial_projection_xy']), '#7e8790', ls='--')
                OLD._frozen_points(ax, task)
            elif stage == 1:
                ax.imshow(np.dstack([np.zeros_like(mask), mask, np.zeros_like(mask), mask*.15]))
                case['NMS_display_audit'] = OLD._nms(ax, task['new_chain_observation'], bundle)
            elif stage == 2:
                _context(ax, task); _old_o(ax, old_task['spec']['domains'])
            elif stage == 3:
                _new_canvas(ax, image, task['spec']['domains'], bounds); _context(ax, task)
            else:
                key = METHODS[stage-4]
                _pose_panel(ax, projected[key], arms[key], key, metrics[key], bounds, task['states'])
            path = assets/f'case{index}_stage{stage+1}.png'; DRAW._save(fig, path, CAPTIONS[stage])
            case['stages'].append(dict(title=title, caption=CAPTIONS[stage], asset=str(path.relative_to(OUTPUT))))
        cases.append(case); pictures.append((image, bounds, projected, metrics))
    fig, axes = plt.subplots(3, len(CASES), figsize=(5*len(CASES), 10), squeeze=False)
    for col, (image, bounds, projected, metrics) in enumerate(pictures):
        for row, key in enumerate(METHODS):
            ax = axes[row, col]; ax.imshow(image)
            if projected[key] is not None:
                DRAW._wire(ax, projected[key], COLORS[key], width=1.5)
            ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); ax.set_aspect('equal'); ax.set_axis_off()
            ax.set_title((CASES[col]+'\n' if row == 0 else '')+NAMES[key]+'\n'+OLD._metric_text(metrics[key]), fontsize=9)
    fig.suptitle('실제 사진·미리 정한 세 사례 · e → 18c native → 19 native\nO 원 위치제약만 보류한 대조실험 / wire는 모델투영, 참조·GT wire 없음', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(OUTPUT/'overview_actual.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    C.write(OUTPUT/'PROCESS_DATA.json', dict(schema='O_LOCALIZATION_ABSTENTION_REAL_PROCESS_V4_19',
        predeclared_cases=list(CASES), fixed_uniform_methods=list(METHODS), cases=cases, source_bindings=bindings,
        original_O_is_empirical_raw_prediction_region_not_occluder_outline=True,
        new_O_is_existing_closed_canvas_not_new_physical_localization=True,
        new_inference_or_fit=0, new_reference_pose_or_corner_reads=0, S_raw_not_read=True,
        U_not_precise_point=True, default_e_unchanged=True))
    sections = []
    for case in cases:
        table = ''.join('<tr><td>'+html.escape(NAMES[key])+'</td><td>'+OLD._metric_text(case['methods'][key]['metric'])+
            '</td><td>'+html.escape(str(case['methods'][key]['WD_family']))+'</td><td>'+('joint 수용' if case['methods'][key]['accepted'] else 'fallback')+
            '</td><td>'+str(sum(not r['inside_original_ellipse'] for r in case['methods'][key]['original_O_projection_audit']))+'</td></tr>' for key in METHODS)
        stages = ''.join('<h3>'+s['title']+'</h3><img src="'+s['asset']+'" alt="실제 사진 저장 처리 단계"><p>'+s['caption']+'</p>' for s in case['stages'])
        sections.append('<details open><summary>'+html.escape(case['id'])+'</summary><p>고정 상태: '+' '.join(case['states'])+
            '</p><div class="tablewrap"><table><tr><th>고정 방법</th><th>명목 참조 오차</th><th>W/D family</th><th>상태</th><th>옛 O 원 밖 투영 수</th></tr>'+table+
            '</table></div><p>옛 원 밖 개수는 저장 모델투영의 기하 진단이며 실제 KP 위치나 참조 검사가 아닙니다.</p><a href="'+_url(C.ROOT/case['image'])+'">원본 RGB</a>'+stages+'</details>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>O 위치제약 보류 · 실제 사진 과정</title><style>body{margin:0;background:#edf2f6;color:#173044;font-family:system-ui,sans-serif}main{max-width:1080px;padding:18px;margin:auto}img{display:block;max-width:100%;height:auto}details{background:white;margin:18px 0;padding:16px;border-radius:10px}summary{font-weight:bold}p{line-height:1.6}.tablewrap{overflow-x:auto}table{border-collapse:collapse;max-width:100%;font-size:14px}td,th{padding:7px;border:1px solid #cbd6df;text-align:left}@media(max-width:500px){main{padding:8px}details{padding:9px}table{font-size:11px}td,th{padding:4px}}</style><main><h1>O 위치제약 보류: 실제 사진 대조실험</h1><p>세 사례는 새19 점수를 보기 전에 기존 O-hard-guard 진단에서 고정했습니다. 모든 사례에서 동일한 e → 18c native-only → 19 native-only를 비교하며, 사진별 최적 방법을 섞지 않았습니다. 대표성·성공을 보장하는 표본은 아닙니다.</p><p>O는 계속 O입니다. raw 예측 중심의 경험 원 제약만 보류하고 원래의 닫힌 캔버스를 유지했습니다. 원은 실제 가림 물체 외곽이 아니며 새 박스도 독립적인 물리 관측이 아닙니다. B/I·NMS·U·부분관측·잡음·모델과 연속 최적화 정책은 같습니다. 실제 feasible 후보와 선택 시작점은 달라질 수 있습니다.</p><p>wire는 저장 모델투영이고 사진 위 측정점이 아닙니다. S는 모델투영만, U는 범위만 사용합니다. GT/참조 wire·새 선 fitting·NN 호출은 없습니다. 추가143 오차는 저장 annotation/PnP에서 유도한 명목 참조 기준으로 독립 계측이 아닙니다. 반복 진단한 탐색 자료의 인과 대조실험이며 새 기본 방법이나 전체 목표 완료를 선언하지 않습니다.</p><div class="tablewrap">'''+_summary_table(reports, parent_reports)+'''</div><img src="overview_actual.png" alt="세 고정 사례의 세 방법 실제 사진 비교"><p><a href="PROCESS_DATA.json">모든 표시 값·출처 JSON</a> · <a href="'''+_url(OLD.OUTPUT/'index.html')+'''">18c 기존 과정</a> · <a href="'''+_url(DRAW.OUTPUT/'index.html')+'''">첫49 e 과정</a></p>'''+''.join(sections)+'''</main></html>'''
    (OUTPUT/'index.html').write_text(page)
    for path, digest in bindings.items():
        actual = Path(path) if Path(path).is_absolute() else C.ROOT/path
        if C.sha(actual) != digest:
            raise ValueError('Bound source/input changed during drawing')
    C.write(OUTPUT/'RENDER_SEAL.json', dict(source_sha256=C.sha(Path(__file__)), input_sha256=bindings,
        assets={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        cases=3, stages_per_case=7, actual_RGB_only=True, predeclared_case_order=list(CASES),
        uniform_methods=list(METHODS), no_GT_pose_corner_reads_or_overlays=True, new_inference_or_fit=0))
    print(dict(output=str(OUTPUT), cases=list(CASES), methods=list(METHODS)), flush=True)


if __name__ == '__main__':
    render()
