"""Actual-image process gallery from sealed predictions; no new inference or fit."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse

from . import common as C
from . import evaluation_interval_v4_2b as E
from . import statewise_evaluation_v4 as V
from .statewise_joint_visualize_v2 import EDGES

OUT = E.OUTPUT.parent / 'process_gallery_20261014_v1'
OLD = C.DOC / 'nonsevere_v1/posefix_sub_comparison_v1'
CURRENT = 'CURRENT_STATEWISE'
LABELS = {'NATIVE_BASE': '원 YOLO + PnP', 'INITIAL_N3_SUBPIX': 'N3+SubPix + PnP',
          'POSEFIX_RAW': 'PoseFix 원출력 + PnP',
          'POSEFIX_SUBPIX_PF_ANCHOR': 'PoseFix+SubPix (PF 기준 제한)',
          'POSEFIX_SUBPIX_YOLO_ANCHOR': 'PoseFix+SubPix (YOLO 기준 제한)',
          CURRENT: '현재 상태별 보정 (N3 입력)'}
FILTERS = ('POSEFIX_RAW', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR',
           'INITIAL_N3_SUBPIX', CURRENT)
COLORS = {'S': '#b763e5', 'B': '#008cff', 'I': '#00b36b', 'O': '#f79300',
          'T': '#ec4545', 'U': '#999999'}
HUMAN = {'DIRECT_VISIBLE': '보임', 'EXTERNAL_OCCLUDED': '외부 가림',
         'SELF_OCCLUDED': '자기 가림', 'UNANNOTATED': '주석 없음'}
STAGE_TITLES = ['1. 첫 입력: N3+SubPix KP와 초기 자세',
                '2. 초기 기하에서 S / B / I 역할 정의',
                '3. RGB 보임 판정 + 화면 안팎 검사',
                '4. B/I 영상 보정 또는 원좌표 유지',
                '5. O 영역 + 불확실한 T 보류',
                '6. 최종 PnP 자세와 S 투영']
STAGE_NOTES = ['점=non-S 입력 KP, 선=초기 PnP 모델 투영. PoseFix 입력과 별개입니다.',
               '모델 투영으로 정한 조건부 역할입니다. B/I의 실제 보임 여부는 다음 단계에서 판단합니다.',
               '색=예측 상태. S는 RGB 판정 제외, T/U는 RGB 미실행일 수 있습니다. 정답과는 별개입니다.',
               '×=원 KP, ●=영상 보정, ○=원예측 fallback. 작은 화살표는 실제 저장된 이동량입니다.',
               '주황 영역=CAL 오차로 확대한 O 위치 범위. 화면과 겹친 조건부 T 원은 U로 보류합니다.',
               '선=하나의 강체 자세 투영. B/I 점=고정한 영상 관측. S=최종 자세 투영만 사용합니다.']


def setup_font():
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'font.size': 9, 'figure.facecolor': 'white'})


def wire(ax, points, color, alpha=1., dashed=False):
    points = np.asarray(points)
    for a, b in EDGES:
        ax.plot(points[[a, b], 0], points[[a, b], 1], color=color, alpha=alpha,
                lw=1.5, linestyle='--' if dashed else '-')


def point(ax, xy, label, color, marker='o', filled=True, size=34, offset=None):
    xy = np.asarray(xy, float)
    if not np.isfinite(xy).all():
        return
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    outside = not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
    if outside:
        at = np.clip(xy, [x0+3, y0+3], [x1-3, y1-3])
        height, width = ax.images[0].get_array().shape[:2]
        canvas_outside = not (0 <= xy[0] < width and 0 <= xy[1] < height)
        delta = xy-at
        arrow = ('>' if delta[0] >= 0 else '<') if abs(delta[0]) >= abs(delta[1]) else ('v' if delta[1] >= 0 else '^')
        ax.scatter(*at, color=color, marker=arrow, s=size)
        text = str(label)+(' (영상 밖 표시)' if canvas_outside else ' (확대 범위 밖 표시)')
    else:
        at = xy; text = str(label)
        if marker in ('x', '+'):
            ax.scatter(*at, color=color, marker=marker, s=size, linewidths=1.3)
        else:
            ax.scatter(*at, c=[color] if filled else 'none', edgecolors=color,
                       marker=marker, s=size, linewidths=1.2)
    dx = 5 if at[0] < (x0+x1)/2 else -5
    first = str(label).split(':')[0].split(' ')[0]
    dy = 8 if first.isdigit() and int(first)%2 else -6
    if offset is not None: dx,dy=offset
    ax.annotate(text, at, xytext=(dx, dy), textcoords='offset points', fontsize=8,
                color=color, ha='left' if dx > 0 else 'right',
                bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': .5})


def viewport(row):
    rt = row['runtime']; g = rt['initial_geometry']; h, w = row['raw_hw']
    hidden = np.asarray(g['initial_self_hidden'], bool)
    raw = np.asarray(rt['original_raw_points'], float)[:8][~hidden]
    points = np.vstack((raw, g['initial_projection_xy'], rt['keypoint_readout']['rigid_model_corners']))
    points = points[np.isfinite(points).all(1)]
    points = np.clip(points, [0, 0], [w, h])
    low, high = points.min(0), points.max(0)
    pad = max(55., (high[0]-low[0])*.16, (high[1]-low[1])*.30)
    return [max(0, low[0]-pad), min(w, high[0]+pad),
            max(0, low[1]-pad*1.4), min(h, high[1]+pad)]


def image_axis(ax, rgb, box, title):
    ax.imshow(rgb); ax.set_xlim(box[:2]); ax.set_ylim(box[3], box[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11, pad=10)


def supported_lines(ax, record):
    for ray in record.get('rays', []):
        if not ray.get('supported') or 'normal' not in ray:
            continue
        xy = np.asarray(ray.get('observed_points', []), float)
        if xy.size == 0:
            continue
        mask = np.asarray(ray.get('inlier_mask', np.ones(len(xy), bool)), bool)
        xy = xy[mask]
        if not len(xy):
            continue
        n = np.asarray(ray['normal']); c = float(ray['offset'])
        tangent = np.array([-n[1], n[0]])
        s = xy @ tangent
        segment = -c*n + np.array([s.min()-3, s.max()+3])[:, None]*tangent
        ax.plot(segment[:, 0], segment[:, 1], color='#ffdb00', lw=1.2, alpha=.9)
        ax.scatter(xy[:, 0], xy[:, 1], color='#ffdb00', s=3, alpha=.65)


def method_projection(entry, row):
    if entry['method'] == CURRENT:
        return np.asarray(row['runtime']['keypoint_readout']['rigid_model_corners'])
    if entry['method'] == 'INITIAL_N3_SUBPIX':
        return np.asarray(row['runtime']['initial_geometry']['initial_projection_xy'])
    p = entry['actual_pose']
    return C.project(C.cuboid(*p['cf_extents']), p['R_cf'], p['centroid'], row['K'])


def prepare_cases():
    sealed, protocol, _ = V._verify_prediction_seal(E.OUTPUT)
    score = C.read(E.OUTPUT / 'POSTSEAL_SCORING.json')
    if score['prediction_seal']['sha256'] != C.sha(E.OUTPUT / 'PREDICTION_SEAL.json'):
        raise ValueError('Current score seal mismatch')
    old_seal = C.read(OLD / 'PREDICTION_SEAL.json')
    if not old_seal['before_scoring'] or C.sha(OLD / 'PREDICTIONS.jsonl.gz') != old_seal['predictions_sha256']:
        raise ValueError('PoseFix comparator seal mismatch')
    old = list(C.rows(OLD / 'SCORED_ROWS.jsonl.gz'))
    old_metrics = C.read(OLD / 'COMPARISON.json')
    by_id = {}
    for r in old:
        if r['partition'] == 'TEST' and r['method'] in LABELS:
            by_id.setdefault(r['id'], {})[r['method']] = r
    current_score = {r['id']: r for r in score['rows'] if r['partition'] == 'TEST' and r['cohort'] == 'PRIMARY_NONSEVERE'}
    cases = []
    for r in sealed:
        if r['id'] not in current_score:
            continue
        fid = r['id']; methods = deepcopy(by_id[fid]); scored = current_score[fid]
        methods[CURRENT] = {'method': CURRENT, 'actual_pose': r['runtime']['actual_pose'],
                            'native_points': r['runtime']['keypoint_readout']['reported_points'],
                            'metric': scored['metrics'][V.ARM_ALL]}
        for method, entry in methods.items():
            m = entry['metric']
            expected = m['translation_cm'] < 5. and m['rotation_deg'] < 5.
            if bool(m['success_5cm_5deg']) != expected:
                raise ValueError('Threshold mismatch: '+fid+' '+method)
        if set(methods) != set(LABELS):
            raise ValueError('Every method must remain available: '+fid)
        cases.append({'row': r, 'scored': scored, 'methods': methods})
    if len(cases) != 49:
        raise ValueError('Keep all49 TEST cases')
    fail = {method: [c['row']['id'] for c in cases if not c['methods'][method]['metric']['success_5cm_5deg']]
            for method in LABELS}
    for method, n in zip(FILTERS, (15, 10, 11, 9, 6)):
        if len(fail[method]) != n:
            raise ValueError('Failure population changed: '+method)
        if method != CURRENT and old_metrics['summary']['TEST/'+method]['frames'] != 49:
            raise ValueError('Comparator population mismatch')
    if len(set().union(*(set(fail[m]) for m in FILTERS))) != 16:
        raise ValueError('Expected16 union failure cases')
    return cases, fail, protocol


def render_case(task):
    setup_font()
    case, output = task; row = case['row']; rt = row['runtime']; g = rt['initial_geometry']
    path = C.ROOT / row['image']
    if C.sha(path) != row['image_sha256']:
        raise ValueError('Actual source RGB changed')
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Cannot decode actual RGB')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    box = viewport(row); raw = np.asarray(rt['original_raw_points'], float)[:8]
    initial = np.asarray(g['initial_projection_xy']); final = np.asarray(rt['keypoint_readout']['rigid_model_corners'])
    hidden = set(np.flatnonzero(g['initial_self_hidden']).tolist())
    records = {r['native_id']: r for r in rt['state_observation']['records']}
    corrections = {r['native_corner']: r for r in rt['visible_correction']['records']}
    measurements = {r['native_corner']: r for r in rt['frozen_visible_measurements']}
    roles = ['S' if k in hidden else g['initial_roles'][k] for k in range(8)]
    before = [records[k].get('original_T_hypothesis', records[k])['state'] for k in range(8)]
    stem = row['id'].replace(':', '__'); folder = Path(output) / stem; folder.mkdir()
    shutil.copy2(path, folder / ('source'+path.suffix))
    aspect=(box[1]-box[0])/(box[3]-box[2])
    fig_height=max(6.5,2*(5.25/aspect+.7)+1.)
    fig, axes = plt.subplots(2, 3, figsize=(17, fig_height))
    for n, ax in enumerate(axes.flat):
        image_axis(ax, rgb, box, STAGE_TITLES[n])
        ax.text(.01, -.08, STAGE_NOTES[n], transform=ax.transAxes, fontsize=8,
                va='top', wrap=True)
    # Initial input: S raw outputs never become evidence in this method.
    wire(axes.flat[0], initial, '#e6bb00', .8)
    for k in range(8):
        if k not in hidden:
            point(axes.flat[0], raw[k], k, '#007fdd', 'x')
    axes.flat[0].text(.02, .03, f'초기 S IDs: {sorted(hidden)} → 점 관측 제외',
                      transform=axes.flat[0].transAxes, bbox={'facecolor':'white','alpha':.85,'pad':3})
    wire(axes.flat[1], initial, '#e6bb00', .6)
    for k, state in enumerate(roles):
        point(axes.flat[1], initial[k], f'{k}:{state}', COLORS[state], 's')
    for k, state in enumerate(before):
        if state == 'S':
            continue
        score = records[k].get('p_visible')
        label = f'{k}:{state}'+(f' {score:.2f}' if score is not None else '')
        point(axes.flat[2], raw[k], label, COLORS[state], 'x')
    axes.flat[2].text(.02, .03, '숫자=보임 점수(물리 확률 아님)', transform=axes.flat[2].transAxes,
                      bbox={'facecolor':'white','alpha':.85,'pad':3})
    for k, measured in measurements.items():
        state = rt['states'][k]; rec = corrections[k]; supported_lines(axes.flat[3], rec)
        axes.flat[3].scatter(*raw[k], color='#555555', marker='x', s=35)
        xy = np.asarray(measured['xy']); corrected = measured['source'] != 'VISIBLE_BACKBONE_FALLBACK'
        shift = float(np.linalg.norm(xy-raw[k]))
        if corrected and shift > 1e-8:
            axes.flat[3].annotate('', xy, xytext=raw[k], arrowprops={'arrowstyle':'->','color':COLORS[state],'lw':1.4})
        label = f'{k}:{state} '+(f'보정 {shift:.1f}px' if corrected else '원좌표')
        point(axes.flat[3], xy, label, COLORS[state], filled=corrected)
    for k, m in measurements.items():
        point(axes.flat[4], m['xy'], f'{k}:{rt["states"][k]}', COLORS[rt['states'][k]], size=24)
    for domain in rt['ot_domains']['domains']:
        if domain['shape'] != 'ellipse':
            raise ValueError('This actual dataset uses ellipse O regions')
        axes.flat[4].add_patch(Ellipse(domain['center_xy'], *[2*x for x in domain['half_axes_px']],
                    angle=domain.get('angle_deg', 0.), fill=False, edgecolor=COLORS['O'], lw=1.8))
        point(axes.flat[4], domain['center_xy'], f'{domain["native_corner"]}:O 영역', COLORS['O'], 'x')
    for guard in rt['truncation_interval_guard']['records']:
        disk = guard.get('localization_disk')
        if disk:
            axes.flat[4].add_patch(Circle(disk['center_xy'], disk['radius_px'], fill=False,
                                        edgecolor=COLORS['U'], linestyle='--', lw=1.2, alpha=.8))
            point(axes.flat[4], disk['center_xy'], f'{guard["native_corner"]}:T→{guard["final_state"]}', COLORS['U'], 'x')
    wire(axes.flat[5], initial, '#e6bb00', .5, True); wire(axes.flat[5], final, '#00babb')
    for k, state in enumerate(rt['states']):
        if state in ('B','I'):
            point(axes.flat[5], measurements[k]['xy'], f'{k}:{state} 관측', COLORS[state], size=26)
        elif state == 'S':
            point(axes.flat[5], final[k], f'{k}:S 투영', COLORS[state], 'x')
        elif state in ('O','T','U'):
            point(axes.flat[5], final[k], f'{k}:{state} 모델 가설', COLORS[state], 's', False)
    metric = case['methods'][CURRENT]['metric']; status = '기준 만족' if metric['success_5cm_5deg'] else '기준 미달'
    suffix = ' / fallback' if case['scored']['actual_pose_fallback'] else (' / 모호한 자세 집합' if case['scored']['material_ambiguity'] else '')
    fig.suptitle(f'{row["id"]} | 현재 방법: 위치 {metric["translation_cm"]:.3f}cm, 회전 {metric["rotation_deg"]:.3f}° / {status}{suffix}', fontsize=13)
    fig.text(.025, .015, '실제 TEST 사진의 저장된 과정입니다. 상태별 과정 입력은 N3+SubPix이며 PoseFix에 이 과정을 적용한 기록은 아닙니다.\n'
             'S의 영상 점 관측은 사용하지 않습니다. B/I 관측과 최종 모델 투영은 서로 다릅니다. 사람 주석은 아래 표의 사후 검증용입니다.', fontsize=9)
    fig.subplots_adjust(left=.025, right=.99, top=.90, bottom=.10, hspace=.38, wspace=.10)
    fig.savefig(folder/'process.png', dpi=160)
    for n, ax in enumerate(axes.flat):
        bounds = ax.get_position(); xpad=.012; ypad=.085
        from matplotlib.transforms import Bbox
        bbox = Bbox.from_extents(max(0,bounds.x0-xpad), max(0,bounds.y0-ypad),
                                min(1,bounds.x1+xpad), min(1,bounds.y1+.05)).transformed(fig.transFigure).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(folder/f'stage{n+1}.png', dpi=190, bbox_inches=bbox)
    plt.close(fig)
    # Separate methods, each with its own archived coordinates and pose.
    fig, axes = plt.subplots(2,3,figsize=(16,fig_height+.3))
    for ax, (method, label) in zip(axes.flat, LABELS.items()):
        entry = case['methods'][method]; m=entry['metric']; ok=m['success_5cm_5deg']
        image_axis(ax,rgb,box,label+f'\n위치 {m["translation_cm"]:.3f}cm / 회전 {m["rotation_deg"]:.3f}° / '+('만족' if ok else '미달'))
        wire(ax,method_projection(entry,row),'#00babb' if ok else '#e74444')
        if method == CURRENT:
            for k,m in measurements.items():point(ax,m['xy'],k,'#2e69d2','x',size=23)
            for k in hidden:point(ax,final[k],f'{k}:S 투영',COLORS['S'],'x',size=23)
            for k,state in enumerate(rt['states']):
                if state in ('O','T','U'):point(ax,final[k],f'{k}:{state} 가설',COLORS[state],'s',False,size=23)
        else:
            for k,xy in enumerate(entry['native_points'][:8]):point(ax,xy,k,'#2e69d2','x',size=23)
    fig.suptitle(row['id']+' | 서로 다른 방법의 저장된 최종 결과 비교',fontsize=13)
    fig.text(.02,.02,'파랑 ×=각 방법의 관측 KP(현재 방법은 B/I만). S/가림점 가설은 별도 표시. 선=각 방법의 PnP 모델 투영.\n'
             'PoseFix/SubPix는 저장된 출력으로 비교합니다. S/B/I/O/T 보정은 N3+SubPix 입력에서 실행됐습니다.',fontsize=9)
    fig.subplots_adjust(left=.02,right=.99,top=.89,bottom=.10,wspace=.08,hspace=.32)
    fig.savefig(folder/'comparison.png',dpi=155);plt.close(fig)
    # Native KP close-ups show genuine supported lines and exact small shifts.
    fig, axes=plt.subplots(2,4,figsize=(15,8))
    details=[]
    for k,ax in enumerate(axes.flat):
        state=rt['states'][k];rec=corrections.get(k,{});m=measurements.get(k)
        q=None if k in hidden else raw[k].tolist()
        if k in hidden:
            ax.set_facecolor('#f2eafa');ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,.53,'S: 직접 영상 관측 없음\n최종 PnP 투영으로만 완성',ha='center',va='center',transform=ax.transAxes)
        elif 0<=raw[k,0]<image.shape[1] and 0<=raw[k,1]<image.shape[0]:
            image_axis(ax,rgb,[max(0,raw[k,0]-28),min(image.shape[1],raw[k,0]+28),
                              max(0,raw[k,1]-28),min(image.shape[0],raw[k,1]+28)],'')
            supported_lines(ax,rec);point(ax,raw[k],'원좌표','#555555','x',offset=(-7,8))
            if m:point(ax,m['xy'],'보정' if m['source']!='VISIBLE_BACKBONE_FALLBACK' else '원좌표 유지',COLORS[state],filled=m['source']!='VISIBLE_BACKBONE_FALLBACK',offset=(7,-8))
        else:
            ax.set_facecolor('#eeeeee');ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,.5,'입력 좌표가 화면 밖\n'+f'{before[k]} → {state}',ha='center',va='center',transform=ax.transAxes)
        human=case['scored']['human_states_native'][k]
        title=f'KP{k}: 역할 {roles[k]} → 판정 {before[k]} → 최종 {state}'
        title+='\n사람 주석: '+HUMAN.get(human,human)
        ax.set_title(title,fontsize=10,color=COLORS[state],pad=8)
        details.append({'native_id':k,'geometric_role':roles[k],'before_interval_state':before[k],'final_state':state,
            'human_state':human,'p_visible_score':records[k].get('p_visible'),'RGB_evaluated':records[k].get('rgb_evaluated'),
            'raw_xy':q,'observed_BI_xy':m['xy'] if m else None,'corrected':bool(m and m['source']!='VISIBLE_BACKBONE_FALLBACK'),
            'measurement_source':m.get('source') if m else None,'correction_reason':rec.get('reason'),
            'correction_record_phase_state':rec.get('state'),'state_reason':records[k].get('reason'),
            'final_model_projection_xy':final[k].tolist(),'readout_source':rt['keypoint_readout']['point_sources'][k],
            'initial_S_raw_or_local_RGB_displayed':False})
    fig.suptitle(row['id']+' | 각 KP의 실제 영상 근거와 보정 여부',fontsize=12)
    fig.text(.02,.025,'노랑=실제 영상에서 지지된 선, 회색 ×=입력, 색 점=최종 B/I 관측.\n'
             '주석은 사후 검증용이며 추론 입력이 아닙니다. S 칸에는 영상 패치를 표시하지 않습니다.',fontsize=9)
    fig.subplots_adjust(left=.035,right=.985,top=.88,bottom=.13,wspace=.22,hspace=.36)
    fig.savefig(folder/'keypoints.png',dpi=165);plt.close(fig)
    images=['process.png','comparison.png','keypoints.png']+[f'stage{k}.png' for k in range(1,7)]
    payload={'id':row['id'],'session':row['session'],'folder':stem,'source_image':stem+'/source'+path.suffix,
             'source_image_sha256':row['image_sha256'],'viewport_xyxy':[box[0],box[2],box[1],box[3]],
             'states':rt['states'],'before_interval_states':before,'geometric_roles':roles,'KP':details,
             'metrics':{m:e['metric'] for m,e in case['methods'].items()},
             'pose_fallback':case['scored']['actual_pose_fallback'],'ambiguous_family':case['scored']['material_ambiguity'],
             'physical_native_compatible':case['scored']['physical_native_compatible'],
             'OT_changes_pose_vs_NO_OT':case['scored']['OT_changes_pose_vs_no_OT'],
             'asset_sha256':{name:C.sha(folder/name) for name in images},
             'process_input':'N3_SUBPIX','PoseFix_statewise_pipeline_executed':False}
    C.write(folder/'CASE.json',payload)
    return payload


HTML = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 KP 추론·판정·보정 과정</title>
<style>body{font-family:system-ui,"NanumGothic",sans-serif;margin:0;background:#f3f5f8;color:#17212c}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:25px;margin:0 0 12px}p{line-height:1.6}.box{background:white;border:1px solid #dfe5eb;border-radius:12px;padding:18px;margin:16px 0}.controls{display:flex;gap:15px;align-items:center;flex-wrap:wrap}select,button{padding:9px;font-size:15px;border:1px solid #bcc9d6;border-radius:7px;background:white}button.active{background:#173d67;color:white}label{display:flex;gap:7px;align-items:center}.legend{display:flex;gap:16px;flex-wrap:wrap;font-weight:600;font-size:14px}.viewer img{width:100%;height:auto;display:block}.viewer{background:white;border-radius:10px;padding:6px;margin:16px 0}.tabs{display:flex;gap:7px;flex-wrap:wrap}.muted{color:#526274;font-size:14px}.warning{color:#994422}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e0e5ea;white-space:nowrap}th{background:#eef3f8}.pass{color:#157045}.fail{color:#bd2b38}.badge{display:inline-block;padding:3px 7px;border-radius:5px;background:#eaf0f7;margin:2px}.case-list{display:flex;flex-wrap:wrap;gap:6px;max-height:180px;overflow:auto}.case-list button{font-size:12px;padding:6px}.note{font-size:14px;border-left:4px solid #178fc0;padding-left:12px}a{color:#1261a4}</style>
<main><h1>실제 사진으로 보는 KP 추론 → 판정 → 보정 → 자세</h1>
<p>같은 clean/moderate TEST49에서 결과가 나온 뒤 정확도 기준을 넘긴 사례를 확인합니다. <strong>미달은 추론 누락이 아니라 위치 ≥5cm 또는 회전 ≥5°입니다.</strong></p>
<div class="note">방법 비교는 별도 실험들의 결과입니다. 아래 여섯 단계의 상태별 과정은 <strong>N3+SubPix 입력</strong>에서 실행됐습니다. PoseFix에 이 상태별 과정을 실행한 기록으로 읽으면 안 됩니다.</div>
<div class="box controls"><label>기준 방법 <select id="method"></select></label><label><input id="failedOnly" type="checkbox" checked>기준 미달 사진만</label><span id="count"></span><label>사례 <select id="case"></select></label><button id="previous">이전</button><button id="next">다음</button></div>
<div id="caseList" class="case-list"></div>
<div class="box"><div class="legend"><span style="color:#b763e5">S 자기 가림</span><span style="color:#008cff">B 보이는 경계</span><span style="color:#00b36b">I 보이는 내부</span><span style="color:#f79300">O 외부 가림 예측</span><span style="color:#ec4545">T 화면 밖 가설</span><span style="color:#777">U 판정 보류</span></div><p id="summary"></p><div class="scroll"><table id="metrics"></table></div></div>
<div class="tabs" id="tabs"></div><p id="stageNote" class="muted"></p><div class="viewer"><a id="imageLink" target="_blank"><img id="image" alt="실제 영상에 저장된 단계별 결과를 표시한 그림"></a></div>
<div class="box"><h2 style="font-size:19px">각 KP의 판정과 보정 기록</h2><p class="muted">보임 점수는 판정기의 출력이며 검증된 물리 확률이 아닙니다. 사람 주석은 사후 확인용입니다. 숫자는 원 영상의 픽셀 좌표입니다.</p><div class="scroll"><table id="kp"></table></div></div>
<details class="box"><summary>이 결과를 읽을 때 확인할 점</summary><p>초기 B/I는 기하학적 역할이고 실제 보임 판정은 다음 단계에서 합니다. B/I에 영상 근거가 부족하면 원예측을 유지합니다. S는 영상 점 관측 없이 최종 자세로만 투영합니다. O 원은 가림 물체의 윤곽이 아니라 CAL 오차로 확대한 위치 범위입니다. 불확실한 T는 U로 보류했으며 최종 TEST의 T는 0개입니다. 현재 TEST에서는 O 영역을 제거해도 선택 자세가 같습니다. 모형과 실제 파렛트의 대응 불일치가 있는 사례도 포함했습니다. 평가 기준은 기존 주석에서 파생됐고, 이미 반복 사용한 탐색 데이터입니다.</p></details>
<p class="muted">저장된 실제 결과의 시각화입니다. 새 학습·새 추론·새 자세 선택은 수행하지 않았습니다. <a href="VISUALIZATION_RECEIPT.json">출처 및 그림 검증 기록</a> · <a href="FAILURE_SETS.json">각 방법의 미달 사진 목록</a></p></main>
<script id="data" type="application/json">__DATA__</script><script>
const DATA=JSON.parse(document.getElementById('data').textContent), CASES=DATA.cases, LABELS=DATA.labels;
const $=x=>document.getElementById(x),esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const human={DIRECT_VISIBLE:'보임',EXTERNAL_OCCLUDED:'외부 가림',SELF_OCCLUDED:'자기 가림',UNANNOTATED:'주석 없음'};
const tabs=[['process.png','전체 6단계'],['comparison.png','방법별 비교'],['keypoints.png','각 KP 확대'],...Array.from({length:6},(_,k)=>[`stage${k+1}.png`,`${k+1}단계`])];
let subset=[],active=CASES.find(c=>c.id==='eval_pallet07:1778652130452698368')?.id||CASES[0].id,tab='process.png';
$('method').innerHTML=DATA.filters.map(m=>`<option value="${m}">${esc(LABELS[m])} — 미달 ${DATA.failure_sets[m].length}/49</option>`).join('');
$('tabs').innerHTML=tabs.map(([p,n])=>`<button data-tab="${p}">${esc(n)}</button>`).join('');
function metric(m){return `${m.translation_cm.toFixed(3)}cm / ${m.rotation_deg.toFixed(3)}°`;}
function xy(p){return p?`(${p[0].toFixed(1)}, ${p[1].toFixed(1)})`:'—';}
function updateFilter(){let m=$('method').value;subset=CASES.filter(c=>!$('failedOnly').checked||!c.metrics[m].success_5cm_5deg);if(!subset.some(c=>c.id===active))active=subset[0]?.id;$('count').textContent=`표시 ${subset.length}/49장`; $('case').innerHTML=subset.map(c=>`<option value="${esc(c.id)}">${esc(c.id)} · ${metric(c.metrics[m])}</option>`).join('');$('caseList').innerHTML=subset.map((c,k)=>`<button data-id="${esc(c.id)}">${k+1}. ${esc(c.id.split(':').pop())}</button>`).join('');render();}
function render(){let c=CASES.find(x=>x.id===active);if(!c)return;$('case').value=active;const m=c.metrics.CURRENT_STATEWISE;let extra=!c.physical_native_compatible?' · 등록 모델과 reference 대응 불일치':'';extra+=c.pose_fallback?' · fallback':c.ambiguous_family?' · 모호한 자세 집합':'';$('summary').innerHTML=`<strong>${esc(c.id)}</strong> — 현재 ${metric(m)} · <span class="${m.success_5cm_5deg?'pass':'fail'}">${m.success_5cm_5deg?'기준 만족':'기준 미달'}</span>${esc(extra)}<br>기하 역할: ${c.geometric_roles.join(' ')} → 초기 판정: ${c.before_interval_states.join(' ')} → 최종 상태: ${c.states.join(' ')}`;
$('metrics').innerHTML='<tr><th>방법</th><th>위치 오차</th><th>회전 오차</th><th>기준</th></tr>'+Object.keys(LABELS).map(k=>{let x=c.metrics[k];return `<tr><td>${esc(LABELS[k])}</td><td>${x.translation_cm.toFixed(3)}cm</td><td>${x.rotation_deg.toFixed(3)}°</td><td class="${x.success_5cm_5deg?'pass':'fail'}">${x.success_5cm_5deg?'만족':'미달'}</td></tr>`}).join('');
$('kp').innerHTML='<tr><th>KP</th><th>기하 역할</th><th>판정 → 최종</th><th>사람 주석</th><th>보임 점수</th><th>RGB 판정</th><th>처리</th><th>입력 좌표</th><th>B/I 관측 좌표</th><th>최종 모델 투영</th></tr>'+c.KP.map(p=>{let action=p.final_state==='S'?'PnP 투영만':p.observed_BI_xy?(p.corrected?'영상 보정':'원좌표 유지'):p.final_state==='O'?'O 영역 제약':p.final_state==='T'?'화면 밖 제약':'판정 보류';return `<tr><td>${p.native_id}</td><td>${p.geometric_role}</td><td>${p.before_interval_state} → ${p.final_state}</td><td>${human[p.human_state]||esc(p.human_state)}</td><td>${p.p_visible_score==null?'—':p.p_visible_score.toFixed(3)}</td><td>${p.RGB_evaluated?'실행':'제외/미실행'}</td><td>${action}</td><td>${xy(p.raw_xy)}</td><td>${xy(p.observed_BI_xy)}</td><td>${xy(p.final_model_projection_xy)}</td></tr>`}).join('');
let url=`${c.folder}/${tab}`;$('image').src=url;$('imageLink').href=url;$('stageNote').textContent=tab.startsWith('stage')?DATA.stage_notes[Number(tab[5])-1]:tab==='comparison.png'?'각 칸은 별도 방법입니다. PoseFix 원출력과 두 SubPix 조건, N3 초기값 및 현재 결과를 비교합니다.':tab==='keypoints.png'?'작은 보정 이동을 확대했습니다. S는 영상 패치를 표시하지 않고, 노랑 선은 실제 영상에서 지지된 선만 표시합니다.':'전체 흐름을 한 그림에 표시했습니다. 1~6단계 버튼을 눌러 크게 볼 수 있습니다.';document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));document.querySelectorAll('[data-id]').forEach(b=>b.classList.toggle('active',b.dataset.id===active));}
$('method').onchange=updateFilter;$('failedOnly').onchange=updateFilter;$('case').onchange=()=>{active=$('case').value;render()};$('tabs').onclick=e=>{if(e.target.dataset.tab){tab=e.target.dataset.tab;render()}};$('caseList').onclick=e=>{if(e.target.dataset.id){active=e.target.dataset.id;render()}};
function step(d){let i=subset.findIndex(c=>c.id===active);if(subset.length){active=subset[(i+d+subset.length)%subset.length].id;render()}};$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);updateFilter();
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();output=a.output.resolve()
    if output.exists():raise FileExistsError('Preserve previous gallery')
    cases,failures,protocol=prepare_cases();output.mkdir(parents=True)
    bindings={str(path.relative_to(C.ROOT)):C.sha(path) for path in (
        E.OUTPUT/'PREDICTIONS.jsonl.gz',E.OUTPUT/'PREDICTION_SEAL.json',E.OUTPUT/'POSTSEAL_SCORING.json',
        OLD/'SCORED_ROWS.jsonl.gz',OLD/'COMPARISON.json',OLD/'PREDICTIONS.jsonl.gz',OLD/'PREDICTION_SEAL.json',Path(__file__))}
    rendered={}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(render_case,(case,str(output))):case['row']['id'] for case in cases}
        for job in as_completed(jobs):
            result=job.result();rendered[result['id']]=result
            print(f'{len(rendered)}/49 {result["id"]}',flush=True)
    ordered=[rendered[c['row']['id']] for c in cases]
    for path,digest in bindings.items():
        if C.sha(C.ROOT/path)!=digest:raise RuntimeError('Source/prediction changed: '+path)
    C.write(output/'FAILURE_SETS.json',{'strict_success':'translation_cm<5 AND rotation_deg<5','TEST_frames':49,
             'method_labels':LABELS,'failure_sets':failures,'five_comparator_union':sorted(set().union(*(set(failures[m]) for m in FILTERS)))})
    C.write(output/'CASES.json',ordered)
    bundle={'cases':ordered,'labels':LABELS,'filters':FILTERS,'failure_sets':failures,'stage_notes':STAGE_NOTES}
    (output/'index.html').write_text(HTML.replace('__DATA__',json.dumps(bundle,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    shutil.copy2(__file__,output/'renderer_snapshot.py')
    receipt={'schema':'ACTUAL49_KEYPOINT_PROCESS_GALLERY_V1','all49_TEST_frames_preserved':True,'primary_severe':0,
             'failure_counts':{m:len(failures[m]) for m in LABELS},'five_comparator_failure_union_count':16,
             'input_sha256':bindings,'new_inference':0,'new_pose_fits':0,'new_training':0,
             'process_method':'N3_SUBPIX -> StatewiseIntervalV4_2, frozen actual RGB records',
             'PoseFix_statewise_pipeline_executed':False,'human_labels_posthoc_only':True,
             'source_RGB_files_verified':49,'initial_S_raw_local_RGB_displayed':False,
             'state_definitions_are_conditional_not_human_truth':True,'all_BI_raw_fallbacks_shown':True,
             'final_BI_observations_and_model_projections_separate':True,
             'index_sha256':C.sha(output/'index.html'),'CASES_sha256':C.sha(output/'CASES.json'),
             'FAILURE_SETS_sha256':C.sha(output/'FAILURE_SETS.json'),'rendered':ordered}
    C.write(output/'VISUALIZATION_RECEIPT.json',receipt)
    print('Complete gallery: '+str(output/'index.html'))


if __name__=='__main__':main()
