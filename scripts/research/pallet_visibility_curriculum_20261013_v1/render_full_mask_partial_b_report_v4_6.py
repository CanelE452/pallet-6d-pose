"""Render saved full-mask and partial-B observations after sealed scoring.

No model inference, boundary fit, optimization or parameter selection occurs.
Measured image points, line observations, broad locality assumptions and PnP
model projections use separate symbols. Reference geometry is posthoc only.
"""
from __future__ import annotations

import os
for _thread_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_key] = '1'

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager, patheffects
from matplotlib.patches import Circle, Ellipse, Rectangle
import numpy as np

from . import common as C
from . import evaluate_full_mask_partial_b_v4_6 as D
from . import statewise_evaluation_v4 as E
from . import render_incompatible_pose_compare_v2 as COMP


TARGET = D.TARGET
OUTPUT = D.PARENT.BOUNDARY / 'full_mask_partial_b_report_v4_6'
FONT = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
BLUE = '#00bbff'
GREEN = '#83f65b'
ORANGE = '#ff922a'
YELLOW = '#ffe377'
VIOLET = '#ed7cea'
WHITE = '#f4f7fa'
ARM_NAMES = {
    D.ARMS[0]: '이전 방법: 기존 마스크 / B fallback',
    D.ARMS[1]: '새 마스크: B0·B3의 2D 보정',
    D.ARMS[2]: '새 마스크 + B5의 선 부분 관측',
}


def _binding(path):
    path = Path(path).resolve()
    return {'path': _path_key(path), 'sha256': C.sha(path)}


def _path_key(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else str(path)


def _load(evaluation):
    evaluation = Path(evaluation).resolve()
    complete = C.read(evaluation / 'PREDICTION_COMPLETE.json')
    if (complete['rows'], complete['CAL'], complete['TEST'], complete['severe']) != (95, 46, 49, 0):
        raise ValueError('Require the complete nonsevere population before report')
    protocol = C.read(evaluation / 'PROTOCOL.json')
    E._verify_sources(protocol['source_sha256'])
    E._verify_sources(protocol['input_sha256'])
    E._verify_sources(protocol['protected_user_sha256'])
    bindings = {}
    for partition, count in (('CAL', 46), ('TEST', 49)):
        seal = C.read(evaluation / f'{partition}_PREDICTION_SEAL.json')
        journal = evaluation / f'{partition}_PREDICTIONS.jsonl.gz'
        if (seal['rows'] != count or C.sha(journal) != seal['predictions_sha256']
                or C.sha(evaluation / 'PROTOCOL.json') != seal['protocol_sha256']
                or not seal['before_new_annotation_score']):
            raise ValueError('Prediction seal mismatch')
        for path in (journal, evaluation / f'{partition}_PREDICTION_SEAL.json'):
            bindings.update({str(path.relative_to(C.ROOT)): C.sha(path)})
    score_seal = C.read(evaluation / 'SCORE_SEAL.json')
    scores = {}
    for partition in ('CAL', 'TEST'):
        path = evaluation / f'{partition}_POSE_SCORE.json'
        if C.sha(path) != score_seal[f'{partition}_pose_score_sha256']:
            raise ValueError('Postseal score binding mismatch')
        scores[partition] = C.read(path)
        if not scores[partition]['all_frames_include_fallbacks']:
            raise ValueError('Report must include fallback frames')
        bindings[str(path.relative_to(C.ROOT))] = C.sha(path)
    rows = list(C.rows(evaluation / 'TEST_PREDICTIONS.jsonl.gz'))
    if len(rows) != 49 or len({r['id'] for r in rows}) != 49:
        raise ValueError('TEST population changed')
    row = next(r for r in rows if r['id'] == TARGET)
    g = row['runtime']['initial_geometry']
    if row['states'] != ['B', 'I', 'U', 'B', 'O', 'B', 'B', 'S']:
        raise ValueError('Target state record changed')
    mask_dir = C.ROOT / protocol['mask_directory']
    mask_seal = C.read(mask_dir / 'SEAL.json')
    if C.sha(mask_dir / 'RECORDS.json') != mask_seal['records_sha256']:
        raise ValueError('Mask records changed')
    meta = next(r for r in C.read(mask_dir / 'RECORDS.json') if r['id'] == TARGET)
    mask_path = mask_dir / meta['mask_npz']
    if C.sha(mask_path) != meta['mask_file_sha256']:
        raise ValueError('Actual mask binding changed')
    with np.load(mask_path, allow_pickle=False) as z:
        masks = {k: np.array(z[k]) for k in ('pallet_mask_raw', 'pallet_mask_clean', 'previous_pallet_mask')}
    image_path = C.ROOT / row['image']
    if C.sha(image_path) != row['image_sha256'] or meta['image_sha256'] != row['image_sha256']:
        raise ValueError('Actual source photo changed')
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    if image.shape[:2] != tuple(g['image_hw']):
        raise ValueError('Photo and native K canvas differ')
    # Source predictions, masks and their postseal score are verified before
    # this reference-only read. Never infer any image KP from these corners.
    reference_binding = scores['TEST']['target_binding']
    reference_path = C.ROOT / reference_binding['path']
    if C.sha(reference_path) != reference_binding['sha256']:
        raise ValueError('Historical posthoc reference changed')
    labels = C.read(reference_path)['rows']
    if isinstance(labels, list):
        labels = {r['id']: r for r in labels}
    truth = labels[TARGET]['target']['truth']
    reference_wire = C.project(C.cuboid(*truth['body_xyz']), truth['body_R'], truth['t'], g['K'])
    # Photo wire uses each arm's own native model/basis. No C2 reordering,
    # manual camera rotation or GT-selected corner correspondence is applied.
    xyz = np.asarray(g['native_xyz'], float)
    projection = {arm: C.project(xyz, value['actual_pose']['R_cf'],
                              value['actual_pose']['centroid'], g['K'])
                  for arm, value in row['arms'].items()}
    if row['contract']['changed_complete_B_ids'] != [0, 3]:
        raise ValueError('Target complete observation inventory differs')
    if [r['native_corner'] for r in row['partial_boundary_observation']['boundary_observations']] != [5]:
        raise ValueError('Target partial observation inventory differs')
    raw = {}
    for k in range(8):
        if g['initial_self_hidden'][k] or row['states'][k] == 'S':
            continue
        raw[k] = list(row['old_correction']['raw_points'][k])
    for path in (Path(__file__), FONT, image_path, mask_path, mask_dir / 'RECORDS.json',
                 mask_dir / 'SEAL.json', evaluation / 'PROTOCOL.json',
                 evaluation / 'PREDICTION_COMPLETE.json', evaluation / 'SCORE_SEAL.json', reference_path):
        bindings[_path_key(path)] = C.sha(path)
    return dict(row=row, geometry=g, raw=raw, image=image, image_path=image_path,
                mask_meta=meta, masks=masks, scores=scores, projection=projection,
                reference_wire=reference_wire, reference_truth=truth, bindings=bindings,
                protected=protocol['protected_user_sha256'])


def _label(ax, text, point, color, offset=(7, -12), size=11):
    ax.annotate(text, point, xytext=offset, textcoords='offset points', color=color,
                fontsize=size, fontweight='bold', zorder=30,
                path_effects=[patheffects.withStroke(linewidth=2.7, foreground='#17202c')])


def _photo(ax, context, title, limits=None):
    image = context['image']; h, w = image.shape[:2]
    ax.set_facecolor('#dfe5ec'); ax.imshow(image)
    ax.set_xlim(*(limits[0] if limits else (-8, w+8)))
    ax.set_ylim(*(limits[1] if limits else (h+12, -8)))
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel('영상 x (px)'); ax.set_ylabel('영상 y (px)')
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor='#95a3b5', linestyle='--', linewidth=1))


def _point(ax, point, text='', color=BLUE, marker='x', offset=(7, -12), size=75):
    ax.scatter(*point, s=size, color=color, marker=marker, linewidths=2, zorder=25)
    if text: _label(ax, text, point, color, offset)


def _wire(ax, points, color=ORANGE, dashed=False):
    points = np.asarray(points)
    for a, b in COMP.EDGES:
        ax.plot(points[[a,b],0], points[[a,b],1], c='#162030', lw=3.8, alpha=.7,
                ls='--' if dashed else '-', zorder=5)
        ax.plot(points[[a,b],0], points[[a,b],1], c=color, lw=1.8,
                ls='--' if dashed else '-', zorder=6)


def _line(ax, normal, offset, interval, color=GREEN, dashed=False, width=2.5):
    n = np.asarray(normal, float); t = np.array([-n[1], n[0]])
    p = -float(offset)*n + np.asarray(interval)[:,None]*t
    ax.plot(p[:,0], p[:,1], c='#152030', lw=width+2, alpha=.65, zorder=13,
            ls='--' if dashed else '-')
    ax.plot(p[:,0], p[:,1], c=color, lw=width, zorder=14, ls='--' if dashed else '-')


def _rays(record):
    items = []
    for origin, values in (('expanded', record.get('expanded_rays', [])),
                           ('selected', record.get('selected_rays', [])),
                           ('recentered', [x.get('remeasured_ray') for x in record.get('observed_line_recentering', [])])):
        for value in values:
            if isinstance(value, dict): items.append((origin, value))
    return items


def _draw_ray(ax, ray, color=GREEN):
    points = np.asarray(ray.get('observed_points', []), float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        return
    inliers = np.asarray(ray.get('inlier_mask', []), bool)
    if inliers.shape != (len(points),):
        inliers = np.ones(len(points), dtype=bool)
    ax.scatter(points[:,0], points[:,1], c='#f9efbc', s=12, zorder=16)
    if not ray.get('supported') or not inliers.any() or 'normal' not in ray:
        return
    n = np.asarray(ray['normal']); t = np.array([-n[1], n[0]])
    s = points[inliers] @ t
    _line(ax, n, ray['offset'], [s.min(), s.max()], color)
    ax.scatter(points[inliers,0], points[inliers,1], c=color, s=20, zorder=17)


def _saved(fig, path):
    fig.savefig(path, dpi=145, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _build(context, output):
    row=context['row']; rec={r['native_corner']:r for r in row['correction']['boundary_recovery_records']}
    fit={r['native_corner']:r for r in row['frozen_BI_fit']}
    partial=row['partial_boundary_observation']['boundary_observations'][0]
    h,w=context['image'].shape[:2]
    assets=output/'assets';assets.mkdir()
    stages=[]
    def add(filename,title,caption,draw):
        draw(assets/filename)
        stages.append(dict(image='assets/'+filename,title=title,caption=caption))
    def masks(path):
        fig,axes=plt.subplots(1,3,figsize=(17,5.4))
        for ax,key,title in zip(axes,('previous_pallet_mask','pallet_mask_raw','pallet_mask_clean'),
                               ('이전 MobileSAM','새 MobileSAM 원본','새 마스크 작은 성분 제거')):
            _photo(ax,context,title)
            rgba=np.zeros((h,w,4));rgba[:,:,:3]=[.34,.91,.51];rgba[:,:,3]=.38*(context['masks'][key]>.5)
            ax.imshow(rgba);ax.axis('off')
        box=context['mask_meta']['provider_provenance']['prompt_box_xyxy']
        for ax in axes[1:]:
            ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                   fill=False,ec=BLUE,lw=1.4,ls='--'))
        fig.tight_layout();_saved(fig,path)
    add('01_masks.png','1. 보정에 사용한 마스크 범위',
        '같은 실제 사진입니다. 새 프롬프트는 고정 초기 모델 전체의 투영 범위와 영상 안 비S 예측점 범위를 포함합니다. 점 프롬프트는 사용하지 않았습니다. Tiny 상태 분류·S/B/I/O/U·초기 자세는 바꾸지 않았습니다. 작은 성분만 제거했고 구멍 채우기나 윤곽 합성은 하지 않았습니다.',masks)
    def overview(path):
        fig,ax=plt.subplots(figsize=(11,7.8));_photo(ax,context,'2. 새 실제 영상 2D 보정과 남은 fallback')
        for k in (0,1,3,5,6):
            old=context['raw'][k];new=fit[k]['xy']
            _point(ax,old,f'raw{k}',WHITE,'o',(-27,14),45)
            if k in (0,3):
                ax.annotate('',xy=new,xytext=old,arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.8})
                _point(ax,new,f'B{k} 보정',BLUE,'x',(9,-10))
            elif k!=5:
                _point(ax,new,f'{row["states"][k]}{k} fallback',YELLOW,'x',(8,-12))
        display=partial['association_provenance']['normal_projected_display_xy']
        _point(ax,display,'B5 normal 표시',GREEN,'D',(-135,-16),50)
        _point(ax,context['projection'][D.ARMS[2]][5],'P5 모델',ORANGE,'s',(12,14),45)
        ax.text(.02,.03,'○ 원 예측 / × 2D 관측 / ◇ 선 위 표시용 점 / □ PnP 모델 투영',
                transform=ax.transAxes,color=WHITE,fontsize=11,bbox={'facecolor':'#15202c','alpha':.8})
        fig.tight_layout();_saved(fig,path)
    add('02_observations.png','2. B0·B3는 2D 보정, B5는 부분 관측, B6은 보류',
        'B0·B3는 두 실제 RGB 경계선과 기존 품질·유한 표본 지지 조건을 통과한 관측점입니다. B5는 한 선만 통과해 정확한 2D KP를 만들지 않습니다. B6은 충분히 검증된 선이 없어 기존 예측점 fallback을 유지합니다. I1도 이번 변경에서 보정하지 않았습니다.',overview)
    def details(path):
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        limits={0:((-10,100),(405,305)),3:((-10,120),(445,335)),5:((465,585),(350,265)),6:((470,585),(410,310))}
        for ax,k in zip(axes.flat,(0,3,5,6)):
            state='2D 보정' if k in (0,3) else '선 1개 부분 관측' if k==5 else '미보정 fallback'
            _photo(ax,context,f'B{k}: {state}',limits[k])
            seen=set()
            for origin,ray in _rays(rec[k]):
                signature=json.dumps(ray,sort_keys=True)
                if signature in seen:continue
                seen.add(signature);_draw_ray(ax,ray)
            _point(ax,context['raw'][k],f'raw{k}',WHITE,'o',(7,13),45)
            if k in (0,3):_point(ax,fit[k]['xy'],f'B{k} 보정',BLUE,'x')
            elif k==5:
                _point(ax,partial['association_provenance']['normal_projected_display_xy'],'표시 전용',GREEN,'D',(-74,-14),45)
            else:_point(ax,fit[k]['xy'],'fallback',YELLOW,'x',(7,-12))
        fig.tight_layout();_saved(fig,path)
    add('03_actual_line_samples.png','3. 실제 RGB 경계 표본과 통과한 선',
        '작은 점은 저장된 실제 검색 표본, 녹색 선분은 품질 조건을 통과한 선의 실제 inlier 구간입니다. 검증되지 않은 선을 KP로 채우지 않았습니다. B5→4는 RGB 표본 12개가 지지하지만 B6의 표본은 최종 선 조건을 통과하지 못했습니다.',details)
    def partialview(path):
        line=partial['lines'][0];prov=line['provenance'];n=np.asarray(line['normal_unit_xy']);b=line['offset_px']
        actual=prov['actual_observed_inlier_tangent_interval_px'];interval=line['support_tangent_interval_px'];margin=line['support_extrapolation_px']
        fig,axes=plt.subplots(1,2,figsize=(15,6.2))
        _photo(axes[0],context,'B5: 실제 선의 normal 방향만 관측',((470,590),(345,275)))
        _photo(axes[1],context,'허용 tangent 범위는 넓은 CAL 가정',((-30,900),(520,220)))
        for ax in axes:
            _line(ax,n,b,[interval[0]-margin,interval[1]+margin],GREEN,True,1.5)
            _line(ax,n,b,actual,GREEN,False,3)
            _point(ax,context['raw'][5],'raw5',WHITE,'o',(-43,14),45)
            display=partial['association_provenance']['normal_projected_display_xy']
            _point(ax,display,'normal 표시',GREEN,'D',(-85,-15),50)
            _point(ax,context['projection'][D.ARMS[2]][5],'P5 모델',ORANGE,'s',(8,14),45)
        axes[0].annotate('',xy=display,xytext=context['raw'][5],arrowprops={'arrowstyle':'->','color':GREEN,'lw':2})
        axes[1].text(.02,.04,'굵은 실선: 실제 inlier 구간\n점선: CAL raw-locality 범위 + 기존 11px 여유',
            transform=axes[1].transAxes,color='#203040',fontsize=11,bbox={'facecolor':'white','alpha':.9})
        fig.tight_layout();_saved(fig,path)
    add('04_B5_partial.png','4. B5의 표시점은 정확한 KP가 아닙니다',
        'B5의 원 2D 점 likelihood를 최종 목적식에서 제거하고 선의 수직(normal) 잔차만 넣었습니다. 선 방향(tangent)은 기존 CAL 비S 최대 오차 반경 311.54px의 raw 중심 구간과 실제 inlier 구간을 합친 넓은 hard 범위입니다. ◇는 원점의 수직 투영을 설명용으로 표시한 것이며 full KP·정답·교점 관측이 아닙니다. 새 B 전용 확률 보장이나 공분산 재학습은 없습니다.',partialview)
    def ranges(path):
        fig,ax=plt.subplots(figsize=(10.4,10));_photo(ax,context,'5. 기존 O/U 영역과 부분 관측을 동시에 만족',((-15,675),(835,-20)))
        # Domains remain in the frozen upstream parent rather than rebuilt
        # from a candidate pose or posthoc reference.
        old_data=C.read(D.PARENT.BOUNDARY/'stepwise_case_216683_v4/PROCESS_DATA.json')
        for domain in old_data['domain_observations']['O_T']+row['u_region_observation']['u_regions']:
            k=domain['native_corner'];center=domain['center_xy'];axespx=domain['half_axes_px'];color='#ffb52e' if domain['state']=='O' else '#98a5b5'
            ax.add_patch(Ellipse(center,2*axespx[0],2*axespx[1],angle=domain.get('angle_deg',0),fill=False,ec=color,lw=2,ls='--'))
            _point(ax,center,f'{domain["state"]}{k} 중심',color,'+',(-110,16),40)
            _point(ax,context['projection'][D.ARMS[2]][k],f'P{k} 모델',ORANGE,'s',(12,-17),45)
        _wire(ax,context['projection'][D.ARMS[2]],ORANGE)
        _point(ax,context['projection'][D.ARMS[2]][7],'S7 모델만',VIOLET,'s',(9,12),40)
        ax.text(20,805,'회색 영역 = 실제 영상 밖 / U는 canvas로 자르지 않음',fontsize=12,color='#263b50')
        fig.tight_layout();_saved(fig,path)
    add('05_OT_U_joint.png','5. O/U 영역·B 부분 범위·기하 조건은 함께 적용',
        'O4의 실제 CAL 경험 영역과 U2의 보수적 raw 중심 영역은 바꾸지 않았습니다. U2는 영상 밖 예측으로 RGB 점 관측 없이 범위만 사용합니다. S7은 새 RGB 관측 없이 PnP 모델 투영만 표시합니다. 그림의 모든 □는 추정 모델 위치이며 영상에서 검출한 점이 아닙니다.',ranges)
    def poses(path,reference=False):
        fig,axes=plt.subplots(1,3,figsize=(18,6.5))
        target=context['scores']['TEST']['target']['metrics']
        for ax,arm in zip(axes,D.ARMS):
            m=target[arm];_photo(ax,context,ARM_NAMES[arm]+f'\n{m["translation_cm"]:.2f}cm / {m["rotation_deg"]:.2f}°')
            _wire(ax,context['projection'][arm],ORANGE)
            if reference:
                _wire(ax,context['reference_wire'],GREEN,True)
            else:
                for k in (0,1,3,5,6):
                    if arm==D.ARMS[2] and k==5:
                        _line(ax,partial['lines'][0]['normal_unit_xy'],partial['lines'][0]['offset_px'],
                              partial['lines'][0]['provenance']['actual_observed_inlier_tangent_interval_px'],GREEN)
                        continue
                    xy=context['raw'][k] if arm==D.ARMS[0] else fit[k]['xy']
                    color=BLUE if arm!=D.ARMS[0] and k in(0,3) else YELLOW
                    _point(ax,xy,str(k),color,'x',size=45)
                _point(ax,context['projection'][arm][7],'S7',VIOLET,'s',size=35)
        fig.tight_layout();_saved(fig,path)
    add('06_three_poses.png','6. 점/선 관측을 바꾼 실제 세 가지 PnP 결과',
        '주황은 각 arm에서 실제 저장된 PnP 모델 투영입니다. B0·B3의 보정과 B5의 부분 관측을 넣었지만 이 사진의 위치오차는 10.11 → 13.61 → 14.60cm로 악화했습니다. 회전오차는 4.73 → 4.29 → 4.54°입니다. 형상·Q·상태·prior·CAL fitΣ/가중치·O/U 범위는 동일합니다.',lambda p:poses(p))
    add('07_reference_comparison.png','7. 봉인된 예측 후의 reference 비교',
        '녹색 점선은 저장된 body_R/body_xyz와 reference 중심을 투영한 물리 box입니다. 주황은 각 방법의 native 모델입니다. 초록 코너를 영상 KP 관측이나 동일 native 번호의 정답으로 사용하지 않았고, 수동 90° 회전·번호 재정렬·다시 맞추기를 적용하지 않았습니다. 이 사례는 native 번호 등록 검사 불일치 사례이며 모든49 점수에 그대로 포함됩니다.',lambda p:poses(p,True))
    return stages


def _table(scores):
    body=[]
    for partition in ('CAL','TEST'):
        for arm in D.ARMS:
            summary=scores[partition]['summary_all'][arm]
            body.append(f'<tr><td>{partition}</td><td>{html.escape(ARM_NAMES[arm])}</td><td>{summary["translation_cm"]["mean"]:.6f}</td><td>{summary["rotation_deg"]["mean"]:.6f}</td><td>{summary["successes_5cm_5deg"]}/{summary["frames"]}</td><td>{summary["fallback_count"]}</td></tr>')
    return '<table><thead><tr><th>전체 집합</th><th>방법</th><th>평균 위치(cm)</th><th>평균 회전(°)</th><th>5cm·5° 성공</th><th>fallback 포함</th></tr></thead><tbody>'+''.join(body)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=D.OUTPUT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError('Preserve existing report: '+str(output))
    font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name();plt.rcParams['axes.unicode_minus']=False
    context=_load(args.evaluation)
    process_path=D.PARENT.BOUNDARY/'stepwise_case_216683_v4/PROCESS_DATA.json'
    context['bindings'][str(process_path.relative_to(C.ROOT))]=C.sha(process_path)
    output.mkdir(parents=True);stages=_build(context,output)
    photo=output/'assets/source_actual.png';shutil.copyfile(context['image_path'],photo)
    if C.sha(photo)!=C.sha(context['image_path']):raise ValueError('Source image copy changed')
    target=context['scores']['TEST']['target'];row=context['row']
    report=dict(schema='ACTUAL_FULL_MASK_PARTIAL_B_CASE_V4_6',id=TARGET,
        input_bindings=context['bindings'],source_photo='assets/source_actual.png',
        state_native=row['states'],initial_S_ids=[7],raw_nonS=context['raw'],
        initial_geometry=deepcopy(context['geometry']),mask_provenance=context['mask_meta']['provider_provenance'],
        changed_complete_B_ids=row['contract']['changed_complete_B_ids'],
        complete_B_measurements=[m for m in row['frozen_BI_fit'] if m['native_corner'] in (0,3)],
        retained_I_and_B_fallback_ids=[1,6],partial_B_observation=deepcopy(row['partial_boundary_observation']),
        partial_B_reported_full_2D_xy=None,target_score=target,
        saved_arm_poses={arm:row['arms'][arm]['actual_pose'] for arm in D.ARMS},
        model_projections={arm:q.tolist() for arm,q in context['projection'].items()},
        reference_body_wire_xy=context['reference_wire'].tolist(),
        reference_wire_basis='body_R/body_xyz; postseal only; no native identity correspondence',
        summaries={partition:context['scores'][partition]['summary_all'] for partition in ('CAL','TEST')},
        stages=stages,all95_include_fallbacks=True,TEST_reused_exploratory=True,
        S_raw_rendered_or_used=False,U_precise_point_measurement=False,
        no_camera_axis_or_corner_reordering=True,new_RGB_model_forwards=0,new_line_fits=0,new_pose_fits=0,
        full_pose_improvement_on_target=False,
        limitations=['B6 remains fallback and I1 unchanged; all-B-corrected claim is false.',
            'B5 display projection is not a full observed KP.',
            'Normal Sigma and broad tangent radius are borrowed CAL assumptions, not new uncertainty calibration.',
            'Mask and native line ownership are conditional; neither is physically certified.',
            'Target pose error worsens; nearly equal all49 meanT is not statistical superiority.'])
    C.write(output/'CASE_REPORT.json',report)
    sections=''.join(f'<section id="step{i}"><h2>{html.escape(s["title"])}</h2><p>{html.escape(s["caption"])}</p><a href="{s["image"]}"><img src="{s["image"]}" alt="{html.escape(s["title"])}"></a></section>' for i,s in enumerate(stages,1))
    links=' · '.join(f'<a href="#step{i}">{i}</a>' for i in range(1,len(stages)+1))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B 경계 보정과 부분 관측: 실제 결과</title><style>body{margin:0;background:#edf1f7;color:#223248;font:17px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:30px;line-height:1.35}h2{font-size:23px}section,.intro{background:white;border:1px solid #d5dee9;border-radius:12px;padding:20px;margin:24px 0}img{display:block;max-width:100%;height:auto;margin:auto}a{color:#2158a2}nav{position:sticky;top:0;background:#fffefa;padding:10px;z-index:2;border-bottom:1px solid #d5dee9}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:680px;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #d7e0eb;text-align:right}td:nth-child(1),td:nth-child(2){text-align:left}.warning{border-left:5px solid #cf6522;padding-left:16px}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:12px}section,.intro{padding:13px}h1{font-size:25px}}</style><main><h1>B0·B3 보정과 B5 부분 관측을 실제 사진에서 확인</h1><p><code>eval_pallet07:1778652166837872128</code> · 실제 저장 마스크와 공식95 예측/사후 점수를 사용했습니다.</p><nav>과정 '''+links+''' · <a href="CASE_REPORT.json">좌표·출처 JSON</a></nav><div class="intro"><p class="warning"><strong>결과: B0·B3는 영상 경계점으로 보정했고 B5는 선 관측을 사용했습니다. B6은 보류입니다. 이 사진의 자세오차는 개선되지 않았습니다.</strong> 위치오차 10.11 → 13.61 → 14.60cm, 회전오차 4.73 → 4.29 → 4.54°입니다.</p><p>완전한 2D 관측점 ×, normal 표시점 ◇, PnP 모델 투영 □를 구분합니다. S7은 모델 투영만, U2는 넓은 영역만 사용합니다. GT/reference는 봉인된 예측의 사후 비교에만 표시합니다.</p><div class="scroll">'''+_table(context['scores'])+'''</div><p>CAL46·TEST49 모두 severe 제외, 실패·fallback 포함입니다. TEST는 이미 살펴본 재사용 집합이며 탐색 결과입니다. TEST 평균 위치의 매우 작은 차이와 동일 성공44/49는 방법의 우월성을 증명하지 않습니다.</p></div>'''+sections+'''<section><h2>해석의 한계</h2><p>선 표본은 실제 영상에서 얻었지만, 초기 모델의 native 경계 대응은 조건부 가정입니다. 이 사진의 native 번호 등록 검사 불일치를 사후 번호/카메라 회전으로 고치지 않았습니다. 관측의 2D 변화와 물리 자세오차를 별개로 평가합니다.</p><p>B5의 넓은 tangent 범위와 normal 분산은 기존 CAL 예산을 빌렸습니다. 새 B 전용 교정·확률보장·물리 소유권 확인은 없습니다. 이 페이지 생성으로 모델·경계선·PnP를 다시 계산하지 않았습니다.</p><a href="assets/source_actual.png">원본 실제 사진</a></section></main></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    E._verify_sources(context['bindings']);E._verify_sources(context['protected'])
    artifacts={str(p.relative_to(output)):C.sha(p) for p in output.rglob('*') if p.is_file()}
    C.write(output/'REPORT_SEAL.json',dict(schema='ACTUAL_FULL_MASK_PARTIAL_B_REPORT_SEAL_V4_6',
        input_bindings=context['bindings'],artifact_sha256=artifacts,frames=1,stage_images=len(stages),
        RGB_model_forwards=0,boundary_fits=0,PnP_fits=0,S_raw_coordinate_reads=0,
        GT_is_postseal_reference_only=True,protected_user_hashes_unchanged=True))
    print(json.dumps({'output':str(output),'stages':len(stages),'target_metrics':target['metrics']},ensure_ascii=False))


if __name__=='__main__':main()
