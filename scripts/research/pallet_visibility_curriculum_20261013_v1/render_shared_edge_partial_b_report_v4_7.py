"""Postseal actual-photo report of corner-reach and shared B-edge observations.

Rendering only: no image predictor, line fitting, PnP or parameter selection.
All observation status and pose outcomes are derived from saved official rows.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt, font_manager
from matplotlib.patches import Ellipse, Rectangle

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_shared_edge_partial_b_v4_7 as D
from . import render_full_mask_partial_b_report_v4_6 as R

TARGET=D.TARGET
OUTPUT=D.PARENT.BOUNDARY/'shared_edge_partial_b_report_v4_7'
NAMES={D.ARMS[0]:'이전 방법: 원 B fallback',
       D.ARMS[1]:'새 마스크 + incident 경계 검색',
       D.ARMS[2]:'새 보정 + 공유 경계 부분 관측'}


def _load(evaluation):
    evaluation=Path(evaluation).resolve();protocol=C.read(evaluation/'PROTOCOL.json')
    complete=C.read(evaluation/'PREDICTION_COMPLETE.json')
    if (complete['rows'],complete['CAL'],complete['TEST'],complete['severe'])!=(95,46,49,0):
        raise ValueError('Require all95 predictions sealed before report')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(protocol[key])
    paths=[Path(__file__),Path(R.__file__),R.FONT,evaluation/'PROTOCOL.json',evaluation/'PREDICTION_COMPLETE.json',evaluation/'SCORE_SEAL.json']
    scores={};score_seal=C.read(evaluation/'SCORE_SEAL.json')
    for partition,n in (('CAL',46),('TEST',49)):
        seal=C.read(evaluation/f'{partition}_PREDICTION_SEAL.json');journal=evaluation/f'{partition}_PREDICTIONS.jsonl.gz'
        score_path=evaluation/f'{partition}_POSE_SCORE.json'
        if (seal['rows']!=n or not seal['before_new_annotation_score']
                or C.sha(journal)!=seal['predictions_sha256']
                or C.sha(evaluation/'PROTOCOL.json')!=seal['protocol_sha256']
                or C.sha(score_path)!=score_seal[f'{partition}_pose_score_sha256']):
            raise ValueError('Official prediction/score seal mismatch')
        scores[partition]=C.read(score_path)
        if not scores[partition]['all_frames_include_fallbacks'] or scores[partition]['severe']!=0:
            raise ValueError('No hidden failure/severe filter')
        paths += [journal,score_path,evaluation/f'{partition}_PREDICTION_SEAL.json']
    rows=list(C.rows(evaluation/'TEST_PREDICTIONS.jsonl.gz'))
    if len(rows)!=49 or len({r['id'] for r in rows})!=49:raise ValueError('Population changed')
    row=next(r for r in rows if r['id']==TARGET);g=row['runtime']['initial_geometry']
    image_path=C.ROOT/row['image']
    if C.sha(image_path)!=row['image_sha256']:raise ValueError('Actual photo changed')
    image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    if image.shape[:2]!=tuple(g['image_hw']):raise ValueError('Image/K canvas changed')
    mask_dir=C.ROOT/protocol['mask_directory'];mask_seal=C.read(mask_dir/'SEAL.json')
    if C.sha(mask_dir/'RECORDS.json')!=mask_seal['records_sha256']:raise ValueError('Mask records changed')
    meta=next(r for r in C.read(mask_dir/'RECORDS.json') if r['id']==TARGET)
    mask_path=mask_dir/meta['mask_npz']
    if C.sha(mask_path)!=meta['mask_file_sha256'] or meta['image_sha256']!=row['image_sha256']:
        raise ValueError('Mask/image binding changed')
    with np.load(mask_path,allow_pickle=False) as z:
        masks={k:np.array(z[k]) for k in ('pallet_mask_raw','pallet_mask_clean','previous_pallet_mask')}
    paths += [image_path,mask_path,mask_dir/'RECORDS.json',mask_dir/'SEAL.json']
    # Reference access follows complete prediction and explicit score seals.
    reference_binding=scores['TEST']['target_binding'];reference_path=C.ROOT/reference_binding['path']
    if C.sha(reference_path)!=reference_binding['sha256']:raise ValueError('Reference binding changed')
    labels=C.read(reference_path)['rows']
    if isinstance(labels,list):labels={r['id']:r for r in labels}
    truth=labels[TARGET]['target']['truth']
    reference_wire=C.project(C.cuboid(*truth['body_xyz']),truth['body_R'],truth['t'],g['K'])
    paths.append(reference_path)
    old_process_path=D.PARENT.BOUNDARY/'stepwise_case_216683_v4/PROCESS_DATA.json'
    old_process=C.read(old_process_path);paths.append(old_process_path)
    projection={a:C.project(np.asarray(g['native_xyz']),v['actual_pose']['R_cf'],v['actual_pose']['centroid'],g['K'])
                for a,v in row['arms'].items()}
    raw={}
    for k in range(8):
        if g['initial_self_hidden'][k] or row['states'][k]=='S':continue
        raw[k]=list(row['old_correction']['raw_points'][k])
    fit={v['native_corner']:v for v in row['frozen_BI_fit']}
    partial={v['native_corner']:v for v in row['partial_boundary_observation']['boundary_observations']}
    status={}
    for k in fit:
        status[k]='부분 관측' if k in partial else '2D 보정' if fit[k]['source']!='VISIBLE_BACKBONE_FALLBACK' else 'fallback'
    old_score_path=D.PREVIOUS/'TEST_POSE_SCORE.json';old_seal=C.read(D.PREVIOUS/'SCORE_SEAL.json')
    if C.sha(old_score_path)!=old_seal['TEST_pose_score_sha256']:raise ValueError('Previous failed result changed')
    old_score=C.read(old_score_path);paths += [old_score_path,D.PREVIOUS/'SCORE_SEAL.json']
    return dict(row=row,geometry=g,raw=raw,image=image,image_path=image_path,mask_meta=meta,masks=masks,
        scores=scores,projection=projection,reference_wire=reference_wire,fit=fit,partial=partial,status=status,
        domains=old_process['domain_observations']['O_T']+row['u_region_observation']['u_regions'],
        previous_target_score=old_score['target'],
        bindings={R._path_key(p):C.sha(p) for p in paths},protected=protocol['protected_user_sha256'])


def _actual_rays(context,k):
    correction=context['row']['correction']
    records={r['native_corner']:r for r in correction['records']}
    recovery={r['native_corner']:r for r in correction['boundary_recovery_records']}
    values=[('complete_reference',v) for v in records[k].get('rays',[])]+R._rays(recovery[k])
    seen=set();result=[]
    for origin,ray in values:
        key=json.dumps(ray,sort_keys=True)
        if key not in seen:result.append((origin,ray));seen.add(key)
    return result


def _build(c,output):
    assets=output/'assets';assets.mkdir();stages=[];row=c['row'];h,w=c['image'].shape[:2]
    def add(filename,title,caption,draw):
        draw(assets/filename);stages.append(dict(image='assets/'+filename,title=title,caption=caption))
    def masks(path):
        fig,axes=plt.subplots(1,3,figsize=(17,5.5))
        for ax,key,title in zip(axes,('previous_pallet_mask','pallet_mask_raw','pallet_mask_clean'),('원 마스크','전체 초기 기하 box로 얻은 원본','작은 성분 제거: 실제 사용 마스크')):
            R._photo(ax,c,title);rgba=np.zeros((h,w,4));rgba[:,:,:3]=[.32,.89,.45];rgba[:,:,3]=.4*(c['masks'][key]>.5);ax.imshow(rgba);ax.axis('off')
        fig.tight_layout();R._saved(fig,path)
    add('01_actual_masks.png','1. 같은 사진·고정 상태, 보정용 마스크 범위 확대',
        '새 마스크는 초기 모델 전체의 투영과 비S 영상 안 예측 범위를 box guide로 사용했습니다. MobileSAM의 기존 quality argmax로 선택하고 작은 연결 성분만 제거했습니다. 원 마스크와 모든 후보는 보존했습니다. 이번 incident 경계 검색 비교에서 새 SAM 추론은 없으며 Tiny 상태도 그대로입니다.',masks)
    def measured(path):
        fig,ax=plt.subplots(figsize=(11,7.8));R._photo(ax,c,'2. 실제 2D 보정 / 선 부분 관측 / fallback')
        for k,m in c['fit'].items():
            raw=c['raw'][k];R._point(ax,raw,f'raw{k}',R.WHITE,'o',(-28,15),40)
            if k in c['partial']:
                p=c['partial'][k]['association_provenance']['normal_projected_display_xy']
                R._point(ax,p,f'B{k} normal 표시',R.GREEN,'D',(-120,-14),45)
                R._point(ax,c['projection'][D.ARMS[2]][k],f'P{k} 모델',R.ORANGE,'s',(8,15),40)
            elif c['status'][k]=='2D 보정':
                xy=m['xy'];ax.annotate('',xy=xy,xytext=raw,arrowprops={'arrowstyle':'->','color':R.BLUE,'lw':1.7});R._point(ax,xy,f'{row["states"][k]}{k} 보정',R.BLUE,'x')
            else:R._point(ax,m['xy'],f'{row["states"][k]}{k} fallback',R.YELLOW,'x')
        ax.text(.02,.025,'○ 원 예측 / × 실제 2D 관측 / ◇ 설명용 normal 투영 / □ PnP 모델',transform=ax.transAxes,color=R.WHITE,fontsize=11,bbox={'facecolor':'#172030','alpha':.85})
        fig.tight_layout();R._saved(fig,path)
    inventory=' · '.join(f'{row["states"][k]}{k}: {s}' for k,s in sorted(c['status'].items()))
    add('02_point_and_partial_observations.png','2. 각 점의 실제 처리 결과',inventory+'입니다. 부분 관측의 원 B점은 최종 2D likelihood에서 제거하며 full KP를 생성하지 않습니다. 기존 성공 B와 I 측정은 그대로 유지합니다.',measured)
    def details(path):
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        limits={0:((-10,110),(405,305)),3:((-10,130),(445,335)),5:((475,610),(350,265)),6:((470,615),(425,300))}
        for ax,k in zip(axes.flat,(0,3,5,6)):
            R._photo(ax,c,f'B{k}: {c["status"][k]}',limits[k])
            for _,ray in _actual_rays(c,k):R._draw_ray(ax,ray)
            R._point(ax,c['raw'][k],f'raw{k}',R.WHITE,'o',(7,15),40)
            if k in c['partial']:
                for line in c['partial'][k]['lines']:
                    prov=line['provenance'];actual=prov.get('actual_observed_inlier_tangent_interval_px')
                    if actual:R._line(ax,line['normal_unit_xy'],line['offset_px'],actual,R.GREEN)
                R._point(ax,c['partial'][k]['association_provenance']['normal_projected_display_xy'],'normal 표시',R.GREEN,'D',(-92,-15),40)
            else:R._point(ax,c['fit'][k]['xy'],'보정' if c['status'][k]=='2D 보정' else 'fallback',R.BLUE if c['status'][k]=='2D 보정' else R.YELLOW,'x')
            R._point(ax,c['projection'][D.ARMS[2]][k],f'P{k}',R.ORANGE,'s',(8,15),35)
        fig.tight_layout();R._saved(fig,path)
    add('03_supported_RGB_edges.png','3. 경계의 실제 RGB 표본과 모델 투영을 비교',
        '작은 점은 저장된 실제 검색 표본, 녹색 선분은 품질 조건을 통과한 실제 inlier 구간입니다. 검색 반경의 floor는 기존 along_end36 + context7 =43px이며 RGB·방향12°·RMS·표본수·finite support11px 수치는 유지합니다. 높은 sine보다 먼저11px 지지 조건을 통과한 교점을 선택합니다.',details)
    def partials(path):
        count=max(1,len(c['partial']));fig,axes=plt.subplots(1,count,figsize=(9*count,6.5),squeeze=False)
        if not c['partial']:
            R._photo(axes[0,0],c,'부분 관측 없음');axes[0,0].text(.02,.03,'모든 상태는 저장 결과 그대로입니다.',transform=axes[0,0].transAxes)
        for ax,(k,obs) in zip(axes.flat,c['partial'].items()):
            bounds=[np.array([0.,0.]),np.array([float(w),float(h)])]
            for line in obs['lines']:
                n=np.asarray(line['normal_unit_xy']);t=np.array([-n[1],n[0]])
                lo,hi=line['support_tangent_interval_px'];margin=line['support_extrapolation_px']
                bounds += [-line['offset_px']*n+(lo-margin)*t,-line['offset_px']*n+(hi+margin)*t]
            bounds=np.asarray(bounds);lo=bounds.min(axis=0)-20;hi=bounds.max(axis=0)+20
            R._photo(ax,c,f'B{k}: normal 관측 + 넓은 tangent 범위',((lo[0],hi[0]),(hi[1],lo[1])))
            for line in obs['lines']:
                interval=line['support_tangent_interval_px'];margin=line['support_extrapolation_px'];p=line['provenance']
                R._line(ax,line['normal_unit_xy'],line['offset_px'],[interval[0]-margin,interval[1]+margin],R.GREEN,True,1.5)
                actual=p.get('actual_observed_inlier_tangent_interval_px')
                if actual:R._line(ax,line['normal_unit_xy'],line['offset_px'],actual,R.GREEN,False,3)
            R._point(ax,c['raw'][k],f'raw{k}',R.WHITE,'o',(-25,14),40)
            R._point(ax,obs['association_provenance']['normal_projected_display_xy'],'normal 표시',R.GREEN,'D',(-94,-14),40)
            R._point(ax,c['projection'][D.ARMS[2]][k],f'P{k} 모델',R.ORANGE,'s',(9,14),40)
            ax.text(.02,.03,'실선 = 관측 inlier / 점선 = CAL raw-locality 가정 포함 범위\n공유 선은 새로운 독립 표본이 아니며 분산 증가는 가정입니다.',transform=ax.transAxes,fontsize=11,bbox={'facecolor':'white','alpha':.9})
        fig.tight_layout();R._saved(fig,path)
    add('04_shared_edge_partial_range.png','4. 공유 경계는 exact KP가 아니라 normal·범위 관측',
        '실제로 검증한 B-to-B edge는 같은 두 endpoint의 선 관측으로 사용할 수 있습니다. 아직 full 보정이 없는 B만 받으며 S/O/U/T/I는 제외합니다. source 선을 새 독립 영상 증거로 세지 않고 고정된 보수적 분산 증가를 사용합니다. tangent 범위는 실제 inlier 구간과 기존 CAL raw-localization 반경을 합친 가정이며 확률 보장은 없습니다. ◇의 x/y 전체를 정확한 점으로 PnP에 넣지 않습니다.',partials)
    def ranges(path):
        fig,ax=plt.subplots(figsize=(10.4,10));R._photo(ax,c,'5. O/U 영역과 모든 B 부분 범위의 joint PnP',((-15,675),(835,-20)))
        for d in c['domains']:
            k=d['native_corner'];color='#ffb52e' if d['state']=='O' else '#9ba9b9';axes=d['half_axes_px'];center=d['center_xy']
            ax.add_patch(Ellipse(center,2*axes[0],2*axes[1],angle=d.get('angle_deg',0),fill=False,ec=color,lw=2,ls='--'))
            R._point(ax,center,f'{d["state"]}{k} 중심',color,'+',(-115,17),35);R._point(ax,c['projection'][D.ARMS[2]][k],f'P{k} 모델',R.ORANGE,'s',(10,-14),40)
        R._wire(ax,c['projection'][D.ARMS[2]],R.ORANGE);R._point(ax,c['projection'][D.ARMS[2]][7],'S7 모델만',R.VIOLET,'s',(8,13),35)
        ax.text(20,805,'회색 = 실제 영상 밖 / U는 canvas로 자르지 않음',fontsize=12,color='#263b50')
        fig.tight_layout();R._saved(fig,path)
    add('05_joint_O_U_ranges.png','5. O/U는 기존 영역 그대로, S는 PnP 투영만',
        '기존 O4 영역과 영상 밖 U2의 넓은 영역을 모든 B 부분 관측과 함께 적용합니다. U를 점 비용·새 image line·PnP seed로 사용하지 않으며 S7의 raw/RGB 검색은 없습니다. 주황 □는 추정 모델의 투영입니다.',ranges)
    def poses(path,reference=False):
        fig,axes=plt.subplots(1,3,figsize=(18,6.5));metrics=c['scores']['TEST']['target']['metrics']
        for ax,a in zip(axes,D.ARMS):
            m=metrics[a];R._photo(ax,c,NAMES[a]+f'\n{m["translation_cm"]:.2f}cm / {m["rotation_deg"]:.2f}°');R._wire(ax,c['projection'][a],R.ORANGE)
            if reference:R._wire(ax,c['reference_wire'],R.GREEN,True)
            else:
                for k in c['fit']:
                    if a==D.ARMS[2] and k in c['partial']:continue
                    xy=c['raw'][k] if a==D.ARMS[0] else c['fit'][k]['xy'];R._point(ax,xy,str(k),R.BLUE if a!=D.ARMS[0] and c['status'][k]=='2D 보정' else R.YELLOW,'x',size=40)
        fig.tight_layout();R._saved(fig,path)
    target=c['scores']['TEST']['target']['metrics'];sequence=' → '.join(f'{target[a]["translation_cm"]:.2f}cm / {target[a]["rotation_deg"]:.2f}°' for a in D.ARMS)
    add('06_three_saved_PnP_results.png','6. 실제 저장된 세 가지 자세 결과',
        sequence+'입니다. 같은 초기 자세·native XYZ/Q·CAL prior·fitΣ/weights·상태·O/U 영역을 사용했습니다. 보정된 2D 경계와 물리 자세오차의 변화는 따로 평가합니다. original V4.5 기준을 생략하지 않았습니다.',lambda p:poses(p))
    add('07_postseal_reference_wire.png','7. reference는 봉인 후 물리 box 비교에만 표시',
        '녹색 점선은 body_R/body_xyz와 reference 중심의 사후 투영, 주황은 각 방법 자체의 모델 투영입니다. 이 사례의 native 번호 등록 불일치를 수동 카메라90° 회전·번호 재정렬로 고치지 않았습니다. 초록 코너를 동일 native KP 관측으로 사용하지 않았고 all49 점수에서 이 사례도 유지했습니다.',lambda p:poses(p,True))
    return stages


def _table(c):
    body=[]
    for partition in ('CAL','TEST'):
        for a in D.ARMS:
            s=c['scores'][partition]['summary_all'][a]
            body.append(f'<tr><td>{partition}</td><td>{html.escape(NAMES[a])}</td><td>{s["translation_cm"]["mean"]:.6f}</td><td>{s["rotation_deg"]["mean"]:.6f}</td><td>{s["translation_cm"]["max"]:.4f}</td><td>{s["rotation_deg"]["max"]:.4f}</td><td>{s["successes_5cm_5deg"]}/{s["frames"]}</td><td>{s["fallback_count"]}</td><td>{s["ambiguous_count"]}</td></tr>')
    return '<table><thead><tr><th>전체 집합</th><th>방법</th><th>평균 위치 cm</th><th>평균 회전 °</th><th>최대 위치 cm</th><th>최대 회전 °</th><th>5cm·5° 성공</th><th>fallback 포함</th><th>모호성</th></tr></thead><tbody>'+''.join(body)+'</tbody></table>'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation',type=Path,default=D.OUTPUT);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();output=args.output.resolve()
    if output.exists():raise FileExistsError('Preserve existing report: '+str(output))
    font_manager.fontManager.addfont(str(R.FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(R.FONT)).get_name();plt.rcParams['axes.unicode_minus']=False
    c=_load(args.evaluation);output.mkdir(parents=True);stages=_build(c,output)
    shutil.copyfile(c['image_path'],output/'assets/source_actual.png')
    if C.sha(output/'assets/source_actual.png')!=C.sha(c['image_path']):raise ValueError('Actual image copy changed')
    target=c['scores']['TEST']['target'];base=target['metrics'][D.ARMS[0]];final=target['metrics'][D.ARMS[2]]
    improvement=final['translation_cm']<base['translation_cm'];result='위치오차 감소' if improvement else '위치오차 개선 실패'
    sb=c['scores']['TEST']['summary_all'][D.ARMS[0]];sf=c['scores']['TEST']['summary_all'][D.ARMS[2]]
    cb=c['scores']['CAL']['summary_all'][D.ARMS[0]];cf=c['scores']['CAL']['summary_all'][D.ARMS[2]]
    global_note=(f'이 사례는 최종 {final["translation_cm"]:.2f}cm로 5cm 위치 기준을 여전히 넘습니다. '
        f'TEST 성공은 {sb["successes_5cm_5deg"]}/49 → {sf["successes_5cm_5deg"]}/49, '
        f'최대 위치오차는 {sb["translation_cm"]["max"]:.2f} → {sf["translation_cm"]["max"]:.2f}cm입니다. '
        f'CAL 최대 위치오차는 {cb["translation_cm"]["max"]:.2f} → {cf["translation_cm"]["max"]:.2f}cm로 악화했고, '
        f'TEST 모호성 표시는 {sb["ambiguous_count"]} → {sf["ambiguous_count"]}건입니다. 전체 문제 해결·최상 성능을 주장하지 않습니다.')
    inventory=' · '.join(f'{c["row"]["states"][k]}{k}: {s}' for k,s in sorted(c['status'].items()))
    report=dict(schema='SHARED_B_EDGE_ACTUAL_PHOTO_REPORT_V4_7',id=TARGET,input_bindings=c['bindings'],stages=stages,
        raw_nonS=c['raw'],states=c['row']['states'],statuses={str(k):v for k,v in c['status'].items()},
        fixed_BI_fit=c['row']['frozen_BI_fit'],partial_B_observation=c['row']['partial_boundary_observation'],
        saved_arm_poses={a:v['actual_pose'] for a,v in c['row']['arms'].items()},
        model_projections={a:q.tolist() for a,q in c['projection'].items()},reference_body_wire_xy=c['reference_wire'].tolist(),
        target_score=target,previous_v4_6_target_score=c['previous_target_score'],
        summaries={p:c['scores'][p]['summary_all'] for p in ('CAL','TEST')},
        mask_provenance=c['mask_meta']['provider_provenance'],target_translation_improved_vs_original=improvement,
        target_success_5cm_5deg=final['success_5cm_5deg'],global_limitations_text=global_note,
        complete_all95_include_fallbacks=True,TEST_reused_exploratory=True,
        new_model_forwards=0,new_line_fits=0,new_PnP_fits=0,S_raw_coordinate_reads=0,
        partial_display_is_full_KP=False,U_precise_cost=False,reference_postseal_only=True,
        manual_camera_rotation_or_native_reordering=False,new_uncertainty_calibration=False)
    C.write(output/'CASE_REPORT.json',report)
    sections=''.join(f'<section id="step{i}"><h2>{html.escape(s["title"])}</h2><p>{html.escape(s["caption"])}</p><a href="{s["image"]}"><img src="{s["image"]}" alt="{html.escape(s["title"])}"></a></section>' for i,s in enumerate(stages,1))
    links=' · '.join(f'<a href="#step{i}">{i}</a>' for i in range(1,len(stages)+1))
    old=c['previous_target_score']['metrics'];oldseq=' → '.join(f'{m["translation_cm"]:.2f}cm' for m in old.values())
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B 경계 보정: 실제 측정과 공유 선</title><style>body{margin:0;background:#edf1f7;color:#223248;font:17px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:30px;line-height:1.35}h2{font-size:23px}section,.intro{background:white;border:1px solid #d5dee9;border-radius:12px;padding:20px;margin:24px 0}img{display:block;max-width:100%;height:auto;margin:auto}a{color:#2158a2}nav{position:sticky;top:0;background:#fffefa;padding:10px;z-index:2}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:680px;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #d7e0eb;text-align:right}td:nth-child(1),td:nth-child(2){text-align:left}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:12px}section,.intro{padding:13px}h1{font-size:25px}}</style><main><h1>B 경계의 실제 영상 보정과 공유 선 부분 관측</h1><p><code>eval_pallet07:1778652166837872128</code></p><nav>과정 '''+links+''' · <a href="CASE_REPORT.json">좌표·출처 JSON</a></nav><div class="intro"><p><strong>'''+html.escape(inventory)+'''</strong></p><p><strong>'''+result+f': 원 {base["translation_cm"]:.2f}cm / {base["rotation_deg"]:.2f}° → 최종 {final["translation_cm"]:.2f}cm / {final["rotation_deg"]:.2f}°.'+'''</strong> full 점 보정과 부분 선 관측을 구분하고, 추정 PnP 투영은 별도 기호로 표시합니다.</p><p>'''+html.escape(global_note)+'''</p><div class="scroll">'''+_table(c)+'''</div><p>CAL46·TEST49는 severe 제외, 모든 실패·fallback 포함입니다. TEST는 이미 살펴본 재사용 집합이며 탐색 결과입니다. 새 마스크를 넣었던 이전 V4.6 실패 결과('''+oldseq+''')도 보존했습니다. 성공 수나 평균의 작은 변화로 통계적 우월성을 주장하지 않습니다.</p></div>'''+sections+'''<section><h2>남는 가정과 한계</h2><p>공유 RGB 선은 두 endpoint를 잇는 초기 native 기하 대응을 가정합니다. 같은 선을 새 독립 증거로 부풀리지 않으며 분산 증가는 engineering 가정입니다. CAL의 넓은 tangent 예산은 새 B 전용 확률 교정이 아닙니다.</p><p>S raw/RGB는 사용하지 않았고 U는 영역만 사용합니다. reference body wire는 사후 비교입니다. 이 사례의 native 번호 등록 불일치를 수동 번호나 카메라 축 변환으로 고치지 않았습니다. 2D 경계 보정과 물리 3D 자세오차를 동일한 성공으로 취급하지 않습니다.</p><a href="assets/source_actual.png">원본 실제 사진</a></section></main></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    E._verify_sources(c['bindings']);E._verify_sources(c['protected'])
    artifacts={str(p.relative_to(output)):C.sha(p) for p in output.rglob('*') if p.is_file()}
    C.write(output/'REPORT_SEAL.json',dict(input_bindings=c['bindings'],artifact_sha256=artifacts,frames=1,
        stage_images=len(stages),S_raw_coordinate_reads=0,new_RGB_model_forwards=0,new_line_fits=0,new_PnP_fits=0,
        postseal_reference_only=True,protected_user_hashes_unchanged=True))
    print(json.dumps({'output':str(output),'stages':len(stages),'target_score':target},ensure_ascii=False))


if __name__=='__main__':main()
