"""Postseal actual-image stages; fixed native-only pair, never case-best arms.

Reads only sealed production observations/poses and completed score rows.
Reference poses/corners are never opened or drawn. All actual NMS alternatives
remain in source data; display pixels are deduplicated solely for legibility.
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
from matplotlib.patches import Polygon

from . import common as C
from . import refine_grazing_image_chain_mixtures_v4_18c as EXP
from . import observe_grazing_native_edge_chains_v4_18b as OBS
from . import render_frozen_wd_process_v4_13e_v2 as DRAW


OUTPUT=EXP.OUTPUT.parent/'refined_grazing_mixture_process_v4_18c'
PRIMARY=EXP.ARMS[1]
CACHED=EXP.PARENT.ARMS[1]
BASELINE=EXP.ARMS[0]
NAMES={BASELINE:'기존 e',CACHED:'새 NMS cached·native',EXP.PARENT.ARMS[2]:'새 NMS cached·native+mask',
    PRIMARY:'새 NMS continuous·native',EXP.ARMS[2]:'새 NMS continuous·native+mask'}
TITLES=('① 초기 모델과 고정 B/I 기록','② 실제 마스크·모든 NMS 선 대안',
    '③ O/U 및 기존 접선 범위','④ 기존 e 저장 모델','⑤ 새 cached native 선택 모델','⑥ continuous native 풀 선택 모델')
CAPTIONS=(
    '회색 wire는 초기 모델투영. 분홍 ×는 유지된 완전 B/I 관측, 빈 회색 +는 부분 B/I의 보관 q로서 정밀 RGB 코너가 아님. U/S의 입력 점은 표시하지 않음.',
    '녹색 틴트는 저장 실제 마스크. 회색은 모든 실제 NMS 대안, 청록은 현재 문제의 부분관측에 대응하는 물리 edge 대안. fallback은 이 제약들을 푼 해가 아님. 같은 픽셀은 표시만 중복 제거. 선은 실제 인라이어 구간만 표시.',
    'O는 경험 영역, U는 CAL 최대오차에서 빌린 넓은 범위. 회색 띠/점선은 기존 부분 B/I의 접선 구간이며 새 정밀 KP가 아님. 확대창 밖 범위는 잘려 보이고 원래 전체 수치는 JSON에 보존.',
    '파란 wire는 봉인된 기존 e 자세의 모델투영. 영상 측정점 또는 참조/GT wire가 아님.',
    '주황 wire는 같은 새 grazing/NMS 관측과 새 CAL 잡음으로 저장 후보를 재평가한 native-only 선택. 새로운 연속 fitting을 한 단계는 아님.',
    '초록 wire는 동일 관측의 연속 SE3 최적화 끝점과 모든 기존 후보를 포함한 풀의 native-only 선택. 선택 origin이 CACHED면 기존 후보가 유지된 것. 유일성/전역최적성 증명 아님.')


def _url(path):
    return quote(os.path.relpath(Path(path).resolve(),OUTPUT),safe='/')


def _load_completed(folder):
    protocol=C.read(folder/'PROTOCOL.json');pred=C.read(folder/'PREDICTION_SEAL.json')
    score=C.read(folder/'SCORE_SEAL.json')
    if (pred['rows']!=238 or not pred['all3_arms_complete']
            or C.sha(folder/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json')!=pred['protocol_sha256']
            or C.sha(folder/'PREDICTION_SEAL.json')!=score['prediction_seal']['sha256']):
        raise ValueError('Complete238 predictions and postseal scores required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        EXP.E._verify_sources(protocol[key])
    if folder.resolve()==EXP.OUTPUT.resolve():
        compatibility=C.read(folder/'SCORER_COMPATIBILITY_RECEIPT.json')
        compatibility_seal=C.read(folder/'SCORER_COMPATIBILITY_SEAL.json')
        if (C.sha(folder/'SCORER_COMPATIBILITY_RECEIPT.json')!=compatibility_seal['receipt_sha256']
                or compatibility['predictions_unchanged_sha256']!=pred['predictions_sha256']
                or compatibility['field_used_in_numeric_metrics'] is not False):
            raise ValueError('Explicit source-bound metadata-only scorer compatibility required')
        for key in ('adapter_source','scorer_source','prediction_seal','score_seal'):
            EXP.E._verify_binding(compatibility[key])
    predictions=list(C.rows(folder/'PREDICTIONS.jsonl.gz'))
    if len(predictions)!=238 or [r['id']for r in predictions]!=protocol['ids']:
        raise ValueError('All original238 rows/order required')
    scores={};reports={}
    for partition,count in (('CAL',46),('TEST',49),('REMAINDER',143)):
        path=folder/f'{partition}_SCORE.json'
        if C.sha(path)!=score['score_sha256'][partition]:
            raise ValueError('Postseal score changed')
        report=C.read(path)
        if len(report['rows'])!=count:
            raise ValueError('No failed/fallback/native-incompatible frame may be dropped')
        reports[partition]=report;scores.update({r['id']:r for r in report['rows']})
    return protocol,{r['id']:r for r in predictions},scores,reports


def _model(arm,K):
    pose=arm['actual_pose']
    if not pose.get('available',True) or pose.get('R_cf') is None or pose.get('centroid') is None:
        return None
    xyz=C.cuboid(*pose['cf_extents']);R=np.asarray(pose['R_cf']);t=np.asarray(pose['centroid'])
    if 'physical_registration_Q' in pose:
        if not np.allclose(R@np.asarray(pose['physical_registration_Q']),pose['R_physical'],atol=1e-10,rtol=0):
            raise ValueError('Displayed physical/native pose registration differs')
    if arm.get('solution',{}).get('family') is not None:
        if not np.allclose(xyz,arm['solution']['family']['native_xyz'],atol=1e-12,rtol=0):
            raise ValueError('Use the selected model extents/native basis, not another WD model')
    camera=xyz@R.T+t
    if np.any(camera[:,2]<=0):raise ValueError('Stored displayed model depth invalid')
    return C.project(xyz,R,t,K)


def _frozen_points(ax,task):
    partial={b['native_corner']for b in task['boundary_merged']}
    for measurement in task['spec']['visible']:
        k=measurement['native_corner']
        if task['initial_self_hidden'][k] or task['states'][k]not in ('B','I'):
            raise ValueError('S/U/O/T are excluded before displayed point access')
        if k in partial:
            DRAW._points(ax,{k:measurement['xy']},'#7f8590','+',prefix=task['states'][k]+' 보관',offset=(5,-11))
        else:
            DRAW._points(ax,{k:measurement['xy']},'#e53d9b','x',prefix=task['states'][k],offset=(5,-11))


def _nms(ax,observation,active_bundle):
    active={tuple(sorted((g['native_corner'],g['native_neighbor'])))for g in active_bundle['groups']}
    all_points=[];active_points=[];line_count=0
    for ray in observation['edge_records']:
        pair=tuple(sorted((ray['native_corner'],ray['native_neighbor'])))
        for chain in ray.get('chains',()):
            xy=np.asarray(chain['observed_points']);actual=xy[np.asarray(chain['inlier_mask'],bool)]
            all_points.extend(actual)
            if pair in active:
                active_points.extend(actual)
                n=np.asarray(chain['normal']);t=np.array([-n[1],n[0]]);lo,hi=chain['actual_inlier_tangent_interval_px']
                segment=-float(chain['offset'])*n+np.asarray([lo,hi])[:,None]*t
                ax.plot(segment[:,0],segment[:,1],c='#00b8ba',lw=.8,alpha=.55);line_count+=1
    if all_points:
        pixels=np.unique(np.round(np.asarray(all_points),6),axis=0)
        ax.scatter(pixels[:,0],pixels[:,1],s=4,c='#81858b',alpha=.45)
    if active_points:
        pixels=np.unique(np.round(np.asarray(active_points),6),axis=0)
        ax.scatter(pixels[:,0],pixels[:,1],s=5,c='#00b8ba',alpha=.7)
    else:
        ax.text(.02,.96,'이번 자세비용에 새 NMS 물리 edge 없음',transform=ax.transAxes,va='top',fontsize=9,
            bbox=dict(facecolor='white',alpha=.8))
    return dict(all_archived_chain_count=sum(len(r.get('chains',()))for r in observation['edge_records']),
        active_physical_edges=[list(p)for p in sorted(active)],displayed_active_directed_chain_count=line_count,
        unique_pixel_display_is_not_measurement_or_mixture_component_deduplication=True)


def _tangent_ranges(ax,records):
    for record in records:
        for line in record['lines']:
            n=np.asarray(line['normal_unit_xy']);t=np.array([-n[1],n[0]])
            offset=float(line['offset_px']);lo,hi=line['support_tangent_interval_px']
            width=float(np.sqrt(line['normal_variance_px2']))
            quad=np.array([-offset*n+lo*t-width*n,-offset*n+hi*t-width*n,
                -offset*n+hi*t+width*n,-offset*n+lo*t+width*n])
            ax.add_patch(Polygon(quad,fc='#777c85',ec='#777c85',alpha=.1,lw=.8))
            segment=-offset*n+np.asarray([lo,hi])[:,None]*t
            ax.plot(segment[:,0],segment[:,1],c='#737881',ls='--',lw=.9)


def _metric_text(metric):
    rotation=metric['rotation_deg']
    return f"{metric['translation_cm']:.3f}cm / "+('R미확정'if rotation is None else f'{rotation:.3f}°')


def render():
    if OUTPUT.exists():raise FileExistsError('Preserve every original/partial report')
    protocol,prediction,scores,reports=_load_completed(EXP.OUTPUT)
    cached_protocol,cached,cached_scores,cached_reports=_load_completed(EXP.PARENT.OUTPUT)
    remainder=reports['REMAINDER']['rows'];test=reports['TEST']['rows']
    worst=max(remainder,key=lambda r:r['metrics'][PRIMARY]['translation_cm'])['id']
    regressions=[r for r in test if r['metrics'][PRIMARY]['translation_cm']>r['metrics'][BASELINE]['translation_cm']+1e-12]
    ids=['wood_night_01:031344',worst]
    if regressions:
        ids.append(max(regressions,key=lambda r:r['metrics'][PRIMARY]['translation_cm']-r['metrics'][BASELINE]['translation_cm'])['id'])
    ids=list(dict.fromkeys(ids))
    tasks={t['id']:t for t in C.rows(EXP.OUTPUT/'INPUTS.jsonl.gz')if t['id']in ids}
    observations={r['id']:r for r in C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')if r['id']in ids}
    if set(tasks)!=set(ids)or set(observations)!=set(ids):raise ValueError('All selected diagnostic inputs required')
    DRAW._font();OUTPUT.mkdir(parents=True);assets=OUTPUT/'assets';assets.mkdir()
    paths=(Path(__file__),Path(DRAW.__file__),EXP.OUTPUT/'PROTOCOL.json',EXP.OUTPUT/'INPUT_SEAL.json',
        EXP.OUTPUT/'PREDICTION_SEAL.json',EXP.OUTPUT/'PREDICTIONS.jsonl.gz',EXP.OUTPUT/'SCORE_SEAL.json',
        EXP.PARENT.OUTPUT/'PREDICTION_SEAL.json',EXP.PARENT.OUTPUT/'PREDICTIONS.jsonl.gz',EXP.PARENT.OUTPUT/'SCORE_SEAL.json',
        OBS.OUTPUT/'OBSERVATION_SEAL.json',OBS.OUTPUT/'OBSERVATIONS.jsonl.gz',
        EXP.OUTPUT/'SCORER_COMPATIBILITY_RECEIPT.json',EXP.OUTPUT/'SCORER_COMPATIBILITY_SEAL.json',
        Path(__file__).with_name('score_refined_grazing_mixtures_v4_18c.py'))
    bindings={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    for folder in (EXP.OUTPUT,EXP.PARENT.OUTPUT):
        for partition in ('CAL','TEST','REMAINDER'):
            path=folder/f'{partition}_SCORE.json';bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    cases=[];overview=[]
    for number,fid in enumerate(ids):
        task=tasks[fid];row=prediction[fid];old=cached[fid];observed=observations[fid]
        if EXP.E._digest(row['arms'][BASELINE])!=EXP.E._digest(old['arms'][BASELINE]):
            raise ValueError('Exact unchanged e baseline required')
        for key in ('states','visible_measurements','domains','u_regions'):
            expected=task['spec']['visible']if key=='visible_measurements'else task['spec']['domains']if key=='domains'else task[key]
            if observed[key]!=expected:raise ValueError('Same sealed actual observation profile required')
        image_path=C.ROOT/task['image'];mask_path=C.ROOT/observed['mask_binding']['path']
        if C.sha(image_path)!=task['image_sha256']or C.sha(mask_path)!=observed['mask_binding']['sha256']:
            raise ValueError('Bound actual RGB and mask required')
        bindings[task['image']]=task['image_sha256'];bindings[observed['mask_binding']['path']]=observed['mask_binding']['sha256']
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        with np.load(mask_path,allow_pickle=False)as arrays:mask=arrays['pallet_mask_clean'].copy()
        geometry=observed['initial_geometry'];K=geometry['K'];states=task['states']
        qold=_model(row['arms'][BASELINE],K);qcached=_model(old['arms'][CACHED],K);qnew=_model(row['arms'][PRIMARY],K)
        points=[np.asarray(geometry['initial_projection_xy'])]+[q for q in (qold,qcached,qnew)if q is not None]
        for ray in task['new_chain_observation']['edge_records']:
            for chain in ray.get('chains',()):
                observed_points=np.asarray(chain['observed_points'])
                points.append(observed_points[np.asarray(chain['inlier_mask'],bool)])
        xy=np.vstack(points);lo=np.maximum(xy.min(0)-18,0);hi=np.minimum(xy.max(0)+18,[image.shape[1],image.shape[0]])
        bounds=(lo[0],hi[0],hi[1],lo[1])
        arm=row['arms'][PRIMARY];bundle=arm['new_image_line_group_authentication'][arm['WD_family']]
        metrics={BASELINE:scores[fid]['metrics'][BASELINE],CACHED:cached_scores[fid]['metrics'][CACHED],
            EXP.PARENT.ARMS[2]:cached_scores[fid]['metrics'][EXP.PARENT.ARMS[2]],
            PRIMARY:scores[fid]['metrics'][PRIMARY],EXP.ARMS[2]:scores[fid]['metrics'][EXP.ARMS[2]]}
        case=dict(id=fid,partition=row['partition'],image=task['image'],image_sha256=task['image_sha256'],
            original_states=row['original_states'],current_states=states,new_grazing_U_native_ids=row['grazing_adaptation']['geometry_gate']['new_U_native_ids'],
            source_mask=observed['mask_binding'],initial_geometry=geometry,visible_frozen_BI=task['spec']['visible'],
            partial_boundary=task['boundary_merged'],O_T_domains=task['spec']['domains'],U_regions=task['u_regions'],
            all_actual_NMS_chain_observation=task['new_chain_observation'],active_groups=bundle,
            e_pose=row['arms'][BASELINE]['actual_pose'],cached_native_pose=old['arms'][CACHED]['actual_pose'],
            continuous_native_pose=arm['actual_pose'],continuous_selection_origin=arm['keypoint_readout']['selected_endpoint_origin'],
            metrics=metrics,stages=[],crop_bounds=bounds,no_reference_pose_or_corner_read_or_drawn=True)
        for index,title in enumerate(TITLES):
            fig,ax=DRAW._base(image,title)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
            if index==0:
                DRAW._wire(ax,np.asarray(geometry['initial_projection_xy']),'#7e8790',ls='--');_frozen_points(ax,task)
            elif index==1:
                ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.15]))
                case['NMS_display_audit']=_nms(ax,task['new_chain_observation'],bundle)
            elif index==2:
                DRAW._domains(ax,task['spec']['domains'],task['u_regions']);_tangent_ranges(ax,task['boundary_merged'])
            else:
                q=(qold,qcached,qnew)[index-3];color=(DRAW.BLUE,DRAW.ORANGE,DRAW.GREEN)[index-3]
                selected=(row['arms'][BASELINE],old['arms'][CACHED],arm)[index-3]
                if q is None:ax.text(.03,.05,'저장 자세 없음',transform=ax.transAxes)
                else:
                    DRAW._wire(ax,q,color,width=1.8,ls='-'if selected['accepted']else '--')
                    DRAW._points(ax,{k:q[k]for k,s in enumerate(states)if s=='S'},DRAW.PURPLE,'^',prefix='S모델')
                key=(BASELINE,CACHED,PRIMARY)[index-3]
                ax.text(.02,.97,_metric_text(metrics[key]),transform=ax.transAxes,va='top',fontsize=11,
                    bbox=dict(facecolor='white',alpha=.85))
                if index==5:ax.text(.02,.88,'선택 origin: '+case['continuous_selection_origin'],transform=ax.transAxes,
                    va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85))
            path=assets/f'case{number}_stage{index+1}.png';DRAW._save(fig,path,CAPTIONS[index])
            case['stages'].append(dict(title=title,caption=CAPTIONS[index],asset=str(path.relative_to(OUTPUT))))
        cases.append(case);overview.append((fid,image,bounds,qold,qcached,qnew,metrics,case['continuous_selection_origin']))
    fig,axes=plt.subplots(3,len(cases),figsize=(5*len(cases),10),squeeze=False)
    for column,(fid,image,bounds,qold,qcached,qnew,metrics,origin)in enumerate(overview):
        for row,(q,color,key,label)in enumerate(((qold,DRAW.BLUE,BASELINE,'기존 e'),
                (qcached,DRAW.ORANGE,CACHED,'새 cached native'),(qnew,DRAW.GREEN,PRIMARY,'새 continuous native'))):
            ax=axes[row,column];ax.imshow(image)
            if q is not None:DRAW._wire(ax,q,color,width=1.5)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set_axis_off()
            ax.set_title((fid+'\n'if row==0 else '')+label+': '+_metric_text(metrics[key])+('\n'+origin if row==2 else ''),fontsize=9)
    fig.suptitle('실제 사진·같은 native-only 방법 쌍 · wire는 저장 모델투영\n사례는 봉인 점수 후 진단용 선택 / 참조·GT wire 없음',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUTPUT/'overview_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    C.write(OUTPUT/'PROCESS_DATA.json',dict(schema='REFINED_GRAZING_NMS_REAL_PROCESS_V18C',cases=cases,
        primary_native_pair_fixed_before_case_selection=True,primary_continuous=PRIMARY,primary_cached=CACHED,
        selected_ids=ids,case_selection='031344 plus maximum REMAINDER native-only translation, plus largest TEST translation regression if any',
        source_bindings=bindings,new_RGB_or_NN_or_line_or_pose_fit=0,new_reference_pose_or_corner_reads=0,
        stored_metrics_only=True,S_raw_not_read=True,U_not_precise_point=True,default_e_unchanged=True))
    sections=[]
    for case in cases:
        table=''.join(f'<tr><td>{html.escape(NAMES[key])}</td><td>{_metric_text(value)}</td></tr>'for key,value in case['metrics'].items())
        stages=''.join(f'<h3>{s["title"]}</h3><img src="{s["asset"]}" alt="실제사진 저장 처리단계"><p>{s["caption"]}</p>'for s in case['stages'])
        sections.append(f'<details open><summary>{html.escape(case["id"])}</summary><p>상태: {" ".join(case["current_states"])}</p><table><tr><th>방법</th><th>봉인 참조 오차</th></tr>{table}</table><a href="{_url(C.ROOT/case["image"])}">원본 RGB</a>{stages}</details>')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 NMS·U 범위·연속 PnP 과정</title><style>body{{margin:0;background:#edf2f6;color:#173044;font-family:system-ui,sans-serif}}main{{max-width:1080px;padding:18px;margin:auto}}img{{display:block;max-width:100%;height:auto}}details{{background:white;margin:18px 0;padding:16px;border-radius:10px}}summary{{font-weight:bold}}p{{line-height:1.6}}table{{border-collapse:collapse;max-width:100%;font-size:14px}}td,th{{padding:7px;border:1px solid #cbd6df;text-align:left}}@media(max-width:500px){{main{{padding:8px}}details{{padding:9px}}table{{font-size:12px}}}}</style><main><h1>실제 NMS·U 범위·연속 PnP 과정</h1><p>모든 사례에서 같은 native-only cached/continuous 쌍을 표시했습니다. 표에는 native+mask control도 포함합니다. 031344와 봉인 점수 후 최대오차·회귀 사례는 진단을 위해 골랐으며 런타임 방법 선택에는 쓰이지 않았습니다.</p><p>선·점은 실제 봉인 관측, wire는 저장 모델투영입니다. S 원좌표·U 정밀 코너·참조/GT 자세를 그리지 않았고 새 fitting도 하지 않았습니다. 추가143의 참조는 기존 nominal 주석 PnP이며 독립 실측은 아닙니다. TEST와143은 반복 확인한 탐색 자료입니다. 현재 e 기본값은 이 보고서로 바뀌지 않습니다.</p><img src="overview_actual.png" alt="세 방법의 실제사진 비교"><p><a href="PROCESS_DATA.json">모든 표시 값·원자료·출처 JSON</a> · <a href="{_url(DRAW.OUTPUT/'index.html')}">기존49 e 처리 과정</a></p>{''.join(sections)}</main></html>'''
    (OUTPUT/'index.html').write_text(page)
    for relative,digest in bindings.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Bound data changed while drawing')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(source_sha256=C.sha(Path(__file__)),input_sha256=bindings,
        assets={str(p.relative_to(OUTPUT)):C.sha(p)for p in OUTPUT.rglob('*')if p.is_file()},cases=len(cases),
        stages_per_case=6,actual_RGB_only=True,all_actual_alternatives_in_JSON=True,
        no_reference_pose_or_corner_reads_or_overlays=True,new_inference_or_fit=0))
    print(dict(output=str(OUTPUT),cases=ids,primary=PRIMARY))


if __name__=='__main__':render()
