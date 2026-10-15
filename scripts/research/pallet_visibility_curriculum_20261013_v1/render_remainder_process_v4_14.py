"""Actual-image process report for the postsealed highest-error remaining cases.

Read sealed observations, predictions and the independently declared derived
reference score.  Draw stored data only: no backbone, line or pose fitting.
Reference coordinates are decoded only after the complete143 pose/score gates.
The final49 renderer and its artifacts are left unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import html
import os
import shutil
from urllib.parse import quote

import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import common as C
from . import score_remainder_derived_reference_v4_14 as SCORE
from . import render_frozen_wd_process_v4_13e_v2 as DRAW
from . import score_remainder_closed_sources_v4_14 as CLOSED

ROOT=SCORE.REMAINDER
OBSERVATIONS=ROOT/'fresh_observations_v4_14'
OUTPUT=ROOT/'worst_actual_process_report_v4_14'
TITLES=('① 저장 입력 예측 — 최초 S 제외',
    '② 저장 coarse 자세와 최초 기하 S/B/I',
    '③ RGB 가시성 + 최초 기하역할 상태',
    '④ 실제 SAM 마스크와 지지된 RGB 선',
    '⑤ 보관 BI점과 부분 법선 관측',
    '⑥ O/U 영역과 부분 B/I 접선 범위',
    '⑦ 고정 W/D 안에서 최종 연속 PnP',
    '⑧ 사후 참조 비교 — 저장된 nominal PnP 참조')
CAPTIONS=(
    '회색 × = 저장된 N3+Sub 예측, 이미지 KP 실측 아님. 최초 S의 원좌표는 읽거나 표시하지 않음.',
    '점선과 삼각형 = 저장 coarse 모델투영. 영상 관측 아님. 초기 S/B/I 역할은 이 자세에 조건부.',
    '가시성은 고정 RGB 판단, B/I 종류는 최초 기하역할. O/U는 추정 상태이며 물리 인증 아님. U는 정밀 KP로 채택하지 않음.',
    '초록 틴트 = 실제 full-box SAM 마스크. 초록 선/표본 = 저장 RGB 지지. 새 마스크나 선 추론 없음.',
    '자홍 + = 보관 BI점(원예측 fallback 포함), 회색 × = 남은 예측 fallback. 초록은 실제 선. 부분 관측의 주황 □는 법선 투영 표시이며 완전한 2D KP가 아님.',
    'O 경험영역, U·접선은 CAL 최대오차에서 빌린 범위. 녹색 폭은 기존 공분산 근사이며 확률보증 아님. 사진 밖 범위는 보이는 부분만 잘라 그림.',
    '파랑 = 해당 native 모델의 최종 PnP 투영, S는 보라 모델투영만. OLD 영상 목적함수가 고른 W/D를 고정하고 d가 R/t를 연속 최적화.',
    '주황 = OLD, 파랑 = 최종, 흰 점선 = 저장된 nominal 주석 PnP 참조. 축 부호 미확정·카메라 품질 UNKNOWN, 독립 실측 GT 아님. 참조는 런타임에 사용하지 않음.')


def _model_projection(pose,family,K):
    q,depth=DRAW.V.project_native(np.asarray(family['native_xyz']),np.asarray(pose['R_cf']),
        np.asarray(pose['centroid']),np.asarray(K))
    return q, bool(np.isfinite(q).all() and np.all(depth>0))


def _reference_projection(reference,K):
    if reference['R'] is None:return None
    return C.project(C.cuboid(*reference['xyz']),reference['R'],reference['t'],K)


def _visible_points(ax,visible,excluded):
    for record in visible:
        k=record['native_corner']
        if k in excluded:continue
        if record['state'] not in ('B','I'):
            raise ValueError('Only fixed BI may be shown as an image measurement')
        fallback=record['source']=='VISIBLE_BACKBONE_FALLBACK'
        DRAW._points(ax,{k:record['xy']},'#61758a' if fallback else DRAW.MAGENTA,
            'x' if fallback else '+',prefix=record['state']+(' 예측q' if fallback else ' RGBq'))


def _gated_inputs(score_folder):
    closure=CLOSED.verify_additional_source_closure()
    closed_receipt=C.read(score_folder/'SOURCE_CLOSED_SCORE_RECEIPT_V1.json')
    if (not closed_receipt['passed'] or closed_receipt['source_closure']!=closure
        or C.sha(score_folder/'SCORE_SEAL.json')!=closed_receipt['score_seal_sha256']
        or C.sha(score_folder/'POSE_REFERENCE_SCORE.json')!=closed_receipt['score_sha256']):
        raise ValueError('Additive source closure and closed score receipt required')
    plan=C.read(score_folder/'PLAN.json');preparation=C.read(score_folder/'PREPARATION_SEAL.json')
    if C.sha(score_folder/'PLAN.json')!=preparation['plan_sha256']:
        raise ValueError('Fixed reference preparation changed')
    score_seal=C.read(score_folder/'SCORE_SEAL.json');score_path=score_folder/'POSE_REFERENCE_SCORE.json'
    if (score_seal['rows']!=143 or not score_seal['all_labels_read_after_complete_pose_prediction_seal']
        or C.sha(score_path)!=score_seal['score_sha256']
        or C.sha(score_folder/'PREPARATION_SEAL.json')!=score_seal['preparation_seal_sha256']):
        raise ValueError('Postseal143 derived reference score required')
    predictions=C.ROOT/plan['predictions'];protocol=C.read(predictions/'PROTOCOL.json')
    seal=C.read(predictions/'PREDICTION_SEAL.json');journal=predictions/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=143 or not seal['before_remaining_GT_or_metric_read']
        or C.sha(journal)!=seal['predictions_sha256']
        or C.sha(predictions/'PROTOCOL.json')!=seal['protocol_sha256']
        or C.sha(predictions/'PREDICTION_SEAL.json')!=score_seal['prediction_seal_sha256']
        or seal['ordered_ids']!=plan['ids']):
        raise ValueError('Same full143 prediction seal required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        SCORE._verify(plan[key]);SCORE._verify(protocol[key])
    obs_seal=C.read(OBSERVATIONS/'OBSERVATION_SEAL.json');obs_journal=OBSERVATIONS/'OBSERVATIONS.jsonl.gz'
    if (obs_seal['rows']!=143 or obs_seal['pose_solver_calls']!=0 or obs_seal['annotations_read']
        or C.sha(obs_journal)!=obs_seal['observations_sha256'] or obs_seal['ordered_ids']!=plan['ids']):
        raise ValueError('Original complete143 actual observation binding required')
    scores=C.read(score_path);rows=list(C.rows(journal));observations=list(C.rows(obs_journal))
    if ([r['id'] for r in rows]!=plan['ids'] or [r['id'] for r in scores['rows']]!=plan['ids']
        or [r['id'] for r in observations]!=plan['ids']):
        raise ValueError('Keep every scored143; no failure/native filter')
    mask_seal=C.read(OBSERVATIONS/'masks/SEAL.json');mask_records=C.read(OBSERVATIONS/'masks/RECORDS.json')
    if mask_seal['records']!=143 or C.sha(OBSERVATIONS/'masks/RECORDS.json')!=mask_seal['records_sha256']:
        raise ValueError('Actual143 mask record binding changed')
    return (plan,protocol,scores,{r['id']:r for r in rows},{r['id']:r for r in observations},
            {r['id']:r for r in mask_records},predictions)


def _relative_url(path,output):
    return quote(os.path.relpath(Path(path).resolve(),output),safe='/')


def render(output=OUTPUT,score_folder=SCORE.OUTPUT,case_count=3):
    output=Path(output).resolve();score_folder=Path(score_folder).resolve()
    if output.exists():raise FileExistsError('Preserve every earlier actual-image report')
    if not 1<=case_count<=5:raise ValueError('One to five actual highest-error cases')
    plan,protocol,score,predictions,observations,masks,prediction_folder=_gated_inputs(score_folder)
    scored={r['id']:r for r in score['rows']}
    # This postscore diagnostic chooses pictures; it never chooses production poses.
    ids=sorted(plan['ids'],key=lambda fid:(-scored[fid]['metric']['translation_cm'],plan['ids'].index(fid)))[:case_count]
    annotations={r['id']:r for r in plan['annotations']}
    font=DRAW._font();output.mkdir(parents=True);assets=output/'assets';assets.mkdir()
    bindings={str(p.relative_to(C.ROOT)):C.sha(p) for p in [Path(__file__),Path(DRAW.__file__),Path(SCORE.__file__),Path(CLOSED.__file__),
        score_folder/'PLAN.json',score_folder/'PREPARATION_SEAL.json',score_folder/'SCORE_SEAL.json',
        score_folder/'SOURCE_CLOSED_SCORE_PREPARATION_V1.json',score_folder/'SOURCE_CLOSED_SCORE_RECEIPT_V1.json',
        score_folder/'POSE_REFERENCE_SCORE.json',prediction_folder/'PROTOCOL.json',
        prediction_folder/'PREDICTION_SEAL.json',prediction_folder/'PREDICTIONS.jsonl.gz',
        OBSERVATIONS/'OBSERVATION_SEAL.json',OBSERVATIONS/'OBSERVATIONS.jsonl.gz',
        OBSERVATIONS/'masks/SEAL.json',OBSERVATIONS/'masks/RECORDS.json',
        CLOSED.CLOSURE/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json']}
    if font.is_file():bindings[str(font)]=C.sha(font)
    previous_report=DRAW.OUTPUT
    previous_data=C.read(previous_report/'PROCESS_DATA.json')
    previous_summary=previous_data['all49_summary']
    for path in (previous_report/'PROCESS_DATA.json',previous_report/'index.html'):
        bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
    previous_href=_relative_url(previous_report/'index.html',output)
    table=[]
    for rank,fid in enumerate(sorted(plan['ids'],key=lambda fid:-scored[fid]['metric']['translation_cm']),1):
        row=predictions[fid];result=scored[fid];image=C.ROOT/row['image']
        if not image.is_file() or C.sha(image)!=row['image_sha256']:
            raise ValueError('All143 original image links must resolve to the bound RGB')
        bindings[row['image']]=row['image_sha256']
        table.append(dict(rank=rank,id=fid,session=result['session'],severity=result['severity'],states=row['states'],
            translation_cm=result['metric']['translation_cm'],rotation_deg=result['metric']['rotation_deg'],
            fallback=not result['accepted'],image=row['image'],image_sha256=row['image_sha256'],
            original_RGB_relative_url=_relative_url(image,output)))
    cases=[];pictures=[]
    for index,fid in enumerate(ids):
        row=observations[fid];e=predictions[fid];d=e['partial_global_outline'];initial=row['runtime']['initial_geometry']
        if e['image_sha256']!=row['image_sha256'] or e['states']!=row['states']:
            raise ValueError('Image/state source differs from actual observation')
        K=initial['K'];hidden=initial['initial_self_hidden'];image_path=C.ROOT/e['image']
        if C.sha(image_path)!=e['image_sha256']:raise ValueError('Original image changed')
        image_bgr=cv2.imread(str(image_path))
        if image_bgr is None or list(image_bgr.shape[:2])!=initial['image_hw']:
            raise ValueError('Actual RGB canvas mismatch')
        image=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
        original=assets/f'case{index}_original{image_path.suffix}';shutil.copyfile(image_path,original)
        bindings[e['image']]=e['image_sha256']
        record=masks[fid];mask_path=OBSERVATIONS/'masks'/record['mask_npz']
        if record['image_sha256']!=e['image_sha256'] or C.sha(mask_path)!=record['mask_file_sha256']:
            raise ValueError('Frozen actual mask changed')
        bindings[str(mask_path.relative_to(C.ROOT))]=record['mask_file_sha256']
        with np.load(mask_path,allow_pickle=False) as arrays:mask=arrays['pallet_mask_clean'].copy()
        if mask.shape!=image.shape[:2]:raise ValueError('Actual mask canvas mismatch')
        raw={k:row['correction']['raw_points'][k] for k in range(8) if not hidden[k] and row['states'][k]!='S'}
        typed=d['face_observation']['boundary_observations']
        if d['face_observation']['S_local_queries'] or d['face_observation']['U_corner_queries']:
            raise ValueError('No S or U local image query may be drawn as evidence')
        merged=d['merged_partial_observation']['boundary_observations']
        family=next(f for f in d['arms'][DRAW.P.D.ARMS[1]]['families'] if f['name']==e['frozen_family'])
        old_family=next(f for f in e['old_global_outline']['families'] if f['name']==e['frozen_family'])
        final=e['arms'][SCORE.ARM];before=e['arms']['OLD_G10']
        q,positive=_model_projection(final['actual_pose'],family,K)
        qold,_=_model_projection(before['actual_pose'],old_family,K)
        annotation=annotations[fid];annotation_path=C.ROOT/annotation['annotation']
        if C.sha(annotation_path)!=annotation['annotation_sha256']:raise ValueError('Postseal reference changed')
        bindings[annotation['annotation']]=annotation['annotation_sha256']
        reference=SCORE.reference_from_document(C.read(annotation_path))
        metric=SCORE.metric_for_reference(final['actual_pose'],reference)
        if metric!=scored[fid]['metric']:raise ValueError('Same postseal reference metric required')
        reference_q=_reference_projection(reference,K)
        old_metric=SCORE.metric_for_reference(before['actual_pose'],reference)
        case=dict(id=fid,image_sha256=e['image_sha256'],states=e['states'],frozen_family=e['frozen_family'],
            raw_nonS_forecasts=raw,initial_geometry=deepcopy(initial),
            original_coarse_pose=deepcopy(row['_original_source']['original_coarse_pose']),
            typed_partials=typed,merged_partials=merged,fixed_BI_measurements=row['frozen_BI_fit'],
            O_T_domains=row['_original_source']['ot_domains']['domains'],U_regions=row['u_region_observation']['u_regions'],
            final_model_projection=q.tolist(),final_model_depth_positive=positive,
            final_pose=final['actual_pose'],postseal_metric=metric,old_postseal_metric=old_metric,
            postseal_reference=reference,postscore_picture_choice_not_runtime_routing=True,
            S_raw_not_indexed_or_shown=True,U_is_range_only=True,stages=[])
        for stage,title in enumerate(TITLES):
            fig,ax=DRAW._base(image,title,expanded=stage==5)
            if stage==0:DRAW._points(ax,raw,'#61758a','x',prefix='예측q')
            elif stage==1:
                coarse=np.asarray(initial['initial_projection_xy']);DRAW._wire(ax,coarse,DRAW.ORANGE,ls='--')
                for k,role in enumerate(initial['initial_roles']):
                    label='S' if hidden[k] else role
                    DRAW._points(ax,{k:coarse[k]},DRAW.COLORS[label],'^',prefix='모델'+label)
            elif stage==2:
                for k,point in raw.items():DRAW._points(ax,{k:point},DRAW.COLORS[row['states'][k]],'x',prefix=row['states'][k]+' 입력위치')
            elif stage==3:
                ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.22]));DRAW._partial(ax,merged)
            elif stage==4:
                DRAW._partial(ax,merged);_visible_points(ax,row['frozen_BI_fit'],{r['native_corner'] for r in merged})
            elif stage==5:
                DRAW._domains(ax,case['O_T_domains'],case['U_regions']);DRAW._partial(ax,merged,range_only=True)
            elif stage==6:
                DRAW._wire(ax,q,DRAW.BLUE,labels=True)
                DRAW._points(ax,{k:q[k] for k,s in enumerate(row['states']) if s=='S'},DRAW.PURPLE,'^',prefix='S모델',offset=(-40,-14))
            else:
                DRAW._wire(ax,qold,DRAW.ORANGE,ls='--');DRAW._wire(ax,q,DRAW.BLUE,width=1.8)
                if reference_q is not None:DRAW._wire(ax,reference_q,'white',ls=':',width=2.6)
                else:ax.text(.02,.03,'회전 참조 미확정: 참조 wire 생략, 위치 오차는 유지',transform=ax.transAxes,color='white',fontsize=9)
            path=assets/f'case{index}_stage{stage+1}.png';DRAW._save(fig,path,CAPTIONS[stage])
            case['stages'].append(dict(title=title,asset=str(path.relative_to(output)),caption=CAPTIONS[stage]))
        cases.append(case);pictures.append((image,qold,q,reference_q,old_metric,metric,fid))
    fig,axes=plt.subplots(2,len(pictures),figsize=(5.1*len(pictures),7.9),squeeze=False)
    for index,(image,qold,q,reference_q,old_metric,metric,fid) in enumerate(pictures):
        for i,(points,color,name,values) in enumerate(((qold,DRAW.ORANGE,'OLD',old_metric),(q,DRAW.BLUE,'최종',metric))):
            ax=axes[i,index];ax.imshow(image)
            if reference_q is not None:DRAW._wire(ax,reference_q,'white',ls=':',width=2.2)
            DRAW._wire(ax,points,color,width=1.8);ax.set_axis_off();ax.set_aspect('equal')
            rotation='미확정' if values['rotation_deg'] is None else f'{values["rotation_deg"]:.3f}°'
            ax.set_title(f'{name}: {values["translation_cm"]:.3f}cm / {rotation}',fontsize=12)
            if i==0:ax.text(.5,1.13,fid,transform=ax.transAxes,ha='center',fontsize=9)
    fig.suptitle('실제 remaining143 위치 오차 상위 사례 · 봉인된 자세 그대로\n흰 점선 = 사후 저장 PnP 참조, 독립 실측 GT 아님',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(output/'overview_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    summary=score['summary']
    C.write(output/'PROCESS_DATA.json',dict(schema='FRESH143_WORST_ACTUAL_PROCESS_REPORT_V4_14',
        cases=cases,all143_summary=summary,all143_metric_table=table,first49_summary=previous_summary,
        first49_report_relative_url=previous_href,
        case_selection='highest nominal translation errors after score only',
        bindings=bindings,actual_RGB_images=True,no_AI_generated_image=True,new_inference_or_pose_fit=0,
        reference_used_only_postseal_and_display=True,nominal_derived_reference_policy=SCORE.POLICY,
        reference_independently_measured=False,range_and_normal_noise_not_newly_calibrated=True))
    sections=[]
    for index,case in enumerate(cases):
        metric=case['postseal_metric'];rotation='미확정' if metric['rotation_deg'] is None else f'{metric["rotation_deg"]:.3f}°'
        buttons=''.join(f'<button onclick="stage({index},{k})">{html.escape(s["title"][:15])}</button>' for k,s in enumerate(case['stages']))
        panels=''.join(f'<div class="step" id="c{index}s{k}" style="display:{"block" if k==0 else "none"}"><h2>{html.escape(s["title"])}</h2><img src="{s["asset"]}" alt="실제 사진 처리단계"><p>{html.escape(s["caption"])}</p></div>' for k,s in enumerate(case['stages']))
        sections.append(f'<section class="case" id="case{index}" style="display:{"block" if index==0 else "none"}"><h2>{html.escape(case["id"])}</h2><p>최종 {metric["translation_cm"]:.3f}cm / {rotation} · 고정 W/D: {case["frozen_family"]}</p><nav>{buttons}</nav>{panels}</section>')
    chooser=''.join(f'<option value="{index}">{html.escape(case["id"])}</option>' for index,case in enumerate(cases))
    table_html=[]
    for row in table:
        rotation='미확정' if row['rotation_deg'] is None else f'{row["rotation_deg"]:.3f}'
        table_html.append(f'<tr><td>{row["rank"]}</td><td>{html.escape(row["id"])}</td><td>{html.escape(row["session"])}</td><td>{html.escape(row["severity"])}</td><td>{html.escape(" ".join(row["states"]))}</td><td>{row["translation_cm"]:.3f}</td><td>{rotation}</td><td>{"fallback" if row["fallback"] else "채택"}</td><td><a href="{row["original_RGB_relative_url"]}" target="_blank" rel="noopener">원본 RGB</a></td></tr>')
    all_table='<section class="note"><h2>전체143 결과 — nominal 위치 오차 순</h2><p>각 행은 동일한 저장 참조 중심으로 계산했습니다. 원본 링크는 workspace의 실제 사진을 가리키며 이 페이지에 143개 사진을 복사하지 않았습니다. 표는 좌우로 스크롤할 수 있습니다.</p><div class="table-scroll"><table><thead><tr><th>순위</th><th>프레임</th><th>세션</th><th>심각도</th><th>상태 0–7</th><th>위치 cm</th><th>회전 °</th><th>출력</th><th>사진</th></tr></thead><tbody>'+''.join(table_html)+'</tbody></table></div></section>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 remaining143 오차 상위 처리 과정</title><style>body{{font-family:system-ui,sans-serif;margin:0;background:#edf2f6;color:#173044}}main{{max-width:1120px;margin:auto;padding:22px}}h1{{font-size:25px}}h2{{font-size:20px}}img{{max-width:100%;height:auto;display:block;background:white}}select{{width:100%;padding:12px;font-size:16px}}nav{{display:flex;gap:6px;flex-wrap:wrap;margin:15px 0}}button{{padding:10px;border:1px solid #c7d1d9;background:white;border-radius:7px;cursor:pointer}}.case,.note{{background:white;padding:18px;margin:18px 0;border-radius:12px}}p{{line-height:1.6}}.step p{{background:#f1f5f8;padding:10px}}a{{color:#17658c}}@media(max-width:500px){{main{{padding:10px}}.case{{padding:10px}}h2{{font-size:17px}}}}</style><main><h1>실제 remaining143: 위치 오차 상위 처리 과정</h1><p>전체143개: 평균 {summary['translation_cm']['mean']:.3f}cm, 최대 {summary['translation_cm']['max']:.3f}cm, 5cm 이하 {summary['translation_le5']}/143. 회전 평가 가능 {summary['rotation_eligible_rows']}/143. 실패·fallback·축 미확정 사례를 제외하지 않았습니다.</p><p>저장된 수동 주석 PnP의 nominal 중심에 대한 오차입니다. canonical 부호 미확정이며 카메라 품질 UNKNOWN입니다. 다른 참조 가설을 오차가 작아지는 방향으로 선택하지 않습니다.</p><img src="overview_actual.png" alt="실제 오차 상위 사진의 이전·최종 비교"><label>사례 선택<select onchange="choose(this.value)">{chooser}</select></label>{''.join(sections)}<div class="note"><b>표시 구분</b><p>예측 ×, 보관 RGB BI +, 실제 선·인라이어 초록, 부분 법선표시 주황 □, 최종 모델투영 파랑 ◇, S 모델투영 보라 △. 부분 선 표시는 완전한 KP가 아닙니다. U는 범위만 사용하고 U/S 코너 영상 조회를 추가하지 않습니다.</p><p>143개는 기존 자료에 고정 방법을 새로 실행한 결과이며 새 독립 실측 또는 무접촉 검증셋을 뜻하지 않습니다. 참조는 final 표시와 봉인 후 오차 계산에만 사용했습니다. O/U 영역·접선 범위·분산은 기존 CAL/조건부 근사이며 새 확률보증이 없습니다. 이 페이지는 저장 데이터만 그렸고 새 추론·선 fitting·포즈 fitting을 하지 않았습니다.</p><a href="PROCESS_DATA.json">값·출처 확인 JSON</a></div></main><script>function choose(k){{document.querySelectorAll('.case').forEach((e,i)=>e.style.display=i==k?'block':'none')}}function stage(c,k){{document.querySelectorAll('#case'+c+' .step').forEach((e,i)=>e.style.display=i==k?'block':'none')}}</script></html>'''
    # Keep the process UI intact while adding the complete population table.
    first49=f'<div class="note"><h2>앞선 TEST49</h2><p>49/49가 5cm·5° 이내, 평균 {previous_summary["translation_cm"]["mean"]:.2f}cm, 최대 {previous_summary["translation_cm"]["max"]:.2f}cm입니다. 같은 저장 참조/C2 평가 기준의 결과이며 독립 실측 GT 또는 새 무접촉 검증 결과를 뜻하지 않습니다. <a href="{previous_href}">앞선 실제사진 과정 V2</a></p></div>'
    page=page.replace('</style>',' .table-scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #d7e0e6}th{white-space:nowrap;background:#f1f5f8}td:nth-child(2){overflow-wrap:anywhere;min-width:140px}td:nth-child(5){white-space:nowrap} </style>')
    page=page.replace('<img src="overview_actual.png"',first49+'<img src="overview_actual.png"')
    page=page.replace('</main><script>',all_table+'</main><script>')
    (output/'index.html').write_text(page)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        SCORE._verify(plan[key]);SCORE._verify(protocol[key])
    for relative,digest in bindings.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Renderer input changed: '+relative)
    hashes={str(p.relative_to(output)):C.sha(p) for p in output.rglob('*') if p.is_file()}
    C.write(output/'RENDER_SEAL.json',dict(source_sha256=C.sha(__file__),input_sha256=bindings,
        output_sha256=hashes,case_count=len(cases),steps_per_case=8,
        actual_source_photos=True,model_geometry_not_GT_reordered=True,S_raw_points_hidden=True,
        reference_only_postseal_display=True,new_inference_or_pose_fit=0,protected6_unchanged=True))
    print('REMAINDER_ACTUAL_PROCESS_REPORT_READY',output,ids,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--score-folder',type=Path,default=SCORE.OUTPUT);parser.add_argument('--case-count',type=int,default=3)
    args=parser.parse_args();render(args.output,args.score_folder,args.case_count)
