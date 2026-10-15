"""Six distinct stored-image stages for the three requested postseal cases."""
from pathlib import Path
import html
import os
from urllib.parse import quote

import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import common as C
from . import evaluate_two_stage_top_normal_v4_16c as X
from . import render_two_stage_top_normal_v4_16c_v2 as COMPARE
from . import render_remainder_process_v4_14 as OLD143
from . import render_frozen_wd_process_v4_13e_v2 as DRAW

OUTPUT=X.OUTPUT.parent/'top_normal_process_supplement_v4_16c_v3'
IDS=('wood_night_01:031344','wood_night_01:031426','eval_pallet07:1778652150610404864')
ARM=X.ARMS[1]
TITLES=('① 최초 non-S 입력 예측','② 최초 기하와 고정 RGB 상태','③ 실제 SAM 마스크와 BI/RGB 선 관측',
        '④ O/U 및 부분 B/I 접선 범위','⑤ 새 Stage1: 양쪽 W/D → 선택 가족','⑥ Stage2: 같은 가족 안에서 연속 정제')
CAPTIONS=(
    '회색 ×는 저장 N3+Sub 예측이며 실제 KP 실측이 아님. 최초 S의 원좌표는 읽거나 표시하지 않음.',
    '점선/삼각형은 최초 coarse 모델투영. 초기 S/B/I 역할은 모델에 조건부이며, 색 ×는 저장 RGB 상태의 입력 위치. S는 모델투영만.',
    '초록 틴트는 저장된 실제 full SAM 마스크, 초록 선/점은 저장 RGB 지지. 자홍 +는 보관 BI 관측. 부분 법선의 주황 표시는 완전한 2D KP가 아님.',
    'O 경험영역, U 및 부분 접선은 기존 CAL 오차에서 빌린 범위. 확률보증이나 정밀 KP가 아니며 U에는 점 비용이 없음. 영상 밖 부분은 잘려 보이고 원 수치는 JSON에 보존.',
    '양쪽 W/D의 원 BI·기존 부분선·ALL_T4·전역 마스크·상단면 사전을 같은 목적으로 평가한 저장 Stage1 결과. GT/참조 또는 새 fitting 없음.',
    'Stage1이 고른 가족을 고정하고 D 부분관측+ALL_T4+같은 상단면 사전으로 정제한 저장 자세. 녹색 wire는 PnP 모델투영이며 이미지 측정점이 아님.')


def _url(path):
    return quote(os.path.relpath(Path(path).resolve(),OUTPUT),safe='/')


def render():
    if OUTPUT.exists():raise FileExistsError('Preserve every earlier report')
    predseal=C.read(X.OUTPUT/'PREDICTION_SEAL.json');scoreseal=C.read(X.OUTPUT/'SCORE_SEAL.json')
    X.E._verify_binding(scoreseal['prediction_seal'])
    if predseal['rows']!=238 or not predseal['all3_arms_complete']:raise ValueError('Whole238 required')
    protocol=C.read(X.OUTPUT/'PROTOCOL.json')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):X.E._verify_sources(protocol[key])
    X.CLOSED.verify_additional_source_closure()
    tasks={t['id']:t for t in C.rows(X.ARCHIVE.OUTPUT/'INPUTS.jsonl.gz')if t['id']in IDS}
    rows={r['id']:r for r in C.rows(X.OUTPUT/'PREDICTIONS.jsonl.gz')if r['id']in IDS}
    scores={}
    for partition in ('TEST','REMAINDER'):
        path=X.OUTPUT/f'{partition}_SCORE.json'
        if C.sha(path)!=scoreseal['score_sha256'][partition]:raise ValueError('Frozen score required')
        scores.update({r['id']:r for r in C.read(path)['rows']if r['id']in IDS})
    compare_seal=C.read(COMPARE.OUTPUT/'RENDER_SEAL.json')
    for relative,digest in compare_seal['assets'].items():
        if C.sha(COMPARE.OUTPUT/relative)!=digest:raise ValueError('Comparison report changed')
    DRAW._font();OUTPUT.mkdir(parents=True);assets=OUTPUT/'assets';assets.mkdir()
    bindings={str(p.relative_to(C.ROOT)):C.sha(p)for p in (Path(__file__),X.OUTPUT/'PROTOCOL.json',
        X.OUTPUT/'PREDICTION_SEAL.json',X.OUTPUT/'SCORE_SEAL.json',X.OUTPUT/'PREDICTIONS.jsonl.gz',
        X.ARCHIVE.OUTPUT/'INPUT_SEAL.json',COMPARE.OUTPUT/'RENDER_SEAL.json',COMPARE.OUTPUT/'index.html')}
    cases=[];overview=[]
    for fid in IDS:
        t=tasks[fid];r=rows[fid];source=t['finish_input']['_original_source'];initial=source['initial_geometry']
        image_path=C.ROOT/t['image'];X.E._verify_binding(t['mask_binding'])
        if C.sha(image_path)!=t['image_sha256']:raise ValueError('Bound actual RGB required')
        bgr=cv2.imread(str(image_path));image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        with np.load(C.ROOT/t['mask_binding']['path'],allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
        if tuple(mask.shape)!=tuple(image.shape[:2]):raise ValueError('Mask canvas differs')
        bindings[t['image']]=t['image_sha256'];bindings[t['mask_binding']['path']]=t['mask_binding']['sha256']
        hidden=initial['initial_self_hidden'];states=t['spec']['states'];raw_source=source['original_raw_points']
        raw={k:raw_source[k]for k in range(8)if not hidden[k]and states[k]!='S'}
        audit=r['normal_two_stage_audit'][ARM];family=next(f for f in audit['stage1_all_families']if f['name']==audit['stage1_selected_family'])
        stage1=family['solution'];final=r['arms'][ARM];K=t['spec']['K']
        q1=None if not stage1['accepted']else C.project(family['native_xyz'],stage1['R'],stage1['t'],K)
        q2=C.project(family['native_xyz'],final['actual_pose']['R_cf'],final['actual_pose']['centroid'],K)
        qold=COMPARE._pose_projection(r['arms'][X.ARMS[0]]['actual_pose'],K)
        points=np.asarray(initial['initial_projection_xy']);lo=points.min(0);hi=points.max(0);padding=np.maximum(12,(hi-lo)*.3)
        bounds=(max(0,lo[0]-padding[0]),min(image.shape[1],hi[0]+padding[0]),
                min(image.shape[0],hi[1]+padding[1]),max(0,lo[1]-padding[1]))
        case=dict(id=fid,states=states,raw_nonS_forecasts=raw,initial_geometry=initial,
            actual_RGB_path=t['image'],actual_RGB_sha256=t['image_sha256'],mask_binding=t['mask_binding'],
            original_BI=t['spec']['visible'],actual_partial_observations=t['boundary_all'],
            O_T_domains=t['spec']['domains'],U_regions=t['u_regions'],stage1_selected_family=family['name'],
            stage1_pose=stage1,stage2_pose=final['actual_pose'],metrics=scores[fid]['metrics'],stages=[])
        for index,title in enumerate(TITLES):
            fig,ax=DRAW._base(image,title,expanded=index==3)
            if index!=3:ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
            if index==0:DRAW._points(ax,raw,'#61758a','x',prefix='예측q')
            elif index==1:
                coarse=np.asarray(initial['initial_projection_xy']);DRAW._wire(ax,coarse,DRAW.ORANGE,ls='--')
                for k,role in enumerate(initial['initial_roles']):
                    label='S'if hidden[k]else role
                    DRAW._points(ax,{k:coarse[k]},DRAW.COLORS[label],'^',prefix='모델'+label)
                for k,p in raw.items():DRAW._points(ax,{k:p},DRAW.COLORS[states[k]],'x',prefix=states[k]+' 예측')
            elif index==2:
                ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.24]))
                DRAW._partial(ax,t['boundary_all']);OLD143._visible_points(ax,t['spec']['visible'],{b['native_corner']for b in t['boundary_all']})
            elif index==3:
                DRAW._domains(ax,t['spec']['domains'],t['u_regions']);DRAW._partial(ax,t['boundary_all'],range_only=True)
            elif index==4:
                if q1 is not None:DRAW._wire(ax,q1,DRAW.GREEN,width=1.8)
                else:ax.text(.02,.05,'Stage1 joint-feasible 가족 없음; ORIGINAL fallback, correspondence 미해결',transform=ax.transAxes)
                ax.text(.02,.97,family['name'],transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.85),fontsize=10)
            else:
                DRAW._wire(ax,q2,DRAW.GREEN,width=1.8,ls='-'if final['accepted']else '--')
                DRAW._points(ax,{k:q2[k]for k,s in enumerate(states)if s=='S'},DRAW.PURPLE,'^',prefix='S모델')
            path=assets/f'case{len(cases)}_stage{index+1}.png';DRAW._save(fig,path,CAPTIONS[index])
            case['stages'].append(dict(title=title,caption=CAPTIONS[index],asset=str(path.relative_to(OUTPUT))))
        cases.append(case);overview.append((fid,image,bounds,qold,q2,scores[fid]['metrics']))
    fig,axes=plt.subplots(2,3,figsize=(16,8),squeeze=False)
    for col,(fid,image,bounds,old,new,metrics)in enumerate(overview):
        for row,(name,q,color)in enumerate(((X.ARMS[0],old,DRAW.BLUE),(ARM,new,DRAW.GREEN))):
            ax=axes[row,col];ax.imshow(image);DRAW._wire(ax,q,color,width=1.8)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set_axis_off()
            m=metrics[name];label='기존 e'if row==0 else 'σ2° + ALL_T4 두단계 control'
            ax.set_title(f"{fid}\n{label}: {m['translation_cm']:.2f}cm / {m['rotation_deg']:.2f}°",fontsize=10)
    fig.suptitle('실제사진: 143 개선 사례 · 남은143 최대오차 · 새49 회귀\n색 wire=저장 모델투영, 이미지 실측점 아님; 참조/GT wire 없음',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(OUTPUT/'overview_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    C.write(OUTPUT/'PROCESS_DATA.json',dict(schema='SIX_REAL_IMAGE_STAGES_TOP_NORMAL_V16C',cases=cases,
        control=ARM,source_bindings=bindings,case_selection_after_score_only=True,new_inference_or_fit=0,
        no_reference_coordinates_drawn=True,remaining_metrics_are_nominal_derived_not_independent_metrology=True,
        S_raw_not_indexed=True,U_only_regions=True,default_e_unchanged=True))
    sections=[]
    for k,case in enumerate(cases):
        stages=''.join(f'<h3>{s["title"]}</h3><img src="{s["asset"]}"><p>{s["caption"]}</p>'for s in case['stages'])
        sections.append(f'<details open><summary>{html.escape(case["id"])}</summary><a href="{_url(C.ROOT/case["actual_RGB_path"])}">원본 RGB</a>{stages}</details>')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제사진 6단계: 법선 + ALL_T4 control</title><style>body{{font-family:system-ui,sans-serif;background:#eef2f5;color:#17354b;margin:0}}main{{max-width:1100px;margin:auto;padding:20px}}img{{display:block;max-width:100%;height:auto}}details{{padding:16px;margin:20px 0;background:white;border-radius:10px}}summary{{font-weight:bold;cursor:pointer}}p{{line-height:1.6}}@media(max-width:500px){{main{{padding:8px}}details{{padding:9px}}}}</style><main><h1>실제사진 6단계 — σ2° + ALL_T4 두단계 control</h1><p>세 요청 사례를 저장 데이터로만 그렸습니다. 초기 non-S 예측, 고정 상태, 실제 마스크·관측선, 범위, 새 Stage1, 고정 가족 Stage2를 분리했습니다. O/U·접선은 CAL에서 빌린 범위이고 새 확률보증이 없습니다. 법선은 상관된 초기자세 사전이며 새 영상 관측 또는 유일성 증거가 아닙니다.</p><p>새 control은 TEST49 성공이 49→45로 줄어 채택하지 않았습니다. remaining143의 참조는 nominal 주석 PnP이며 축 부호 미확정·카메라 품질 UNKNOWN입니다. 독립 실측 GT가 아닙니다. 이 과정 패널은 참조/GT 좌표를 그리지 않았고 새 fitting을 하지 않았습니다.</p><img src="overview_actual.png"><p><a href="{_url(COMPARE.OUTPUT/'index.html')}">N3 / PoseFix / e / 두 control 최종 5방법 및 전체238 비교</a> · <a href="{_url(DRAW.OUTPUT/'index.html')}">기존49 성공 처리 과정</a> · <a href="{_url(OLD143.OUTPUT/'index.html')}">기존143 처리 과정 V14</a> · <a href="PROCESS_DATA.json">단계별 원자료·출처</a></p>{''.join(sections)}</main></html>'''
    (OUTPUT/'index.html').write_text(page)
    for relative,digest in bindings.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Evidence changed while drawing')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(cases=3,stages_per_case=6,source_sha256=C.sha(Path(__file__)),
        prediction_seal=X.E._binding(X.OUTPUT/'PREDICTION_SEAL.json'),score_seal=X.E._binding(X.OUTPUT/'SCORE_SEAL.json'),
        actual_RGB_only=True,new_forward_or_fit=0,artifacts={str(p.relative_to(OUTPUT)):C.sha(p)for p in OUTPUT.rglob('*')if p.is_file()}))


if __name__=='__main__':render()
