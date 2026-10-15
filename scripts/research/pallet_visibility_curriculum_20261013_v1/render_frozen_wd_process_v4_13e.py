"""Actual three-image process report from sealed e/d/OLD data, no inference."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
from pathlib import Path
import html
import shutil
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager,patheffects
from matplotlib.patches import Circle,Ellipse,Polygon,Rectangle
from . import common as C,statewise_evaluation_v4 as E,joint_ot_pnp as V
from . import select_frozen_wd_partial_refinement_v4_13e as P
from .interior_observation import NEIGHBORS
from .scoring import pose_metric

OUTPUT=P.OUTPUT.parent/'frozen_wd_process_report_v4_13e'
PREVIOUS=('eval_noapril:1775201430551447040','eval_pallet07:1778652140531310080')
GREEN='#009944';MAGENTA='#d423cb';ORANGE='#dc7800';BLUE='#00c8ff';PURPLE='#b262f3'
COLORS={'B':'#00bfff','I':'#8874ff','O':'#efa300','U':'#7d8995','S':PURPLE,'T':'#df5050'}
EDGES=[(k,m) for k in range(8) for m in NEIGHBORS[k] if k<m]
TITLES=('① 입력: N3+Sub 예측 — S 예측 제외','② 최초 기하분류 S/B/I — coarse 모델 조건부',
        '③ 고정 RGB 상태 — 표시 위치는 입력 예측','④ 실제 SAM 마스크와 RGB 선 표본',
        '⑤ B는 외곽선 / I는 내부선: 부분 관측','⑥ O/U 영역 + 부분 B/I 접선 범위',
        '⑦ OLD가 고른 W/D 안에서 연속 PnP','⑧ 사후 참조 비교 — GT는 이 패널만')


def _font():
    font=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    if font.is_file():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    return font


def _base(image,title,expanded=False):
    h,w=image.shape[:2];fig,ax=plt.subplots(figsize=(9.5,7.3),dpi=140)
    ax.imshow(image);ax.set_xlim(-.08*w,1.08*w);ax.set_ylim((1.28 if expanded else 1.06)*h,-.06*h)
    ax.add_patch(Rectangle((0,0),w,h,fill=False,ec='#778899',ls='--',lw=.8))
    ax.set_aspect('equal');ax.set_axis_off();ax.set_title(title,fontsize=13,pad=12)
    return fig,ax


def _label(ax,xy,text,color,offset=(5,7)):
    ax.annotate(text,xy,xytext=offset,textcoords='offset points',color=color,fontsize=9,
        path_effects=[patheffects.withStroke(linewidth=2.3,foreground='white',alpha=.8)])


def _points(ax,points,color,marker,*,prefix='',labels=True,offset=(5,7)):
    for k,q in points.items():
        q=np.asarray(q,float)
        if not np.isfinite(q).all():continue
        ax.plot(*q,marker=marker,color=color,ms=7,mew=1.4,mfc='none')
        if labels:_label(ax,q,prefix+str(k),color,offset)


def _wire(ax,q,color,*,ls='-',labels=False,width=1.4):
    for a,b in EDGES:ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=width,ls=ls)
    if labels:_points(ax,{k:q[k] for k in range(8)},color,'D',prefix='P')


def _pose_q(pose,family,K):
    q,depth=V.project_native(np.asarray(family['native_xyz']),np.asarray(pose['R_cf']),np.asarray(pose['centroid']),np.asarray(K))
    if np.any(depth<=0):raise ValueError('Positive depth required for displayed model')
    return q


def _reference_q(truth,K):
    # Physical body wire only: no native keypoint identity matching/reordering.
    return C.project(C.cuboid(*truth['body_xyz']),truth['body_R'],truth['t'],K)


def _partial(ax,records,*,range_only=False):
    for record in records:
        k=record['native_corner'];archive=record['association_provenance']
        q=np.asarray(archive.get('original_observed_xy',archive['original_raw_xy']),float)
        if not range_only:_points(ax,{k:q},MAGENTA,'+',prefix=record['state']+' 보관q',offset=(7,9))
        for line in record['lines']:
            n=np.asarray(line['normal_unit_xy']);t=np.array([-n[1],n[0]]);offset=line['offset_px'];p=line['provenance']
            if range_only:
                lo,hi=line['support_tangent_interval_px'];sigma=np.sqrt(line['normal_variance_px2'])
                quad=np.asarray([-offset*n+lo*t-sigma*n,-offset*n+hi*t-sigma*n,-offset*n+hi*t+sigma*n,-offset*n+lo*t+sigma*n])
                ax.add_patch(Polygon(quad,fc=GREEN,ec=GREEN,alpha=.12,lw=1.1))
                segment=-offset*n+np.asarray([lo,hi])[:,None]*t
                ax.plot(segment[:,0],segment[:,1],color=GREEN,lw=1,ls='--')
            else:
                ray=p.get('line_record')
                if ray:
                    points=np.asarray(ray['observed_points']);points=points[np.asarray(ray['inlier_mask'],bool)]
                    ax.scatter(points[:,0],points[:,1],s=12,c=GREEN)
                    lo,hi=ray['actual_inlier_tangent_interval_px']
                else:lo,hi=p['actual_observed_inlier_tangent_interval_px']
                segment=-offset*n+np.asarray([lo,hi])[:,None]*t
                ax.plot(segment[:,0],segment[:,1],color=GREEN,lw=2.2)
        if not range_only and archive.get('normal_projected_display_xy') is not None:
            _points(ax,{k:archive['normal_projected_display_xy']},ORANGE,'s',prefix='법선표시',offset=(7,-15))


def _domains(ax,domains,u):
    for d in domains:
        if d['state']!='O':continue
        if 'center_xy' in d:
            center=d['center_xy'];axes=d['half_axes_px'];angle=d.get('angle_deg',0)
            ax.add_patch(Ellipse(center,2*axes[0],2*axes[1],angle=angle,fc='#efa300',ec='#ce8500',alpha=.13,lw=1.4))
            _label(ax,center,'O'+str(d['native_corner'])+' 경험영역','#ac7200',(7,15))
    for d in u:
        center=d.get('center_xy',d.get('ellipse_center'));axes=d.get('half_axes_px',d.get('half_axes'))
        ax.add_patch(Ellipse(center,2*axes[0],2*axes[1],angle=d.get('angle_deg',0),fc='#8794a1',ec='#63717e',alpha=.12,lw=1.3,ls='--'))
        _label(ax,center,'U'+str(d['native_corner'])+' 빌린 범위','#57626e',(7,-14))


def _save(fig,path,caption):
    fig.text(.5,.012,caption,ha='center',fontsize=10,color='#354454',wrap=True)
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(path,bbox_inches='tight');plt.close(fig)


def render():
    if OUTPUT.exists():raise FileExistsError('Preserve prior actual report')
    protocol=C.read(P.OUTPUT/'PROTOCOL.json');seal=C.read(P.OUTPUT/'PREDICTION_SEAL.json');scoring=C.read(P.OUTPUT/'SCORE_SEAL.json')
    assert seal['rows']==95 and C.sha(P.OUTPUT/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
    assert C.sha(P.OUTPUT/'PROTOCOL.json')==seal['protocol_sha256'] and scoring['prediction_seal']['sha256']==C.sha(P.OUTPUT/'PREDICTION_SEAL.json')
    for field in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[field])
    score=C.read(P.OUTPUT/'TEST_SCORE.json');assert C.sha(P.OUTPUT/'TEST_SCORE.json')==scoring['score_sha256']['TEST']
    metric={r['id']:r for r in score['rows']};worst=max(metric,key=lambda k:metric[k]['metrics'][P.ARMS[2]]['translation_cm'])
    ids=list(dict.fromkeys((*PREVIOUS,worst)))
    prediction={r['id']:r for r in C.rows(P.OUTPUT/'PREDICTIONS.jsonl.gz')}
    d={r['id']:r for r in C.rows(P.PARTIAL/'PREDICTIONS.jsonl.gz')}
    old={r['id']:r for r in C.rows(P.OLD/'PREDICTIONS.jsonl.gz')}
    rows,_,_,_,_,_,_=P.D._load();inputs={r['id']:r for r in rows}
    target=E.COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json'
    assert C.sha(target)==score['target_binding']['sha256'];labels=C.read(target)['rows']
    font=_font();OUTPUT.mkdir(parents=True);assets=OUTPUT/'assets';assets.mkdir()
    bindings={str(p.relative_to(C.ROOT)):C.sha(p) for p in (Path(__file__),P.OUTPUT/'PREDICTION_SEAL.json',P.OUTPUT/'PREDICTIONS.jsonl.gz',P.OUTPUT/'TEST_SCORE.json',P.PARTIAL/'PREDICTIONS.jsonl.gz',P.OLD/'PREDICTIONS.jsonl.gz',target)}
    if font.is_file():bindings[str(font)]=C.sha(font)
    cases=[];pictures=[]
    for j,fid in enumerate(ids):
        row=inputs[fid];e=prediction[fid];dm=d[fid];initial=dm['initial_geometry'];K=initial['K'];hidden=initial['initial_self_hidden']
        image_path=C.ROOT/e['image'];assert C.sha(image_path)==e['image_sha256'];bindings[e['image']]=e['image_sha256']
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        shutil.copyfile(image_path,assets/f'case{j}_original.png')
        mask_path=P.D.OLD10.MASKS/row['_MASK']['mask_npz'];assert C.sha(mask_path)==row['_MASK']['mask_file_sha256'];bindings[str(mask_path.relative_to(C.ROOT))]=C.sha(mask_path)
        with np.load(mask_path,allow_pickle=False) as arrays:mask=arrays['pallet_mask_clean'].copy()
        # First S gate before every raw indexing; S is never an input marker.
        raw={k:row['correction']['raw_points'][k] for k in range(8) if not hidden[k] and e['states'][k]!='S'}
        typed=dm['face_observation']['boundary_observations'];assert dm['face_observation']['S_local_queries']==dm['face_observation']['U_corner_queries']==0
        family=next(f for f in dm['arms'][P.D.ARMS[1]]['families'] if f['name']==e['frozen_family'])
        final=e['arms'][P.ARMS[2]];before=e['arms'][P.ARMS[0]];q=_pose_q(final['actual_pose'],family,K)
        old_family=next(f for f in old[fid]['families'] if f['name']==e['frozen_family']);qold=_pose_q(before['actual_pose'],old_family,K)
        truth=labels[fid]['target']['truth'];reference=_reference_q(truth,K)
        m=metric[fid]['metrics'];assert pose_metric(final['actual_pose'],truth)==m[P.ARMS[2]]
        case=dict(id=fid,image_sha256=e['image_sha256'],states=e['states'],frozen_family=e['frozen_family'],
            raw_nonS_forecasts=raw,initial_geometry_is_conditional=True,
            typed_partials=typed,O_domains=row['_original_source']['ot_domains']['domains'],U_regions=row['u_region_observation']['u_regions'],
            final_model_projection=q.tolist(),final_pose=final['actual_pose'],postseal_metrics=m,
            reference_used_only_final_panel=True,S_raw_not_indexed_or_shown=True,new_inference_or_fit=0,stages=[])
        for stage,title in enumerate(TITLES):
            fig,ax=_base(image,title,expanded=stage==5)
            if stage==0:_points(ax,raw,'#61758a','x',prefix='예측q')
            elif stage==1:
                coarse=np.asarray(initial['initial_projection_xy']);_wire(ax,coarse,ORANGE,ls='--')
                for k,role in enumerate(initial['initial_roles']):_points(ax,{k:coarse[k]},COLORS['S' if hidden[k] else role],'^',prefix='모델'+('S' if hidden[k] else role))
            elif stage==2:
                for k,point in raw.items():_points(ax,{k:point},COLORS[e['states'][k]],'x',prefix=e['states'][k]+' 입력위치')
            elif stage==3:
                ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.22]));_partial(ax,typed)
            elif stage==4:_partial(ax,typed)
            elif stage==5:
                _domains(ax,case['O_domains'],case['U_regions']);_partial(ax,typed,range_only=True)
            elif stage==6:
                _wire(ax,q,BLUE,labels=True);_points(ax,{k:q[k] for k,s in enumerate(e['states']) if s=='S'},PURPLE,'^',prefix='S모델',offset=(-40,-14))
            else:
                _wire(ax,qold,ORANGE,ls='--');_wire(ax,reference,'white',ls=':',width=2.6);_wire(ax,q,BLUE,width=1.8)
            captions=(
                '회색 × = 저장 예측, 실제 KP 측정 아님. 최초 S의 예측좌표는 읽거나 표시하지 않음.',
                '점선·삼각형 = 저장 coarse 모델투영, 영상 관측이 아님. S/B/I는 이 초기 자세에 조건부.',
                'B/I/O/U는 고정 RGB 처리 결과. 좌표 표시에는 원래 예측을 쓰며, U는 정밀 KP로 채택하지 않음.',
                '초록 틴트 = 저장 실제 SAM 마스크. 초록 표본/선 = 실제 RGB 인라이어; 새 마스크·상태 추론 없음.',
                '자홍 + = 보관된 옛 BI점. 초록 = 실제 선. 주황 □ = 법선 투영 표시, 완전한 2D KP가 아님.',
                'O = 경험 영역, U·접선 = CAL 최대오차에서 빌린 범위. 녹색 폭은 기존 공분산 근사, 확률보증 아님. 회색 외곽은 사진 밖.',
                '파랑 와이어/◇ = 해당 모델의 최종 PnP투영. S는 보라 모델투영만. W/D는 OLD 영상 기반 선택을 고정, R/t는 d에서 연속 최적화.',
                '주황 = OLD, 파랑 = 최종, 흰 점선 = 사후 재구성한 물리 참조. 참조는 런타임 선택·선 fitting에 쓰이지 않음.')
            path=assets/f'case{j}_stage{stage+1}.png';_save(fig,path,captions[stage]);case['stages'].append(dict(title=title,asset=str(path.relative_to(OUTPUT)),caption=captions[stage]))
        cases.append(case);pictures.append((image,qold,q,reference,m,fid))
    fig,axes=plt.subplots(2,len(pictures),figsize=(5.1*len(pictures),7.9),squeeze=False)
    for j,(image,qold,q,reference,m,fid) in enumerate(pictures):
        for i,(points,color,name,arm) in enumerate(((qold,ORANGE,'OLD',P.ARMS[0]),(q,BLUE,'최종 e',P.ARMS[2]))):
            ax=axes[i,j];ax.imshow(image);_wire(ax,reference,'white',ls=':',width=2.2);_wire(ax,points,color,width=1.8);ax.set_axis_off();ax.set_aspect('equal')
            error=m[arm];ax.set_title(f'{name}: {error["translation_cm"]:.3f}cm / {error["rotation_deg"]:.3f}°',fontsize=12)
            if i==0:ax.text(.5,1.13,fid,transform=ax.transAxes,ha='center',fontsize=9)
    fig.suptitle('실제 사진의 이전/최종 자세 비교 · 흰 점선은 사후 물리 참조\nTEST49 모두 5cm/5° 이내 · 좌표·이미지·포즈는 봉인된 값 그대로',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(OUTPUT/'overview_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    summary=score['summary'][P.ARMS[2]]
    C.write(OUTPUT/'PROCESS_DATA.json',dict(cases=cases,all49_summary=summary,worst_id=worst,bindings=bindings,
        actual_RGB_images=True,no_AI_generated_image=True,new_inference_or_pose_fit=0,GT_only_postsealed_reference=True,
        TEST_reused_iterative=True,reference_independently_measured=False,range_and_normal_noise_not_newly_calibrated=True))
    body=[]
    for j,case in enumerate(cases):
        values=case['postseal_metrics'][P.ARMS[2]];options=''.join(f'<button onclick="stage({j},{k})">{html.escape(s["title"][:14])}</button>' for k,s in enumerate(case['stages']))
        panels=''.join(f'<div class="step" id="c{j}s{k}" style="display:{"block" if k==0 else "none"}"><h2>{html.escape(s["title"])}</h2><img src="{s["asset"]}" alt="실제 사진 처리단계"><p>{html.escape(s["caption"])}</p></div>' for k,s in enumerate(case['stages']))
        body.append(f'<section class="case" id="case{j}" style="display:{"block" if j==0 else "none"}"><h2>{html.escape(case["id"])}</h2><p>최종 {values["translation_cm"]:.3f}cm / {values["rotation_deg"]:.3f}° · 고정 W/D: {case["frozen_family"]}</p><nav>{options}</nav>{panels}</section>')
    chooser=''.join(f'<option value="{j}">{html.escape(c["id"])}</option>' for j,c in enumerate(cases))
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실제 영상 과정 · W/D 고정 후 부분 선 PnP</title><style>body{{font-family:system-ui,sans-serif;margin:0;background:#edf2f6;color:#173044}}main{{max-width:1120px;margin:auto;padding:22px}}h1{{font-size:25px}}h2{{font-size:20px}}img{{max-width:100%;height:auto;display:block;background:white}}select{{width:100%;padding:12px;font-size:16px}}nav{{display:flex;gap:6px;flex-wrap:wrap;margin:15px 0}}button{{padding:10px;border:1px solid #c7d1d9;background:white;border-radius:7px;cursor:pointer}}.case,.note{{background:white;padding:18px;margin:18px 0;border-radius:12px}}p{{line-height:1.6}}.step p{{background:#f1f5f8;padding:10px}}a{{color:#17658c}}@media(max-width:500px){{main{{padding:10px}}.case{{padding:10px}}h2{{font-size:17px}}}}</style><main><h1>실제 영상의 처리 과정</h1><p>같은 TEST49 전체: 평균 {summary['translation_cm']['mean']:.3f}cm, 최대 {summary['translation_cm']['max']:.3f}cm / 최대 {summary['rotation_deg']['max']:.3f}°. 실패·모호성·이전 native 번호 비호환 사례를 제외하지 않았습니다.</p><img src="overview_actual.png" alt="실제 세 사진의 이전·최종 비교"><label>사례 선택<select onchange="choose(this.value)">{chooser}</select></label>{''.join(body)}<div class="note"><b>표시 구분</b><p>예측 ×, 보관 BI +, 실제 선·인라이어 초록, 부분 법선표시 주황 □, 최종 모델투영 파랑 ◇, S 모델투영 보라 △. 선 표시는 완전한 KP가 아닙니다. U는 범위만 사용하고 U/S 코너 영상 조회를 추가하지 않습니다.</p><p>TEST는 반복 확인한 자료이며, 새 독립 검증셋 성능을 뜻하지 않습니다. 참조 자세는 기존 자료에서 재구성한 값이며 독립 실측은 아닙니다. 공분산·접선/O/U 영역은 조건부 근사로서 새 확률보증이 없습니다. 실제 이미지를 봉인된 값과 함께 그렸고 새 추론·학습·선/포즈 fitting은 하지 않았습니다.</p><a href="PROCESS_DATA.json">값·출처 확인 JSON</a></div></main><script>function choose(k){{document.querySelectorAll('.case').forEach((e,i)=>e.style.display=i==k?'block':'none')}}function stage(c,k){{document.querySelectorAll('#case'+c+' .step').forEach((e,i)=>e.style.display=i==k?'block':'none')}}</script></html>'''
    (OUTPUT/'index.html').write_text(page)
    for field in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[field])
    output_hashes={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()}
    C.write(OUTPUT/'RENDER_SEAL.json',dict(source_sha256=C.sha(__file__),input_sha256=bindings,output_sha256=output_hashes,
        case_count=len(cases),steps_per_case=8,predicted_geometry_not_GT_reordered=True,S_raw_points_hidden=True,
        GT_only_final_panel=True,new_mask_or_state_or_line_or_pose_fit=0,protected6_unchanged=True))
    print('ACTUAL_PROCESS_REPORT_READY',OUTPUT,'cases',ids,flush=True)


if __name__=='__main__':render()
