"""Actual unknown-split image lines, intersection region and matched PnP demo."""
from pathlib import Path
import html

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import common as C
from . import statewise_evaluation_v4 as V
from .render_keypoint_process_gallery_v1 import setup_font

ROOT=V.FIX/'boundary_recovery_v1'
OUT=ROOT/'piecewise_partial_case_v3'


def main():
    if OUT.exists():raise FileExistsError('Preserve previous case report')
    diag_path=ROOT/'kp3_bottom_piecewise_v1/DIAGNOSTIC.json';diag=C.read(diag_path)
    pose_path=ROOT/'partial_pose_v4_4/TARGET_POSE_RESULTS.json';pose=C.read(pose_path)
    seal=C.read(ROOT/'partial_pose_v4_4/TARGET_POSE_SEAL.json')
    if C.sha(pose_path)!=seal['results_sha256']:raise ValueError('Pose result seal changed')
    image_path=C.ROOT/diag['bindings']['actual_image_path']
    if C.sha(image_path)!=diag['bindings']['actual_image_sha256']:raise ValueError('Actual photograph changed')
    image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    contour=np.asarray(diag['actual_contour'],float);fit=diag['selected_representative']
    intersection=np.asarray(fit['intersection_xy'],float)
    allowed=[p for p in diag['split_profile'] if p['both_original_line_guards_pass']
             and p['finite_intersection'] and p['finite_observed_support_pass']]
    candidates=np.asarray([p['intersection_xy'] for p in allowed])
    boot=diag['bootstrap']['records']
    sensitivity=np.asarray([p['one_px_normal_floor_sensitivity_xy'] for p in boot if p['successful']])
    setup_font();OUT.mkdir(parents=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    for ax in axes:
        ax.imshow(image);ax.plot(contour[:,0],contour[:,1],color='#00a777',lw=1.5,label='실제 SAM 아래 윤곽')
        for ray,color,label in ((fit['left'],'#d28317','왼쪽 관측선'),(fit['right'],'#9348b9','오른쪽 관측선')):
            n=np.asarray(ray['normal']);t=np.array([-n[1],n[0]])
            observed=np.asarray(ray['observed_points'])[np.asarray(ray['inlier_mask'],bool)]
            lo,hi=(observed@t).min(),(observed@t).max();s=float(intersection@t)
            segment=-ray['offset']*n+np.asarray([min(lo,s),max(hi,s)])[:,None]*t
            ax.plot(segment[:,0],segment[:,1],color=color,lw=2.5,label=label)
        ax.scatter(*intersection,color='#e07700',marker='x',s=100,linewidths=2,label='두 영상선의 교점 후보')
        ax.set_xticks([]);ax.set_yticks([])
    axes[0].set_xlim(195,510);axes[0].set_ylim(296,225)
    axes[0].set_title('좌우 직선을 따로 맞추고, 나누는 위치도 영상에서 검색',fontsize=12)
    axes[0].text(intersection[0]-58,intersection[1]+12,
        f'교점 후보 ({intersection[0]:.1f}, {intersection[1]:.1f})',fontsize=11,
        bbox={'facecolor':'white','alpha':.88,'edgecolor':'none'})
    axes[1].scatter(candidates[:,0],candidates[:,1],color='#535353',s=22,alpha=.7,label='허용 분할들의 교점')
    axes[1].scatter(sensitivity[:,0],sensitivity[:,1],color='#db4162',s=20,alpha=.5,label='표본 재추출 + 선 위치 1px 감도')
    model_projection=np.asarray(pose['delta']['native3_projection_after'])
    axes[1].scatter(*model_projection,color='#007ed9',s=80,facecolors='none',linewidths=2,
                   label='PnP 모델의 3번 점 투영 (검출 KP 아님)')
    axes[1].set_xlim(sensitivity[:,0].min()-9,sensitivity[:,0].max()+9)
    axes[1].set_ylim(sensitivity[:,1].max()+8,sensitivity[:,1].min()-15)
    axes[1].set_title('작은 선 위치 오차에도 교점의 좌우 위치가 흔들림',fontsize=12)
    axes[1].legend(loc='upper right',fontsize=8)
    fig.suptitle('표시하신 점은 KP로 사용하지 않았습니다 — 실제 좌우 경계와 교점 영역',fontsize=15)
    fig.text(.035,.035,'왼쪽=주황 직선, 오른쪽=보라 직선, 초록=실제 SAM 윤곽. 교점 후보를 실제 KP3의 정답으로 확정하지 않았습니다.\n'
        '오른쪽 붉은 점은 1px 선 오차에 대한 감도이며, 실제 위치의 보장 구간이나 보정된 KP가 아닙니다.',fontsize=10)
    fig.subplots_adjust(left=.025,right=.985,top=.86,bottom=.17,wspace=.10)
    figure=OUT/'LEFT_RIGHT_INTERSECTION_KO.png';fig.savefig(figure,dpi=165);plt.close(fig)
    delta=pose['delta'];cal=C.read(ROOT/'partial_pose_v4_4/CAL_POSTSEAL_CHECK.json')
    info=dict(representative_intersection_xy=intersection.tolist(),admissible_intersection_region=diag['admissible_intersection_region'],
        normal_1px_sensitivity=diag['bootstrap']['separate_1px_normal_floor_sensitivity'],
        marked_user_point_used=False,raw_U3_coordinate_used_as_junction=False,
        PnP_delta=delta,PnP_partial_accepted=pose['partial']['accepted'],U_state_changed=False,
        U_direct_observed_point_created=False,CAL_accepted_partial_vertices=cal['partial_vertices'],
        boundary_total_error_budget_empirically_fitted=False,native_registration_issue_resolved=False,
        new_full_cohort_pose_accuracy_comparison=False,
        bindings=dict(actual_image_sha256=diag['bindings']['actual_image_sha256'],
            diagnostic_sha256=C.sha(diag_path),pose_result_sha256=C.sha(pose_path),figure_sha256=C.sha(figure),renderer_sha256=C.sha(__file__)))
    C.write(OUT/'CASE.json',info)
    region=diag['admissible_intersection_region']['minmax_xy'];noise=diag['bootstrap']['separate_1px_normal_floor_sensitivity']['minmax_xy']
    before,after=delta['native3_projection_before'],delta['native3_projection_after']
    title='좌우 경계 직선과 교점 영역 확인'
    document=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font:16px/1.7 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#172638}}
main{{max-width:1160px;margin:24px auto;padding:0 20px 36px}}h1{{font-size:27px}}.card{{background:white;border:1px solid #d6dee6;border-radius:12px;padding:18px;margin:18px 0}}
img{{max-width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #d9e1e8;padding:10px}}
.small{{font-size:14px;color:#4b5967}}a{{color:#176aab}}@media(max-width:600px){{main{{padding:0 12px}}h1{{font-size:22px}}.card{{padding:10px}}td,th{{font-size:14px;padding:6px}}}}</style>
<main><h1>{title}</h1><p>표시하신 점은 KP의 위치로 사용하지 않았습니다. 이미 보정된 양 끝 B7·B2로 아래 윤곽 구간을 선택하고, 좌우를 나누는 위치도 실제 영상 표본에서 검색했습니다.</p>
<div class="card"><a href="{html.escape(figure.name)}"><img src="{html.escape(figure.name)}" alt="실제 하단 윤곽을 왼쪽과 오른쪽 직선으로 맞춘 결과와 교점 감도 영역"></a><p class="small">이미지를 누르면 원본 크기로 볼 수 있습니다. 주황·보라=관측 직선, 초록=SAM 윤곽, 회색=교점 후보, 빨강=선 오차 감도.</p></div>
<div class="card"><table><tr><th>계산</th><th>원본 이미지 좌표</th></tr>
<tr><td>대표 교점</td><td>({intersection[0]:.2f}, {intersection[1]:.2f})</td></tr>
<tr><td>허용 분할들의 교점</td><td>x {region[0][0]:.2f}–{region[1][0]:.2f}px</td></tr>
<tr><td>선 위치 1px 감도 포함</td><td>x {noise[0][0]:.2f}–{noise[1][0]:.2f}px</td></tr></table>
<p class="small">276개 RGB·마스크 표본, 분할 후보 204개 중 관측 품질과 구간 검사를 통과한 35개. 1px 감도 범위는 실제 위치의 보장 구간이 아닙니다.</p></div>
<p>아래 윤곽은 하나의 직선보다 좌우 두 직선으로 잘 설명됩니다. 다만 완만하고 둥근 외곽을 직선으로 근사한 교점이므로, 이것만으로 실제 KP3와의 대응을 확정하지 않습니다.</p>
<div class="card"><p>두 선을 실제 PnP에 부분 관측으로 연결했습니다. 각 선에 수직인 오차를 soft하게 추가하고, 좌우 위치는 다른 보정된 KP·파렛트 크기·기존 자세 prior와 함께 정합니다. 전체 관측 윤곽의 유한 범위·모델 가시성·화면 범위 조건도 적용합니다.</p>
<p>모델의 3번 점 투영: ({before[0]:.2f}, {before[1]:.2f}) → ({after[0]:.2f}, {after[1]:.2f}), 이동 {delta['projection_change_px']:.3f}px.</p>
<p class="small">파랑 원은 PnP 모델 점이며 영상에서 검출한 KP로 취급하지 않습니다. U는 그대로이고, 완전한 2D 관측을 만들지 않았습니다. 기존 B/I와 가중치·O/T 영역·native/Q는 유지했으며 S는 PnP 투영으로만 계산합니다. 선을 따르는 범위는 관측한 아래 윤곽 전체로 넓게 남겼습니다. 가중치 0은 기존 솔버와 정확히 같은 결과입니다.</p></div>
<p class="small">현재 데이터의 CAL 46장에서는 이 새 관측 조건에 맞는 사례가 0개였습니다. 따라서 오차 예산은 기존 CAL의 fallback-B 예산을 보수적으로 빌린 가정이며, 새 관측의 오차 모델을 학습·검증한 결과가 아닙니다. 같은 윤곽을 재사용하므로 독립 정보나 정확한 자세 확률을 보장하지 않습니다. 이 사례는 재사용 TEST이며 별도 축·코너 등록 불일치를 해결한 결과도 아닙니다.</p>
<p><a href="CASE.json">좌표·이미지 근거 JSON</a> · <a href="../partial_pose_v4_4/TARGET_POSE_RESULTS.json">PnP 비교 결과</a> · <a href="../kp3_bottom_piecewise_v1/DIAGNOSTIC.json">전체 분할·표본 재추출 결과</a></p></main></html>'''
    (OUT/'index.html').write_text(document,encoding='utf-8')
    C.write(OUT/'SEAL.json',dict(html_sha256=C.sha(OUT/'index.html'),figure_sha256=C.sha(figure),
        case_json_sha256=C.sha(OUT/'CASE.json'),renderer_sha256=C.sha(__file__),new_target_GT_score=False))


if __name__=='__main__':main()
