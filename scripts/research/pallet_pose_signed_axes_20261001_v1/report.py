"""Export the rejected signed-axis optimization attempt and input illustrations.

No fitting, candidate selection, PnP or metric calculation. The first fit was
rejected before source/real evaluation. Only historical operational R0 input
illustrations are displayed; no new learned performance is implied.
"""
from . import common as C
import csv
import io
import itertools
import json
import numpy as np

SPECS=[('translation_cm','median','T median (cm)'),('rotation_deg','median','R median (deg)'),
       ('translation_cm','P90','T P90 (cm)'),('rotation_deg','P90','R P90 (deg)')]


def publish(path,value):
    payload=value if isinstance(value,bytes) else (value if isinstance(value,str) else
        json.dumps(C.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():assert path.read_bytes()==payload,('EXISTING_PUBLICATION_DIFFERS',str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(payload)


def csv_export(name,rows):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);publish(C.DOC/name,out.getvalue());return len(rows)


def finite_text(value,digits=6):
    return 'inf (failure retained)' if value is None or np.isposinf(value) else f'{value:.{digits}f}'


def metric_row(model,summary):
    full=summary['full_population']
    parts=[finite_text(full[axis][quantile]) for axis,quantile,_ in SPECS]
    return '| '+model+' | '+' | '.join(parts)+f" | {summary['failed_pose']} |"


def plots(calls,iterations,rejected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=C.DOC/'figures';folder.mkdir(parents=True,exist_ok=True)
    artifacts=[];lam=rejected['certificate']['lambda_l2'];limit=rejected['certificate']['max_gap_upper_bound']
    x=np.asarray([r['call'] for r in calls]);j=np.asarray([r['objective'] for r in calls]);h=np.asarray([r['Huber'] for r in calls])
    gap=np.asarray([r['gradient_l2']**2/(2*lam) for r in calls])
    values=dict(objective_calls=x.tolist(),objective=j.tolist(),Huber=h.tolist(),gradient_gap_upper_bound=gap.tolist(),certificate_limit=limit,
        iteration=[r['iteration'] for r in iterations],iteration_objective_calls=[r['objective_calls'] for r in iterations],
        iteration_objective=[r['objective'] for r in iterations],iteration_gradient_gap_upper_bound=[r['gradient_l2']**2/(2*lam) for r in iterations])
    def finish(fig,name):
        path=folder/name;fig.savefig(path,dpi=145);plt.close(fig);artifacts.append(C.bind(path))
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    axes[0].plot(x,j,label='J = Huber + ridge',color='#0d9488');axes[0].plot(x,h,label='Huber data term',color='#d97706',linestyle='--')
    axes[0].set_title('All objective calls, including line-search trials');axes[0].legend();axes[0].set_ylabel('Mean loss / all2598 TRAIN frames')
    axes[1].plot(x,gap,color='#6366f1',label='||gradient||² / (2 lambda)');axes[1].axhline(limit,color='#dc2626',linestyle='--',label='Required upper bound1e-6')
    axes[1].set_yscale('log');axes[1].set_title(f'Final bound {gap[-1]:.9g} > {limit:g}');axes[1].legend(fontsize=8);axes[1].set_ylabel('Strong-convex objective-gap upper bound (log scale)')
    for ax in axes:ax.set_xlabel('Objective evaluations');ax.grid(alpha=.2)
    fig.suptitle(f'R0_ONLY rejected | {len(calls)} calls, {len(iterations)} iterations\nThis is optimization evidence. Source/real T/R performance was not measured.')
    finish(fig,'rejected_objective_and_gap.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    ix=np.asarray(values['iteration']);ij=np.asarray(values['iteration_objective']);ig=np.asarray(values['iteration_gradient_gap_upper_bound'])
    axes[0].plot(ix,ij,color='#0d9488');axes[0].scatter(ix[-1],ij[-1],color='#dc2626',s=30);axes[0].set_ylabel('Huber + ridge objective')
    axes[0].set_title(f'Every accepted optimizer iteration, including {len(iterations)}')
    axes[1].plot(ix,ig,color='#6366f1');axes[1].axhline(limit,color='#dc2626',linestyle='--',label='Required upper bound1e-6')
    axes[1].scatter(ix[-1],ig[-1],color='#dc2626',s=30);axes[1].set_yscale('log');axes[1].set_ylabel('Gradient-gap upper bound (log scale)');axes[1].legend()
    axes[1].set_title('No best-iteration selection, continuation or restart')
    for ax in axes:ax.set_xlabel('Optimizer iterations');ax.grid(alpha=.2)
    fig.suptitle('Predeclared iteration budget exhausted; optimizer success=false\nA small objective decrease does not override the locked certificate.')
    finish(fig,'rejected_full_iteration_history.png')
    return artifacts,values


def gallery():
    """Use historical R0 panels only; do not open any new real result/cache."""
    import matplotlib.pyplot as plt
    from PIL import Image
    prior_path=C.RBF_DOC/'REPORT_DATA.json';prior=C.read(prior_path)
    selection_path=C.ANCHOR_DOC/'REPORT_DATA.json';old_selection=C.read(selection_path)
    selected=[r['id'] for r in old_selection['illustrations']]
    assert len(selected)==len(set(selected))==6
    prior_rows={r['id']:r for r in prior['illustrations']}
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(signs[i]!=signs[j])==1]
    details,artifacts=[],[]
    for page in range(3):
        fig,axes=plt.subplots(2,1,figsize=(11,14),squeeze=False)
        fig.subplots_adjust(top=.84,bottom=.02,left=.04,right=.96,hspace=.45)
        for rowidx,fid in enumerate(selected[2*page:2*page+2]):
            old=prior_rows[fid];saved=next(p for p in old['panels'] if p['model']=='R0');pose=saved['pose']
            C.verify(old['image'])
            with Image.open(C.ROOT/old['image']['path']) as im:rgb=np.asarray(im.convert('RGB'))
            assert list(rgb.shape[:2])==old['image_hw'] and pose['available']
            corners=signs*np.asarray(pose['cf_extents'])/2
            xyz=corners@np.asarray(pose['R_cf']).T+np.asarray(pose['centroid'])
            camera=xyz@np.asarray(old['K']).T
            assert np.isfinite(camera).all() and (camera[:,2]!=0).all()
            uv=camera[:,:2]/camera[:,2:3]
            np.testing.assert_allclose(corners,saved['corners_cf'],rtol=0,atol=0)
            np.testing.assert_allclose(xyz,saved['corners_camera'],rtol=0,atol=0)
            np.testing.assert_allclose(uv,saved['projected_uv'],rtol=0,atol=0)
            ax=axes[rowidx,0];ax.imshow(rgb);drawn=[]
            for i,j in edges:
                if xyz[[i,j],2].min()>0:ax.plot(uv[[i,j],0],uv[[i,j],1],color='#ef4444',lw=2);drawn.append([i,j])
            ax.set_xlim(-.5,rgb.shape[1]-.5);ax.set_ylim(rgb.shape[0]-.5,-.5)
            dims=' x '.join(f'{100*d:g}' for d in old['dimensions_m'])
            ax.set_title(f"{old['recording']} | HISTORICAL R0 INPUT BASELINE ONLY\nDimensions: {dims} cm | {saved['hypothesis']}\nHistorical R0 T={saved['T_cm']:.2f} cm, R={saved['R_deg']:.2f} deg\nCurrent signed-axis model: NOT EVALUATED",fontsize=10)
            ax.axis('off')
            details.append(dict(id=fid,recording=old['recording'],image=old['image'],image_hw=old['image_hw'],K=old['K'],dimensions_m=old['dimensions_m'],
                corner_signs=signs.tolist(),edges=edges,current_learned_real_evaluated=False,
                display_selection='Fixed previous six IDs; largest historical R0 T per natural recording, not a representative sample.',
                panels=[dict(model='R0',parent='R0',hypothesis=saved['hypothesis'],pose=pose,corners_cf=corners,corners_camera=xyz,
                    projected_uv=uv,T_cm=saved['T_cm'],R_deg=saved['R_deg'],drawn_edges=drawn,
                    pose_source='Historical operational R0 input illustration only; not the current signed-axis model or a new performance calculation.')]))
        fig.suptitle('Historical operational R0 input examples ONLY\nCurrent signed-axis model stopped at training convergence\nNo current learned real routes or T/R evaluation\nSingle RGB + pallet dimensions + existing calibrated K',fontsize=13,y=.985)
        path=C.DOC/'figures'/f'baseline_input_rgb_dimensions_{page+1}.jpg';fig.savefig(path,dpi=115);plt.close(fig);artifacts.append(C.bind(path))
    assert any(np.array_equal(np.asarray(r['dimensions_m']),np.asarray([1.1,.11,1.3])) for r in details)
    selection=dict(complete=True,status='HISTORICAL_R0_INPUT_ILLUSTRATIONS_ONLY',source_selection_report=C.bind(selection_path),historical_baseline_report=C.bind(prior_path),
        current_learned_real_evaluated=False,not_run=C.bind(C.DOC/'EVALUATION_NOT_RUN.json'),actual_RGB_images=6,panels=6,illustrations=details,
        new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0)
    publish(C.DOC/'GALLERY_SELECTION.json',selection)
    return artifacts,details


def main():
    protocol=C.protocol('TRAIN_PROTOCOL')
    reject_path=C.DOC/'REJECTED_R0_ONLY.json';rejected=C.read(reject_path);cert=rejected['certificate']
    prefit=C.read(C.DOC/'PREFIT_REVIEW.json')
    review=C.read(C.DOC/'REJECTED_FIT_VERIFICATION.json')
    absence=C.read(C.DOC/'EVALUATION_NOT_RUN.json')
    assert rejected['complete'] and not rejected['accepted'] and rejected['model']=='R0_ONLY'
    assert rejected['protocol']==C.bind(C.DOC/'TRAIN_PROTOCOL.json')
    assert not cert['PASS'] and not cert['optimizer_success']
    assert cert['gradient_l2_squared_over_2lambda']>cert['max_gap_upper_bound']
    assert cert['iterations']==protocol['solver']['maxiter'] and cert['objective_calls']<=protocol['solver']['maxfun']
    assert prefit['complete'] and prefit['PASS']
    assert review['complete'] and review['verification_PASS'] and not review['optimizer_certificate_PASS']
    assert absence['complete'] and absence['PASS'] and absence['status']=='TRAIN_CONVERGENCE_FAILED'
    for path in absence['absent_artifacts']:assert not (C.ROOT/path).exists(),path
    assert not (C.DOC/'TRAINING_COMPLETE.json').exists()
    assert list(sorted(p.name for p in (C.RAW/'fits').iterdir()))==['R0_ONLY']
    for model in C.MODEL_NAMES:
        assert not (C.DOC/f'FIT_{model}.json').exists() and not (C.RAW/'fits'/model/'final.json').exists()
    for key in ('START','trace'):C.verify(rejected[key])
    trace=[json.loads(s) for s in (C.ROOT/rejected['trace']['path']).read_text().splitlines()]
    calls=[dict(model='R0_ONLY',**r) for r in trace if r['event']=='objective']
    iterations=[dict(model='R0_ONLY',**r) for r in trace if r['event']=='iteration']
    assert [r['call'] for r in calls]==list(range(1,cert['objective_calls']+1))
    assert [r['iteration'] for r in iterations]==list(range(1,cert['iterations']+1))
    assert len(trace)==len(calls)+len(iterations)
    assert calls[-1]['weight_sha']==iterations[-1]['weight_sha']==rejected['final_weight_sha']
    assert calls[-1]['objective']==iterations[-1]['objective']==cert['objective_value']
    assert calls[-1]['Huber']==iterations[-1]['Huber']==cert['Huber']
    assert calls[-1]['gradient_l2']==iterations[-1]['gradient_l2']==cert['gradient_l2']
    weight=np.asarray(rejected['final_weight'],np.float64)
    assert weight.shape==(253,2) and np.isfinite(weight).all()
    export=C.DOC/'rejected_optimizer_state'/'R0_ONLY.json';publish(export,reject_path.read_bytes())
    publish(export.parent/'README.md','# 학습 수렴 거부 상태 — 운영 모델 아님\n\nR0_ONLY의 마지막253×2 파라미터506개를 포함한 원본 REJECTED 영수증의 byte 동일 사본입니다. optimizer가 반복 한도에 도달했고 수렴 인증이 실패했습니다. 승인된 final checkpoint가 아니며 source·실사 추론이나 성능 평가에 사용하지 않았습니다.\n\n[거부 기록](../REJECTED_R0_ONLY.json), [독립 검산](../REJECTED_FIT_VERIFICATION_KO.md), [전체 보고서](../REPORT_KO.md)\n')
    csv_counts={name:csv_export(name,rows) for name,rows in [('TRAINING_OBJECTIVE_LOG.csv',calls),('TRAINING_ITERATION_LOG.csv',iterations)]}
    artifacts,figure_values=plots(calls,iterations,rejected)
    galleries,illustrations=gallery();artifacts.extend(galleries)
    data=dict(complete=True,status='FIT_REJECTED',actual_training_attempts=1,rejected_attempts=1,certified_models=0,
        unattempted_models=['UNION_s1','UNION_s2','UNION_s3'],source_VAL_evaluated=False,source_VAL_routes=0,
        learned_real_evaluated=False,learned_real_routes=0,current_TR_effect_measured=False,
        stable_joint_improvement_achieved=False,goal_complete=False,method_success=False,
        code=C.bind(C.HERE/'report.py'),train_protocol=C.bind(C.DOC/'TRAIN_PROTOCOL.json'),
        rejected_fit=C.bind(reject_path),rejected_fit_verification=C.bind(C.DOC/'REJECTED_FIT_VERIFICATION.json'),
        evaluation_not_run=C.bind(C.DOC/'EVALUATION_NOT_RUN.json'),prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'),
        design=C.bind(C.DOC/'DESIGN_KO.md'),historical_baseline_report=C.bind(C.RBF_DOC/'REPORT_DATA.json'),
        basis=C.bind(C.RBF_DOC/'RBF_BASIS.json'),exports={},
        rejected_exports={'R0_ONLY':dict(local=C.bind(reject_path),published=C.bind(export),deployable=False,parameters=506)},
        csv_rows=csv_counts,objective_calls=len(calls),iterations=len(iterations),
        certificate=cert,solver=rejected['solver'],artifacts=artifacts,figure_values=figure_values,
        actual_RGB_images=6,illustrations=illustrations,gallery=C.bind(C.DOC/'GALLERY_SELECTION.json'),
        baseline_panels=6,current_learned_real_panels=0,new_metric_calls=0,new_image_forwards=0,new_PnP_solves=0,
        numerical_scope='Only saved trace arithmetic and frozen R0 pose projection; no new training, routing or T/R metrics.',
        export_scope='All objective1108 and iteration1000 rows, including metadata and final rejected weight identity; rejected state is not an accepted checkpoint.')
    ratio=cert['gradient_l2_squared_over_2lambda']/cert['max_gap_upper_bound']
    j0=calls[0]['objective'];j1=calls[-1]['objective'];drop=100*(j0-j1)/j0
    fmt=lambda value:f'{value:.12g}'
    lines=['# Signed T/R 회귀: 학습 수렴 단계에서 중단','',
        '**이번 방법의 T/R 개선 효과는 측정하지 않았습니다.** 첫 R0_ONLY 학습이 사전 수렴 조건을 충족하지 못해 거부됐고, 나머지 UNION 세 모델의 학습·source VAL·실사 평가를 실행하지 않았습니다. 안정적인 T·R 동시 개선 목표는 아직 미달성입니다.','',
        f"실제 수행은 **학습 시도1회 / 거부1회 / 인증 모델0개**, objective 호출{len(calls)}회·optimizer 반복{len(iterations)}회입니다. 이는 구현·synthetic 검산 실패도, 측정된 T/R 성능 악화도 아닌 **고정 예산 내 수렴 인증 실패**입니다.",'',
        '운영 입력 계약은 **단일 RGB 이미지 + 팔레트 치수 + 기존 카메라 보정 K**입니다. 아래 실제 사진은 이 입력을 보여주는 이전 R0 예시이며, 현재 회귀 모델의 실사 결과가 아닙니다.','',
        '## 실행된 단계와 실행되지 않은 단계','',
        '| 단계 | 실제 상태 |','|---|---|',
        '| 입력·수학 사전 독립 검산 | PASS |','| R0_ONLY 학습 시도 | 반복 한도에서 종료, 인증 FAIL |',
        '| UNION_s1 / s2 / s3 학습 | 모두 미실행 |','| 승인된 final checkpoint | 0개 |',
        '| source VAL 선택·45개 조건 | 미실행; PASS/FAIL 판정 없음 |','| 실사 learned 선택·5범주 안정성 | 미실행; PASS/FAIL 판정 없음 |','',
        '[미실행 영수증](EVALUATION_NOT_RUN_KO.md)은 승인된 FIT·TRAINING_COMPLETE·source/real 산출물의 부재를 확인합니다. 이 문서의 검산 PASS는 미실행 사실 확인을 뜻하며 방법의 성능 통과를 뜻하지 않습니다.','',
        '## 이번에 시도한 변경','',
        '직전 [고정 RBF253 선택기](../pallet_pose_anchor_rbf_20261001_v1/REPORT_KO.md)의 입력 특징·raw94 정규화·context189·RBF64 센터와 폭·후보 pool·기존 TRAIN2598행·물리적 T/R 참조와 scale을 유지했습니다. 선택 후보 하나를 맞히는 margin CE 대신, R0 운영 anchor에 대한 두 축 변화량을 signed-log1p target으로 회귀하는 Huber 목적함수를 사용했습니다. 변경 동기와 음성 선행은 [설계 문서](DESIGN_KO.md)에 고정했습니다.','',
        '```text','x_c = phi253(candidate_c) − phi253(R0 GEO anchor)',
        'e_c = [(T_c − T_anchor)/sT, (R_c − R_anchor)/sR]',
        'y_c = sign(e_c) * log1p(abs(e_c))',
        'prediction_c = x_c @ W; W shape = (253, 2); bias = 0',
        'J = mean_all2598[mean_valid_candidates_and_2axes Huber(prediction_c − y_c, delta=1)]',
        '    + (lambda/2) * ||W||_F²',
        'sT = 2.4636887551191258 cm; sR = 1.113474019956766 deg; lambda = 1e-4',
        'planned runtime: choose whole pose with minimum max(predicted T change, predicted R change)','```','',
        '기존 유효 후보를 유지하며 모든 후보가 실패한1행은 손실0으로 전체2598행 분모에 남습니다. 유효·유한 쌍에만 차분을 계산해 inf−inf를 피합니다. anchor의 입력 차분과 target은 정확히0입니다. 추론에는 실제 T/R 참조 오차·학습 margin·정답으로 만든 safe mask를 넣지 않도록 구현했습니다. 이 추론 규칙의 실제 데이터 성능은 이번 중단으로 평가하지 않았습니다.','',
        '예측 기준에서 anchor 점수가0인 것은 실제 pose의 두 축이 개선된다는 보장이 아닙니다. 동률에는 원래 R0→가설명 순서를 유지하므로 운영 anchor가 두 번째 가설이면 다른 R0 가설의 동률 선택도 허용됩니다. 새 anchor 우선 규칙이나 실제 정답 기반 routing은 추가하지 않았습니다.','',
        '## 실제 중단 수치','',
        '| 항목 | 기록 값 | 요구 조건 / 해석 |','|---|---:|---|',
        f"| optimizer 반복 | {cert['iterations']} | 최대{protocol['solver']['maxiter']}회 도달 |",
        f"| objective 호출 | {cert['objective_calls']} | 최대{protocol['solver']['maxfun']}회 이내; 반복 한도가 먼저 종료 |",
        f"| optimizer success | {cert['optimizer_success']} | True 필요 |",
        f"| 최종 J = Huber + ridge | {fmt(cert['objective_value'])} | 실사 오차 단위 아님 |",
        f"| Huber data term | {fmt(cert['Huber'])} | 유효 후보×2축 평균 후 전체2598행 평균 |",
        f"| L2 penalty | {fmt(cert['L2_penalty'])} | 모든506개 계수에 적용 |",
        f"| gradient L2 | {fmt(cert['gradient_l2'])} | 마지막 파라미터의 기울기 |",
        f"| 목적함수 gap 상한 | {fmt(cert['gradient_l2_squared_over_2lambda'])} | {cert['max_gap_upper_bound']:g} 이하 필요 |",
        f"| 요구 상한 대비 | {ratio:.6f}배 | 수렴 인증 FAIL |",'',
        f"solver 메시지는 `{rejected['solver']['message']}`입니다. 첫 J={j0:.12g}에서 마지막 J={j1:.12g}로 {drop:.3f}% 감소했지만, 감소 사실만으로 사전 인증 조건을 대신하지 않습니다. 마지막 상태를 채택하지 않았고, 중간의 가장 유리한 반복·추가 학습·재시작도 선택하지 않았습니다.",'',
        '`J(W)−min J ≤ ||gradient J(W)||²/(2λ)`는 강볼록 목적함수의 **최적값과의 차이에 대한 상한**입니다. 상한이1e−6보다 크다는 것은 필요한 보증을 얻지 못했다는 뜻이며, 실제 최적값과의 차이가 그 수치만큼 크다고 확정하는 뜻은 아닙니다. 별도로 optimizer 성공 조건도 실패했습니다.','',
        '![전체 objective 호출과 수렴 상한](figures/rejected_objective_and_gap.png)','',
        'objective 호출에는 line search의 trial도 포함하므로 이 곡선의 각 점이 승인된 optimizer 반복은 아닙니다. 아래에는 승인된1000개 optimizer 반복을 전부 보존했고 마지막 반복을 강조했습니다.','',
        '![마지막 반복까지 포함한 전체 이력](figures/rejected_full_iteration_history.png)','',
        '**이전 margin CE와 이번 Huber는 서로 다른 목적함수이므로 loss 값의 직접 우열 비교를 하지 않습니다.** 이번 거부 파라미터로 TRAIN 후보 argmin, 선택 정확도, T/R 효과, source·실사 효과도 계산하지 않았습니다. 독립 검산은 저장된 마지막 상태의 Huber·기울기·Hessian·기록 연결을 확인하는 범위입니다.','',
        '[거부 상태의 독립 수학 검산](REJECTED_FIT_VERIFICATION_KO.md)은 원래 인증 FAIL을 그대로 재현합니다. 검산 PASS를 optimizer 인증 PASS나 학습 성공으로 바꾸어 해석하지 않습니다.','',
        '| 독립 재계산 | 값 |','|---|---:|',
        f"| Huber + ridge J | {review['independent_recompute']['objective']:.15g} |",
        f"| gradient L2 | {review['independent_recompute']['gradient_l2']:.15g} |",
        f"| 목적함수 gap 상한 | {review['independent_recompute']['certified_gap_upper_bound']:.15g} |",
        f"| 506×506 Hessian 최소 고유값 | {review['Hessian']['min']:.12g} |",
        f"| Hessian 최대 고유값 | {review['Hessian']['max']:.12g} |",
        f"| Hessian condition number | {review['Hessian']['condition_number']:.9f} |",
        f"| 정확히 Huber 경계인 잔차 수 | {review['Hessian']['exact_kink_count']} |",'',
        '마지막 지점의 Hessian은 양의 정부호였고 두 축 사이 블록은0이었습니다. 큰 condition number는 이 지점의 수치적 곡률 차이를 나타내는 관측이며, 이것만으로 반복 한도 도달의 유일한 원인을 확정하지 않습니다. 더 큰 예산이나 다른 solver를 이 결과에 섞지 않았습니다.','',
        '## 실제 RGB와 치수: 기존 R0 입력 예시','',
        '**아래6장은 이전 보고서에서 이미 정한 동일한 실제 RGB입니다. 현재 signed-axis 모델의 learned 결과는 하나도 포함하지 않습니다.** 자연 촬영별 기존 R0 T 오차가 가장 컸던 예시라서 대표 표본이나 개선 근거로 볼 수 없습니다. 입력 치수에는 실제 `110 × 11 × 130 cm` 등이 포함됩니다.','',
        '빨간 선은 이전 운영 R0의 저장된 camera-facing extents·R_cf·centroid를 기존 K로 투영한 것입니다. 새 PnP나 참조 오차 계산은 하지 않았고, 표기된 T/R는 이전 R0의 기록입니다. 현재 모델의 수치로 대체하지 않았습니다.']
    for page in range(1,4):lines+=['',f'![이전 R0 입력 RGB와 치수 {page}](figures/baseline_input_rgb_dimensions_{page}.jpg)']
    lines += ['', '이미지 SHA·원본 크기·치수·K·기존 R0 pose6개·투영좌표는 [갤러리 원자료](GALLERY_SELECTION.json)에 연결했습니다. 최신 **실제로 측정된** learned 실사 결과는 [직전 RBF 보고서](../pallet_pose_anchor_rbf_20261001_v1/REPORT_KO.md)에 있으며, 그 단계에서도 안정적 T·R 공동 개선은 미달성이었습니다. 그 과거 결과를 이번 방법의 성능으로 재사용하지 않습니다.','',
        '## 공개 검증 자료와 한계','',
        '- [사전 설계](DESIGN_KO.md), [학습 프로토콜](TRAIN_PROTOCOL.json), [사전 입력·수학 독립 검산](PREFIT_REVIEW_KO.md)',
        '- [원본 거부 영수증](REJECTED_R0_ONLY.json), [독립 거부 상태 검산](REJECTED_FIT_VERIFICATION_KO.md), [평가 미실행 확인](EVALUATION_NOT_RUN_KO.md)',
        f'- [objective 전체{len(calls)}행 CSV](TRAINING_OBJECTIVE_LOG.csv), [optimizer 반복 전체{len(iterations)}행 CSV](TRAINING_ITERATION_LOG.csv)',
        '- [거부 파라미터506개 — 비운영 상태](rejected_optimizer_state/R0_ONLY.json), [파라미터의 사용 범위](rejected_optimizer_state/README.md)',
        '- [그림·데이터 SHA 연결](REPORT_DATA.json), [공개물 독립 검산](PUBLIC_REVIEW_KO.md), [공개 SHA 목록](PUBLICATION_MANIFEST.json)',
        '- [학습 코드](../../../scripts/research/pallet_pose_signed_axes_20261001_v1/convex_train.py), [미실행 source 평가 코드](../../../scripts/research/pallet_pose_signed_axes_20261001_v1/evaluate_source.py), [미실행 실사 평가 코드](../../../scripts/research/pallet_pose_signed_axes_20261001_v1/evaluate_real.py), [보고서 생성 코드](../../../scripts/research/pallet_pose_signed_axes_20261001_v1/report.py)',
        '', 'source VAL과 실사 DEV는 이전 방법들에서 반복 사용됐습니다. refiner의 기존 source 노출과 selector TRAIN/VAL/TEST 중복105/26/26건이 있어 독립 일반화 시험으로 주장하지 않습니다. 전체 교사 계보에는 기존 이미지9장의 수동 코너38개가 포함됩니다. 이번 새 실사 정답 학습은0개이며, 기존 실사 pose 참조는2D 주석·K·치수로 만든 값으로 독립 장비에서 측정한6D 정답이 아닙니다.','',
        '이번 시도는 동일한 solver 예산에서 수렴 단계가 먼저 중단됐습니다. 방법이 실사 T/R를 개선하거나 악화한다는 결론은 낼 수 없습니다. 반복 수나 tolerance를 바꾼 새 시도를 이번 결과로 합치지 않았습니다. 재현에는 공개 코드·프로토콜과 SHA로 연결된 로컬 원본 특징·참조 캐시가 필요합니다.','']
    publish(C.DOC/'REPORT_KO.md','\n'.join(lines))
    publish(C.DOC/'REPORT_DATA.json',data)
    publish(C.DOC/'README.md','# Signed T/R 회귀 — 학습 수렴 단계 중단\n\n[상세 결과·실제 RGB 입력과 치수](REPORT_KO.md)\n\nR0_ONLY1회가1000반복/1108호출에서 수렴 인증에 실패했습니다. 인증 모델0개이며 나머지 학습·source VAL·실사 평가는 미실행입니다. 이번 방법의 T/R 효과는 측정하지 않았습니다.\n\n[거부 상태 독립 검산](REJECTED_FIT_VERIFICATION_KO.md), [평가 미실행](EVALUATION_NOT_RUN_KO.md), [공개 검증](PUBLIC_REVIEW_KO.md)\n')
    print(json.dumps(dict(PASS=True,status='FIT_REJECTED',csv_rows=csv_counts,figures=len(artifacts),certified_models=0,current_TR_effect_measured=False,goal_complete=False)))


if __name__=='__main__':main()
