"""Build reviewable scientific figures and a report from completed saved rows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from . import common as C

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def number(value):
    return 'NA' if value is None else f'{value:.5f}'


def plots(metrics, scored, fixed, observations, source_root, output):
    folder = output / 'figures'; folder.mkdir(parents=True, exist_ok=True)
    methods = ['N3_SUBPIX', *C.METHODS]
    labels = ['Fixed N3 + SubPix', 'Boundary observations only', 'N3 + validated boundaries',
              'Same observations / no mask', 'N3 / same robust pose policy', 'N3 / same ordinary pose policy']
    data = {(r['method'], r['id']): r for r in fixed + scored}
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for axis, field, unit in zip(axes[:2], ['translation_cm', 'rotation_deg'], ['Position (cm)', 'Rotation (degrees)']):
        for i, method in enumerate(methods):
            values = [r['pose'][field] for r in data.values() if r['method'] == method and r['pose']['available']]
            axis.scatter(values, np.full(len(values), i), s=6, alpha=.25)
            if values:
                q = np.quantile(values, [.1,.5,.9]); axis.plot([q[0],q[2]], [i,i], linewidth=3)
                axis.scatter([np.mean(values)], [i], marker='D', facecolor='white', edgecolor='black', zorder=5)
        axis.set_yticks(range(len(methods)), labels if axis is axes[0] else [])
        axis.invert_yaxis(); axis.set_xscale('symlog', linthresh=1); axis.set_xlabel(unit)
        axis.grid(axis='x', alpha=.2)
    for i, method in enumerate(methods):
        record = metrics['methods'][method]; n=record['new_pose_estimated']; f=record['fallback_used']
        k=record['fixed_control_outputs']; bad=record['no_pose']
        axes[2].barh(i,n,color='#3279a8'); axes[2].barh(i,f,left=n,color='#d59c55')
        axes[2].barh(i,k,left=n+f,color='#929ba3'); axes[2].barh(i,bad,left=n+f+k,color='#b65353')
        axes[2].text(122.5,i,f'{n} new / {f} fallback / {bad} failed' if not k else f'{k} fixed outputs',ha='center',va='center',fontsize=8)
    axes[2].set_yticks(range(len(methods)),[]); axes[2].invert_yaxis(); axes[2].set_xlim(0,245)
    axes[2].set_xlabel('Every eligible frame retained: 245')
    fig.suptitle('Saved full operational pose errors: easy153 + medium92\nDiamonds: means; segments: P10-P90; all finite errors remain visible')
    fig.tight_layout(); fig.savefig(folder/'01_pose_and_status.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(14,5))
    primary=[r for r in scored if r['method']==C.PRIMARY]
    comparator={r['id']:r for r in fixed if r['method']=='N3_SUBPIX'}
    for axis,field,unit in zip(axes,['translation_cm','rotation_deg'],['cm','degrees']):
        points=[(comparator[r['id']]['pose'][field],r['pose'][field],r['new_pose_estimated']) for r in primary if r['pose']['available'] and comparator[r['id']]['pose']['available']]
        a=np.asarray(points,float)
        if len(a):
            axis.scatter(a[:,0],a[:,1],c=np.where(a[:,2]>0,'#3279a8','#d59c55'),s=16,alpha=.6)
            maximum=max(float(a[:,:2].max()),1.); axis.plot([0,maximum],[0,maximum],'k--',linewidth=1)
        axis.set_xscale('symlog',linthresh=1); axis.set_yscale('symlog',linthresh=1)
        axis.set_xlabel('Fixed N3 error ('+unit+')'); axis.set_ylabel('Primary operational error ('+unit+')'); axis.grid(alpha=.2)
    fig.suptitle('Every paired output; blue=new pose, orange=N3 fallback. Below diagonal favors refinement.')
    fig.tight_layout(); fig.savefig(folder/'02_same_frame_pairs.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    if source_root is not None:
        protocol=C.read(C.CORRECTED/'VISUAL_CASE_PROTOCOL.json')
        by_observation={r['id']:r for r in observations}
        fig,axes=plt.subplots(6,2,figsize=(15,24))
        for index,case in enumerate(protocol['cases']):
            image_path=Path(source_root)/case['image']['path']; C.bound(image_path,case['image'],'frozen illustrative RGB')
            image=np.asarray(Image.open(image_path).convert('RGB'))
            for j,method in enumerate(['N3_SUBPIX',C.PRIMARY]):
                axis=axes[index,j]; axis.imshow(image); row=data[(method,case['id'])]
                native=np.asarray(row['native_points'],float)
                if j:
                    obs=by_observation[case['id']]
                    decoded=obs.get('observation',obs.get('decoded',obs))
                    for line in decoded.get('lines',[]):
                        points=np.asarray(line['support_points']); axis.plot(points[:,0],points[:,1],'-',color='#ffd34e',linewidth=1.2)
                valid=np.isfinite(native[:8]).all(1); axis.scatter(native[:8][valid,0],native[:8][valid,1],s=24,facecolor='none',edgecolor='#28c6d2')
                if row.get('actual_pose',{}).get('available'):
                    pose=row['actual_pose']; dims=np.asarray(pose['cf_extents'])/2
                    xyz=np.array([[-dims[0],-dims[1],-dims[2]],[dims[0],-dims[1],-dims[2]],[dims[0],dims[1],-dims[2]],[-dims[0],dims[1],-dims[2]],[-dims[0],-dims[1],dims[2]],[dims[0],-dims[1],dims[2]],[dims[0],dims[1],dims[2]],[-dims[0],dims[1],dims[2]]])
                    camera=xyz@np.asarray(pose['R_cf']).T+np.asarray(pose['centroid']); uvw=camera@np.asarray(row['K']).T; projected=uvw[:,:2]/uvw[:,2,None]
                    for a,b in EDGES:axis.plot(projected[[a,b],0],projected[[a,b],1],color='#ff8738',linewidth=.9)
                for corner in row.get('reprojected_ids',[]):axis.scatter(native[corner,0],native[corner,1],marker='s',s=45,facecolor='none',edgecolor='#ff8738')
                error=row['pose']; axis.set_title(case['label']+' / '+case['id']+'\n'+method+'\nT='+number(error.get('translation_cm'))+'cm, R='+number(error.get('rotation_deg'))+'deg / '+row['output_status'],fontsize=9)
                axis.axis('off')
        fig.suptitle('Same six cases frozen before new scores. Native coordinates=cyan; support=yellow; pose projection/replaced H=orange.',fontsize=11)
        fig.tight_layout(); fig.savefig(folder/'03_frozen_real_cases.png',dpi=110,bbox_inches='tight'); plt.close(fig)


def details(metrics, diagnostics, posthoc, runtime):
    text=['', '## 오류가 발생한 단계와 현재 남은 문제', '',
        '코너의 역할을 예측하는 것, 실제 해당 변의 대응점을 고르는 것, 두 선에서 코너 좌표를 얻는 것, 그 좌표로 물리 자세를 맞추는 것은 서로 다른 검사다. 기존 role은 초기 cuboid의 투영 hull 경계/내부/계산 불가 특징이다. 실제 팔레트 경계의 소유권이나 코너 정확도를 보증하지 않는다. 아래 수치는 기존 감독 오류와 이번 실제 후단 문제를 구분한다.', '',
        '| 단계 | 확인한 문제 | 현재 상태와 근거 |', '| --- | --- | --- |',
        '| 감독 | `training.py:91–94`의 ray inf/깊이 불일치가 곧 no_match가 됨 | 별도 브랜치 오류. 실제 owned wire가 있는 source-test query2164개를 POS로 복구하고 POS/NONE/IGNORE 감독 및9000update 재학습을 이미 완료한 기존 자료를 사용 |',
        '| 역할 정의 | `model.py:49–56`의 역할은 초기 hull의 기하 분류 | 실제 경계 동일성 인증으로 해석하지 않음. P0/CAL의 unsupported를 실사 물리 부재로 바꾸지 않음 |',
        '| 코너 조립 | `learned_infer.py:31–55`는 최소2점 TLS 후 비평행 무한선 교점 | 관측 길이/간격, 합의, 외삽과 covariance 검사를 추가. CAL 경계점P95 1.82680px와 가상 교점P95 15.16357px는 서로 다름 |',
        '| PnP | 잘못된 대응도 서로 합의하면 낮은 잔차를 만들 수 있음 | 유한4점 합의·다중해 유지, 초기 치수/투영 prior와 분기 거절을 기록. 강건 솔버만으로 진짜 대응을 인증하지 못함 |',
        '| 재투영 | 새 자세가 틀리면 숨은 점도 틀린 위치로 투영됨 | H 제외/최종 R,t 교체/재fit 금지를 검산. 숨은 오차 개선과 최종 T/R 개선을 분리 |', '',
        '원래 수정 전 IMAGE_ROLE의0/319는 이전 잘못된 감독 모델의 결과다. 수정 감독으로 재학습한 기존245 결과는220새자세/25Base반환, T15.20239cm/R14.56122°였다. 이번 새 관측/초기N3prior/N3반환 경로와는 여러 처리가 달라 단일 수정의 인과 효과로15.20→10.34를 해석하지 않는다. 원래 N3→cornerSubPix 감독이나 main의 N3 학습 오류로 바꾸어 설명하지 않는다.', '',
        '## 같은 좌표·자세 정책의 절제', '',
        '아래는 같은245장의 후보−대조 차이다. 음수가 낮은 오차다. CI는 사전에 보존한10000개 세션 bootstrap draw를 재사용한95% 구간이며, 여러 절제에 대한 다중 비교 보정은 없다. DEV 반복 사용과 proxy 참조의 한계를 유지한다.', '',
        '| 후보−대조 | n | 위치 차이(cm)와95% CI | 회전 차이(°)와95% CI |',
        '| --- | ---: | --- | --- |']
    for contrast in [C.PRIMARY+'_minus_N3_SUBPIX', C.PRIMARY+'_minus_N3_BASIN_ROBUST',
                     C.PRIMARY+'_minus_N3_VALIDATED_ROLE_NO_MASK', 'N3_BASIN_ROBUST_minus_N3_BASIN_STANDARD']:
        v=metrics['contrasts'][contrast]['common_operational']; m=v['metrics']
        def delta(k):
            d=m[k];return number(d['mean_delta'])+' ['+', '.join(number(x) for x in d['CI95'])+']'
        text.append('| '+contrast+' | '+str(v['common_frames'])+' | '+delta('translation_cm')+' | '+delta('rotation_deg')+' |')
    text += ['', '이번 주경로는 고정 N3를 이기지 못했다. 경계 관측의 추가는 같은 강건 자세 정책의 N3-only 대조보다 위치 오차를 높였다. 마스크 사용은 해당 두 경로에서 위치와 회전에 같은 방향의 효과를 내지 않았다. 강건 PnP도 일반 PnP보다 일관된 우위를 보이지 않았다. 이 결과는 더 큰 가림 판단기나 추가 학습의 필요성을 증명하지 않는다.', '',
        '| 범위 | 방법 | 전체 | 새 자세 | N3 반환 | 위치(cm) | 회전(°) | ADDsym(cm) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for scope in ['easy','medium']:
        for method in ['N3_SUBPIX',C.PRIMARY]:
            r=metrics['strata'][scope]['methods'][method];v=r['metrics']['operational']
            text.append('| '+scope+' | '+method+' | '+str(r['total_frames'])+' | '+str(r['new_pose_estimated'])+' | '+str(r['fallback_used'])+' | '+' | '.join(number(v[k]['mean']) for k in ['translation_cm','rotation_deg','ADDsym_cm'])+' |')
    for contract in ['common_operational','candidate_new_pose']:
        v=metrics['contrasts'][C.PRIMARY+'_minus_N3_SUBPIX'][contract]
        text += ['', contract+': n='+str(v['common_frames'])+', 평균 차이 T='+number(v['metrics']['translation_cm']['mean_delta'])+'cm, R='+number(v['metrics']['rotation_deg']['mean_delta'])+'°. 산출 실패와 큰 오차를 삭제하지 않았다.']
    text += ['', '## 경계 교점이 기존 코너보다 정확한가', '',
        '같은 native 코너 ID, 고정 N3의 참조 phase, 같은 유효 proxy 참조로 비교한다. computed, uncertainty-admitted, hybrid 입력 선택, 실제 최종 표시를 구분한다. 입력으로 선택했지만 새 자세가 없던 경우 최종 출력은 전체 N3로 반환한다.', '',
        '| 교점 범위 | 개수 | 알려진 참조 | ≤8px | N3 평균(px) | 교점 평균(px) | N3보다 개선 | 악화 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    diag=diagnostics['strata']['combined'][C.PRIMARY]
    for key,v in diag['boundary_corner_quality'].items():
        text.append('| '+key+' | '+str(v['total_count'])+' | '+str(v['known_count'])+' | '+str(v['correct_within8px'])+' | '+number(v['N3_error_same_paired_set_px']['mean'])+' | '+number(v['candidate_error_same_paired_set_px']['mean'])+' | '+str(v['improves_vs_N3'])+' | '+str(v['harms_vs_N3'])+' |')
    text += ['', '교점423개 중186개는 N3보다 정확하고237개는 덜 정확했다. 최종 표시된395개에서도177개 개선·218개 악화였다. source query의 높은 조건부 precision, 두 경계의 수치 교차 가능성, 작은 추정 covariance가 실사 코너의 상대 정확도를 보증하지 못한다. 이는 현재 정의→관측 좌표→코너 보정 사이에 남은 검증 공백이다. 실제 물리 경계의 잘못된 소유권, appearance 전이, 선 오차 중 어느 원인이 차지한 비율인지 이번 자료로 인과적으로 확정하지 않는다.', '',
        '![검증한 경계 교점과 같은 코너의 N3 오차](figures/04_boundary_vs_N3.png)', '',
        '## 가림 오판·남은 대응·숨은 재투영', '',
        '| 알려진 마스크 | 전체 | 새 자세 | N3 반환 | T/R 모두 개선 | 둘 다 악화 | 혼합/동률 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mask in ['matching_on_known','wrong_on_known']:
        records=[v for k,v in diag['mask_pose_groups'].items() if k.startswith(mask+'__')]
        groups=diag['mask_pose_groups']
        text.append('| '+mask+' | '+str(sum(v['n'] for v in records))+' | '+str(sum(v['new_pose'] for v in records))+' | '+str(sum(v['fallback'] for v in records))+' | '+str(groups[mask+'__BOTH_TRANSLATION_ROTATION_IMPROVED']['n'])+' | '+str(groups[mask+'__BOTH_TRANSLATION_ROTATION_WORSENED']['n'])+' | '+str(groups[mask+'__MIXED_OR_EQUAL']['n'])+' |')
    primary_rows=[r for r in posthoc if r['method']==C.PRIMARY]
    bad_inlier=sum(r['new_pose_estimated'] and bool(r['pools']['accepted_new_pose_final_inlier']['incorrect_ids']) for r in primary_rows)
    fewer_four=sum(r['new_pose_estimated'] and len(r['pools']['used']['correct_ids'])<4 for r in primary_rows)
    text += ['', '初期H는 fresh N3에서 고정했다. H 재추정 집합이 바뀐'+str(diag['hidden_set_changed'])+'장도 실패로 삭제하지 않았다. 새 자세인데 알려진 틀린 최종 inlier가 있는 프레임은'+str(bad_inlier)+'장, 참조상8px 이내인 used 대응이4개 미만인 새 자세는'+str(fewer_four)+'장이다. 수치 산출과 대응 정확도는 다른 상태다. POSTHOC_ROWS에는 남은 올바른/틀린/unknown 점 ID, 입력과 accepted-new-pose inlier,2D/3D SVD 배치를 기록했다. 배치 rank는 전체 PnP/Jacobian 관측 가능성을 인증하지 않는다.', '',
        '| 코너 집합 | 대응 수 | N3 평균(px) | 최종 평균(px) | N3 P90 | 최종 P90 | ≤5px→>10px 손상 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,v in diag['corner_errors'].items():
        text.append('| '+key+' | '+str(v['comparable_N3_output'])+' | '+number(v['N3_error_px']['mean'])+' | '+number(v['output_error_same_N3_set_px']['mean'])+' | '+number(v['N3_error_px']['P90'])+' | '+number(v['output_error_same_N3_set_px']['P90'])+' | '+str(v['damage_N3_le5_to_output_gt10'])+' |')
    text += ['', '자기 가림 재투영의 오차는 줄었지만 최종 위치·회전 평균은 개선되지 않았다. 새 자세 '+str(diag['actual_hidden_reprojection_frames'])+'프레임에서 예측H를 실제 교체했다. 사람 SELF 집합과 예측H 집합은 다르다. 숨은 오차만으로 자세 개선을 주장하지 않는다. 기존245장의6860개1·2점 오판 진단과 무학습 여섯 대조는 이전 결과에 보존했으며, 이번 새 정책의 실사6860경로를 추가 실행한 것으로 말하지 않는다.', '',
        '## 실제 전체 처리시간과 실행량', '',
        '| 경로 | 측정 n | 평균(ms) | 표본분산(ms²) | 표본SD | 중앙값 | P90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in runtime['arms']:
        v=runtime['summaries'][arm]['full']
        text.append('| '+arm+' | '+str(v['n'])+' | '+' | '.join(number(v[k]) for k in ['mean','sample_variance','sample_std','median','P90'])+' |')
    text += ['', '26장/13세션에서 각 경로20 warmup+130측정, 전체600회 모두 fresh detector와 해당 보정·초기 자세·최종 강건 PnP·재투영을 실행했다. 출력은 봉인된 정확도 실행과 전부 parity PASS다. 모델 로드,RGB decode,GT 채점,parity/영수증·자원 snapshot은 시간 밖이다. 캐시 좌표 재생이나 과거 시간의 합산은 사용하지 않았다.', '',
        '초기 시간 시도는 cached baseline namespace 복구 문제로 모델 초기화 완료 전0호출/0행에서 중단했다. 실패 기록과 frozen runtime.py를 보존했고, 별도 wrapper로 한 outer legacy context를 유지해 동일600회 본문을 한 번 완료했다. 임계값·모델·solver·측정 구간은 바꾸지 않았다. transient 자원 경합은 snapshot 사이에서 놓칠 수 있다는 기존 한계를 유지한다.', '',
        '이번 추가 실행은 학습0update/RGB 생성0, source CAL128의 head8batch, 실사 fresh245에서 detector245+내부 초기화1/N3245/ROLE245 및5방법1225fit경로, 시간600에서 detector600+내부 초기화1/N3450/ROLE150이다. 원시 OpenCV generic/LM 호출량은 GEOMETRY_SEAL/RUNTIME와 BUILD_LEDGER에 실제 진입 계수로 남겼다. 기존9000update를 이번 새 학습량으로 더하지 않는다.', '',
        '공개 검산은 원행에서 평균·분산·SD·중앙값·P90·대응 평균 차이, 최종 R,t의 H 재투영, 입력 provenance, finite bank와 실제 시간 interval/총량을 재계산한다. 세션 bootstrap draw SHA는 확인하지만 공개 표준 라이브러리 검산기는 CI 수치 자체를 독립 재생하지 않는다. 그 범위와 실제 GPU/비공개GT를 재실행하지 않는 한계를 REVIEW_CHECKS에 명시한다.', '',
        '현재 상태는 감독 오류 수리 완료, 관측/코너/PnP 입력 계약 구현·실행·검산 완료, 기존 N3 대비 최종 자세 우위 실패다. API가 실행된다는 사실을 방법론의 정확도 성공으로 바꾸지 않는다. 가장 작은 남은 질문은 실제 코너에 연결된 경계 관측이 이미 정확한 N3 좌표보다 더 정확한 위치를 주는가이다. 이번 한 번의 절제는 그렇지 않은 사례가 더 많음을 확인했다. 독립 자료에서 실제 경계 소유권과 상대 좌표 정확도를 확인하기 전 추가 모델·학습량·가림 분류기의 필요성을 주장하지 않는다.', '']
    return text


def build_text(metrics, calibration, output, diagnostics, posthoc, runtime):
    delta=metrics['contrasts'][C.PRIMARY+'_minus_N3_SUBPIX']['common_operational']['metrics']
    improved=metrics['verdict']['primary_full_operational_T_and_R_improved']
    verdict='이번 고정245장에서는 평균 위치와 회전이 모두 개선됐다.' if improved else '이번 고정245장에서는 평균 위치와 회전이 모두 개선됐다는 조건을 충족하지 못했다.'
    text=['가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?', '',
          '**'+verdict+'** 기존 GEOMETRIC_PROXY 참조의 DEV 결과이며 독립 실측 물리 GT나 일반화 증거가 아니다.', '',
          '기존 감독 오류는 수정된 READY 타깃과 마지막 IMAGE_ROLE 체크포인트로 이미 수리돼 있었다. 이번에는 추가 학습 없이 경계 관측의 채택, 코너 조립, 자세 분기 처리를 고쳤다. 원래 학습과 실패 기록은 변경하지 않았다.', '',
          '기존 코너 조립은 두 개 선택점으로 선을 만들고 비평행인 두 선의 교점을 사용했다. 새 경로는 source CAL에서 정한 대응 신뢰도, 최소3개 query와 지지 간격, 선 합의, 교점의 외삽·불확실성을 검사한다. 최종 TLS 지지가 각 query radius를 만족하는지 다시 확인한다. 반경8px를 넘는 교점은 점-PnP 관측으로 쓰지 않는다.', '',
          '주경로는 유효한 경계 교점이 N3 관측에서8px 이내일 때만 해당 코너의 독립 관측으로 사용한다. 다른 관측은 고정 N3 RGB 좌표다. 자기 가림 H는 어느 좌표든 fit에서 제외한다. 새 자세가 나오면 H를 최종 재투영으로 교체하고, 재투영을 다시 fit하지 않는다. 기본 반환은 전체 N3 좌표와 초기 자세이며 새 자세로 세지 않는다.', '',
          '유한4점 합의는 두 등록 치수 가설의 가설을 보존하지만, 새 fit 채택은 초기 N3 치수와 투영 basin을 유지한다. 이 prior는 fit 관측이 아니며 초기 오류를 승계할 수 있다. 다른 치수의 낮은 잔차와 대안 해는 진단에 남고, 식별 불가능한 수치 동점은 별도 상태다. 전역 유일성은 입증하지 않는다.', '',
          '| 방법 | 전체 | 새 자세 | N3 반환 | 완전 실패 | 위치 평균(cm) | 회전 평균(°) | ADDsym 평균(cm) |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ['BASE','N3_SUBPIX',*C.METHODS]:
        row=metrics['methods'][name]; values=row['metrics']['operational']
        text.append('| '+name+' | '+str(row['total_frames'])+' | '+str(row['new_pose_estimated'])+' | '+str(row['fallback_used'])+' | '+str(row['no_pose'])+' | '+number(values['translation_cm']['mean'])+' | '+number(values['rotation_deg']['mean'])+' | '+number(values['ADDsym_cm']['mean'])+' |')
    text += ['', '주경로−N3의 같은 프레임 평균 차이는 위치 '+number(delta['translation_cm']['mean_delta'])+'cm, 회전 '+number(delta['rotation_deg']['mean_delta'])+'°다. 기본 반환을 포함한 전체 운용 결과와 새 자세 집합을 METRICS.json에서 별도로 제공한다.', '',
             '| 방법·지표 | 평균 | 표본분산 | 표본SD | 중앙값 | P90 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in ['N3_SUBPIX',C.PRIMARY]:
        for metric,values in metrics['methods'][name]['metrics']['operational'].items():
            text.append('| '+name+' / '+metric+' | '+' | '.join(number(values[k]) for k in ['mean','sample_variance','sample_std','median','P90'])+' |')
    confidence=calibration['confidence']['chosen']
    text += ['', '실사 범위는 기존 사용자 지시대로 Clean153+Moderate92=245다. Severe74는 새 실행에서 제외했으며 기존319 결과를 덮어쓰거나245 평균과 혼합하지 않았다. 동일 장면 네 합성 통제 변형은 이번에도 생성하거나 실행하지 않았다.', '',
             'source CAL128의 알려진 query에서 confidence cutoff='+number(confidence['threshold'])+', 채택 '+str(confidence['accepted'])+'개 중2px 이내 POS '+str(confidence['correct'])+'개였다. 이 precision은 IGNORE 제외 조건부 수치이며 실사·코너 precision 보증이 아니다. 같은 장면 query의 상관성과 threshold scan 때문에 Wilson 값도 독립 전이 보증으로 해석하지 않는다.', '',
             'source 교점 참조는 실제 wire가 지지하는 가상 교점이며 물리 정점 소유권의 독립 인증은 아니다. 경계 모델의 calibration 지원 밖 변은 관측 제안을 보류한다. 이를 실사에 해당 물리 변이 없다는 판단으로 바꾸지 않았다.', '',
             '![전체 운용 자세와 산출 상태](figures/01_pose_and_status.png)', '', '![같은 프레임의 실제 차이](figures/02_same_frame_pairs.png)', '', '![사전에 고정한 실제 영상12패널](figures/03_frozen_real_cases.png)', '',
             '원행 GEOMETRY_SEALED/FIXED_GEOMETRY_SEALED는 채점 전 봉인됐고 PREDICTIONS/FIXED_PREDICTIONS는 실제 채점 결과다. CALIBRATION_ROWS, query/geometry diagnostics, 단위검사, 검산, 실행량 및 fresh runtime 원행을 함께 제공한다. 공개 산술 검산은 비공개 가중치 GPU 실행과 실측 물리 GT의 독립 인증을 대신하지 않는다.', '',
             '사용 API와 실행 명령은 README.md/REPRODUCE.md에 있다. main·원본 입력·기존 결과·사용자 수정은 보호 검산 대상이다. 성능을 보고 임계값·seed·학습량을 바꾸는 반복은 수행하지 않는다.', '']
    text += details(metrics,diagnostics,posthoc,runtime)
    destination=output/'RESULT_KO.md'
    with destination.open('x',encoding='utf-8') as stream:stream.write('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(C.PRIVATE)); parser.add_argument('--output',default=str(C.DOC))
    parser.add_argument('--source-root'); parser.add_argument('--charts-only',action='store_true')
    parser.add_argument('--runtime-input',default=str(C.PRIVATE/'runtime_resume'))
    args=parser.parse_args(); folder=Path(args.input); output=Path(args.output)
    metrics=C.read(output/'METRICS.json'); calibration=C.read(output/'CALIBRATION.json')
    scored=list(C.rows(folder/'PREDICTIONS.jsonl.gz')); fixed=list(C.rows(folder/'FIXED_PREDICTIONS.jsonl.gz'))
    observations=list(C.rows(folder/'OBSERVATIONS.jsonl.gz'))
    plots(metrics,scored,fixed,observations,None if args.charts_only else args.source_root,output)
    diagnostics=C.read(output/'DIAGNOSTICS.json'); posthoc=list(C.rows(output/'POSTHOC_ROWS.jsonl.gz'))
    runtime=C.read(Path(args.runtime_input)/'RUNTIME.json')
    pairs=[c for r in posthoc if r['method']==C.PRIMARY for c in r['observations']['boundary_corner_evidence']
           if c['selected_for_hybrid'] and c['N3_error_px'] is not None and c['error_px'] is not None]
    a=np.asarray([[c['N3_error_px'],c['error_px']] for c in pairs],float)
    fig,axis=plt.subplots(figsize=(7,6))
    axis.scatter(a[:,0],a[:,1],s=12,alpha=.45)
    upper=max(1.,a.max());axis.plot([0,upper],[0,upper],'k--',linewidth=1)
    axis.set_xscale('symlog',linthresh=1);axis.set_yscale('symlog',linthresh=1)
    axis.set_xlabel('Same corner / fixed N3 reference error (px)');axis.set_ylabel('Selected boundary intersection reference error (px)')
    axis.set_title('423 selected candidates: 186 better, 237 worse\nSame fixed N3 reference phase; all errors retained')
    axis.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'figures/04_boundary_vs_N3.png',dpi=140);plt.close(fig)
    build_text(metrics,calibration,output,diagnostics,posthoc,runtime)
    C.write_new(output/'FIGURE_BINDINGS.json',dict(schema='boundary_refiner_saved_figures_v2',
                figures=[C.binding(p) for p in sorted((output/'figures').glob('*.png'))],
                frozen_real_case_protocol=C.binding(C.CORRECTED/'VISUAL_CASE_PROTOCOL.json'),
                new_detector_head_PnP_ray_optimizer_calls=0))


if __name__=='__main__':
    main()
