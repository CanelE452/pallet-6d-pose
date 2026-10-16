"""Report all three V20 stages, including failed repairs and exploratory follow-ups."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v20-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context20 import *
from eval_quality20 import ARMS as ALL_ARMS, METHODS

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES=dict(zip(METHODS,('기존 YOLO','V12 점 기준','V12 전체 복원','V16 선 후보 선택','V18 점+선 직접 보정','V19 적용률 L1','V19 적용률 BCE','고정50% 적용','기존 재현 · 크기 가중','균등 손실','방향 정규화','방향 정규화 + 균등','방향 정규화 + 균등 · 영역','방향 정규화 + 균등 · 국소','4배 학습','방향 증강 · 선','방향 증강 · 국소','맞는 점 증강 · 선','맞는 점 증강 · 국소','정규화+균등 · 선 끄기','정규화+균등 · 다른 영상','방향 증강 · 선 끄기','방향 증강 · 다른 영상','맞는 점 증강 · 선 끄기','맞는 점 증강 · 다른 영상')))
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))

def avg(s,a,c,t=None):
    rs=[s['results']['real/'+c]['stock']] if a=='stock' else [s['results']['real/'+c][f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([r['strata'][t]['mean_px'] if t else r['frame_mean_px'] for r in rs]))

def main():
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    s=read(RUN/'SUMMARY_FINAL.json');aud=read(RUN/'AUDIT_FINAL.json')
    checks=['AUDIT_FINAL.json','QUALITY_VIEW_AUDIT.json','DIAGNOSIS.json','DIAGNOSIS_CF.json','DIAGNOSIS_QUALITY.json','INTEGRATION_CHECK.json','INTEGRATION_CHECK_CF.json','INTEGRATION_CHECK_QUALITY.json','REPRODUCTION.json']
    assert all(read(RUN/f)['status']=='PASS' for f in checks)
    assert not s['stable_arms']
    verdict='원인 범위를 좁히고 약한 잘림을 개선했지만, 큰 이득 보존까지 동시에 해결하지 못함'
    lines=['# V20 원인 분석과 수정 검증','',verdict,'',
      '11설정 × 3시드 = 33모델, 총84,000업데이트를 완료했다. 실제 GT 없이 합성 데이터로 학습했다. 초기2×2 대조, 방향 증강/학습량 대조, 맞는 점 증강을 순서대로 실행했다. 뒤의 두 단계는 앞 단계의 결과를 본 뒤 추가한 탐색적 실험이며 새 독립 최종 평가가 아니다.','',
      '## 결과와 해결 범위','',
      '가장 직접적인 수정은 합성 학습에 이미 맞거나 조금만 틀린 기준점도 넣는 것이었다. 잘림20%에서 방향 정규화 대조군14.279→12.862px로 개선됐고, 세 시드 모두 개선 및 실제9세션95%구간[-2.577,-0.495]px를 통과했다. 그러나 기준점12.372px보다 여전히 나쁘다. 코너 가림12.899→13.886px, 잘림40%20.803→22.606px로 악화됐다. 기존 큰 이득 유지와 이미 맞는 점 보호를 동시에 달성한 모델은0개다.','',
      '3시드 평균px, 낮을수록 좋다. 원본은 검출 프레임 평균이고 가림/잘림은 해당 대상점 평균이다. 서로 다른 집계 단위를 혼합해 전체 평균으로 해석하지 않는다. 전체 조건/P90/PCK/검출 실패 포함 지표는 HTML 선택표와 SUMMARY_FINAL.json에 보존했다.','',
      '| 방법 | 원본 | 코너 가림 | 잘림20% 밖 | 잘림40% 밖 | 잘림10% 밖 | 잘림30% 밖 |','|---|---:|---:|---:|---:|---:|---:|']
    for a in METHODS[:19]:lines.append('| '+NAMES[a]+' | '+' | '.join(f'{avg(s,a,c,t):.3f}' for c,t in FOCUS)+' |')
    lines+=['','## 가설별로 무엇을 확인했는가','',
      '| 가설 | 관측과 수정 | 결과 / 해석 |','|---|---|---|',
      '| 좌표·GT 또는 선 경로가 끊겨서 실패 | 원본 수동 GT, 이동 좌표, 고정 보정, 기울기와 선 샘플을 독립 재검산 | 검사 통과. 검사 범위 내 구현 오류로 설명되지 않음 |',
      '| 작은 보정 벡터 입력이 묻힘 | 방향/길이를 분리해 입력 크기 정규화 | 거의 같은 성능. 이것만으로 해결되지 않음 |',
      '| 이동량 가중 손실이 멈출 사례를 약하게 학습 | 크기 가중 vs 균등 손실을 정규화 유무와2×2 비교 | 멈출 사례의 손실 비중 차이는 실재하나 균등화로 안정성이 회복되지 않음 |',
      '| 학습량 부족 | 같은 원본 묶음 순서를4회 반복해8,000업데이트 | 코너 가림 개선, 큰 잘림 악화. 단순 추가 학습으로 해결되지 않음 |',
      '| 틀린 보정 방향을 충분히 못 봄 | 원래/반대/좌우90도 보정 벡터와 합성 GT 최적 적용률로 학습 | 안정성 실패, 새 선 분기의 일관된 추가 이득 미확인 |',
      '| 합성과 실제에서 기준점 품질·필요 보정량이 다름 | 점 ID 구성까지 맞춰 비교; 합성 GT 주변 맞는/작은 오류 기준점 추가 | 약한 잘림 개선. 큰 보정까지 줄어 어려운 조건 악화 |','',
      '정규화는 보정 방향을 단위벡터로, 크기를 기존 보정 반경0.15×박스대각선 대비 비율로 바꾼다. 전체 복원 차이는 기존0.25 반경으로 나눈다. 유효한 비영 벡터의 정보는 유지한다. 입력 RMS나 첫 층 기여량이 작다는 사실만으로 인과적 중요도가 낮다고 단정하지 않았다. 실제 대조 실험에서도 이 수정의 효과는 작았다.','',
      '학습 프로브에서 화면 밖 점의 멈춤 목표(alpha*≤0.05)는 약24%였지만 이동량 가중 손실 질량은 약11%였다. 하지만 균등 손실도 잘림20%를 해결하지 못했다. 이 편향은 존재하되 실패의 단독 주원인으로 확인되지 않았다.','',
      '## 가장 강한 현재 가설: 필요한 보정량의 분포 차이와 구분 실패','',
      '| 잘림20% 화면 밖 점 ·3시드 평균 | 합성 학습 | 합성 평가 | 실제 평가 |','|---|---:|---:|---:|',
      '| 기준점 오차px | 29.29 | 24.33 | 12.37 |',
      '| GT상 거의 멈춰야 하는 비율 | 27.9% | 36.0% | 59.3% |',
      '| GT 최적 평균 적용률 | 61.5% | 54.5% | 35.9% |','',
      '수동 유효점이 주로 위쪽 코너인 점을 고려해,8개 점 ID의 비중을 실제 유효 대상점 구성과 맞춘 사후 진단이다. 이 비중은 학습에 쓰지 않았다. 자세·질감·이미지 크기·특징 분포까지 맞춘 것은 아니므로 차이의 유일 원인을 증명하지 않는다. 큰 잘림40%의 GT 최적 적용률은 합성 학습74.7%/실제67.8%로 차이가 작았다. 약한 잘림에서 실제 기준점이 이미 더 잘 맞고, 덜 움직여야 할 사례가 훨씬 많다는 점이 핵심 관측이다.','',
      '4배 학습 모델은 합성 학습의 화면 밖 점에서 나쁜 보정 적용률37.8%, 좋은 보정79.8%로 분리했다. 그러나 실제 잘림20%에서는51.6%/58.1%, 잘림40%에서는69.7%/69.6%였다. 학습 손실이 줄거나 합성에서 구분을 배웠다는 사실이 실제에서의 구분으로 이어지지 않았다. 단순 최적화 정지보다 잔차 판단의 일반화 문제를 지지한다.','',
      '이 때문에 합성 데이터에 원래 기준점 외에 GT, GT+박스대각선1%/3% 거리의 기준점을 추가했다. 보정 벡터는 각 새 기준점에서 고정 V18 모델로 다시 계산했다. 원래 어려운 사례도 유지했다. 실제 GT는 학습 입력·손실·체크포인트 선택에 쓰지 않았다.','',
      '## 수정은 좋은 보정과 나쁜 보정을 구분했는가','',
      '고정 보정을100% 적용할 때 오차를0.1px 넘게 줄이면 좋은 보정,0.1px 넘게 키우면 나쁜 보정으로 사후 집계했다. 이 구분은 평가 진단이며 추론 필터가 아니다. 적용률은 정답일 확률이 아니다.','',
      '| 조건 | 모델 | 나쁜 보정 적용률 | 좋은 보정 적용률 | 전체 적용률 | GT 최적 평균 |','|---|---|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,a in enumerate(('unit_weighted','unit_weighted_quality')):
            rows=[s['diagnostics']['real/'+c][f'{a}_seed{i}'][t] for i in SEEDS]
            values=[np.mean([r[k] for r in rows]) for k in ('alpha_on_original_harmful','alpha_on_original_helpful','mean_alpha','oracle_alpha')]
            lines.append('| '+LABELS[c]+' | '+NAMES[a]+' | '+' | '.join(f'{v:.1%}' for v in values)+' |')
            ax.bar(np.array([0,1])+j*.34,values[:2],width=.31,label='수정 전' if j==0 else '맞는 점 증강')
        ax.set_xticks([.17,1.17],['나쁜 보정','좋은 보정']);ax.set_ylim(0,1);ax.set_title(LABELS[c]);ax.legend(fontsize=8)
    fig.suptitle('과도한 보정을 줄였지만 필요한 보정도 함께 감소');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'alpha.png',dpi=130);plt.close(fig)
    lines+=['','잘림20%의 평균 적용률35.1%는 GT 최적 평균35.9%와 비슷해졌지만 점별 판단이 맞다는 뜻은 아니다. 코너 가림에서는 좋은/나쁜 보정 모두 약46%, 큰 잘림에서도 약45~47%를 적용한다. 상황별로 필요한 보정을 골라 유지하지 못하고 전반적으로 보수적으로 바뀐 것이 현재 남은 직접적인 실패 양상이다.','',
      '## 손실·데이터 변경과 새 선 특징의 효과를 분리','',
      '음수는 개선. 실제9세션 paired bootstrap5,000회95%구간,3시드 평균이며 다중 비교 보정은 하지 않았다. 추가 개선 판정은 세 시드 각각1% 이상 개선과 구간 상한0 미만을 모두 요구한다.','',
      '| 비교 | 대상 | 오차 차이px | 95%구간 | 시드별 개선% | 판정 |','|---|---|---:|---|---|---|']
    pairs=[('unit_weighted_long_vs_unit_weighted','4배 학습 - 대조'),('unit_weighted_cf_vs_unit_weighted','방향 증강 - 대조'),('unit_weighted_quality_vs_unit_weighted','맞는 점 증강 - 대조'),('unit_weighted_quality_vs_unit_weighted_quality_local','맞는 점 증강: 선 - 국소')]
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,(pair,label) in enumerate(pairs):
            v=s['comparisons']['real/'+c][pair]['subgroups'][t];lo,hi=v['ci95'];mu=v['mean_px'];passed=s['evidence_gates'][pair][c]['pass_gate']
            lines.append('| '+label+' | '+LABELS[c]+f' | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '+', '.join(f'{x:+.2f}' for x in v['per_seed_gain_percent'])+' | '+('통과' if passed else '미통과')+' |')
            ax.plot([lo,hi],[j,j],color='#2563eb');ax.scatter(mu,j,color='#2563eb')
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),[p[1] for p in pairs] if ax is axes[0] else ['']*4);ax.set_ylim(3.5,-.5);ax.set_title(LABELS[c]);ax.set_xlabel('오차 차이px · 음수가 개선');ax.grid(alpha=.2)
    fig.suptitle('수정 대조와 선의 추가 효과 · 실제9세션95%구간');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'effects.png',dpi=130);plt.close(fig)
    lines+=['','맞는 점 증강의 잘림20%는 새 선 특징을 빼도12.864px로 거의 같다. 큰 잘림에서는 선 모델22.606px, 국소 모델23.098px이며 차이의 세션 구간은0보다 작지만 세 번째 시드는 오히려 소폭 악화돼 일관성 기준을 통과하지 못했다. 따라서 약한 잘림 개선을 Hough의 고유한 이득으로 설명할 수 없고, 선이 항상 쓸모없다는 결론도 내리지 않는다.','',
      '국소 대조군도 기존 부모 모델의 선 정보·학습 영향을 공유한다. 이 비교는 새 선 토큰의 추가 효과이며 완전히 선 없는 YOLO와의 비교가 아니다. 추가 선 토큰을 끄거나 샘플 영상 특징만 다른 프레임 것으로 바꾼 결과도 보존했다. 이런 분포 밖 개입은 실제 선 의미의 유용성을 증명하는 충분조건이 아니다.','',
      '## 현재 아키텍처와 학습 범위','',
      '고정 YOLO 특징 → 고정 점/DHT/전체 복원 부모 → 고정 V18 보정 벡터d. 기준점 주변3×3 특징·부모 문맥·점 ID·좌표·보정 방향/크기 + 남은 선 구간 특징 → 적용률alpha. 최종점=p0+alpha*d. 학습 가능한 적용률 모듈의 추론 구조 변경이며 YOLO/DHT 전체 end-to-end 재학습은 아니다.','',
      '선 분기는 연결 확률 상위3개 DHT 선과 원본 영상/검출박스 공통 사각형의 교차 구간에서 각각12개, 총36개 특징을 읽는다. 영역 대조군은 같은 사각형6×6개, 국소 대조군은 추가 토큰0이다. 동일 투영층과attention/적용률 헤드를 쓴다. 남은 선 위를 읽는 경로가 작동해도 그 선이 올바른 파렛트 경계인지, 점을 어느 쪽으로 움직여야 하는지가 자동 보장되지는 않는다.','',
      '여기서 남은 구간은 화면 안쪽이라는 뜻이며 실제로 보이는 파렛트 선만 선별했다는 뜻이 아니다. 샘플 유효성은 사각형 교차로 결정되므로 인공 가림 사각형 안의 선 위치도 포함한다. 고정 SHA 코너 가림 예시에서도 이 현상이 보인다. 가림 영역이나 배경을 읽는 문제가 선 근거의 한계일 수 있지만, 이번 실험은 그 요소만 분리해 고친 대조가 아니므로 주원인으로 확정하지 않는다.','',
      'alpha가0~1인 현재 구조는 원래 벡터의 방향을 뒤집거나 새 방향을 만들 수 없다. 잘못된 방향에는 멈추는 것만 가능하다. 단, GT 최적 적용률이 같은 고정 벡터로도 현재 예측보다 개선하므로 방향 제한만이 이번 실패의 원인은 아니다. GT oracle은 추론할 수 없는 진단 상한이다.','',
      '학습은 합성253원본/1462검출 관측,6조건/11595유효점이다. 각 단계에서 원본3장의 가용 변형을 묶는 동일 배치를 사용했다. 초기6설정×3시드×2000=36000업데이트; 후속4배 학습/방향 증강2설정은36000; 마지막 맞는 점 증강2설정은12000이다. AdamW3e-4, weight decay1e-4, float32, gradient clip5, 최종 체크포인트를 사용했다. 실사 최고 시드나 체크포인트를 고르지 않았다.','',
      '방향 증강은d,-d,좌우90도d의4뷰를 쓰고 GT 최적 적용률을 다시 계산한다. 맞는 점 증강은 원래 기준점과 GT 주변3뷰를 쓴다. 각 증강의2000업데이트와4배 학습8000업데이트는 동일한 원본 관측 forward 수지만 배치 크기와 optimizer 업데이트 수가 다르다. 원래2000업데이트 대조도 함께 제시해 이 차이를 드러냈다.','',
      '맞는 점 증강은 새 기준점에서 V18 보정을 다시 계산하지만 고정 전역 부모 문맥/전체 복원/선 연결 정보는 원래 관측의 것을 유지한다. 따라서 실제로 잘 맞는 부모 모델의 전체 상태를 재현한 증강은 아니다. 이 문맥 불일치와 소수 합성 원본, 부모가 이미 본 합성 잔차를 쓰는 점이 남은 학습 한계다. 이번 결과로 각각의 인과적 기여를 분리하지는 못했다.','',
      '## 독립 검증과 해석 한계','',
      f'원본 수동 클릭 source/ID/유효점 및8조건 좌표 이동을 검산했다. {aud["prediction_frame_rows"]:,}예측 프레임 행의 지표를 독립 재계산했고 최대 차이는{aud["metric_replay_max"]:.3g}이다. {aud["vector_frame_rows"]:,}검출 프레임 행에서 p0+alpha*d 재구성 최대 오차는{aud["retention_replay_max_px"]:.8f}px였다. 원래 기준점/고정 보정 벡터/검출 누락을 유지했고84,000업데이트에 비유한 손실·기울기가 없었다.','',
      '선 샘플 위치의 독립 float64 교차 검산, 선 부호 불변성, 영상 투영층까지0이 아닌 유한 기울기, 추가 토큰 제거 시 출력 변화를 확인했다. 합성 증강의 GT 기준점·반경·고정 보정 재계산도 통과했다. 통합33모델×6조건×4프레임의 좌표가 캐시 추론과0.001px 이내로 일치했다.','',
      '기존 V19 재현에서 초기 엄격한 파라미터1e-6 기준을 약1.15e-6 차이로 넘겨 중단된 이력이 있다. 원본 코드·프로토콜을 보존하고 RESUME_PROTOCOL에 수치적 기능 재현 기준을 기록했다. 전체1462학습 관측 및 평가 예측의 좌표 차이는0.001px 미만, 적용률 차이는1e-4 미만이었다. 비트 단위 동일 재현으로 주장하지 않는다.','',
      '실제297장·9세션·1251수동 유효점, 합성505장의 재사용DEV다. 실제 자연 가림 GT를 새로 확보하거나 독립 최종 holdout을 평가한 실험이 아니다. 인공 가림/잘림 조건이며 검출 실패를 포함한 지표도 별도 제공한다. 클릭 좌표를 정확히 사용했음을 확인했지만 사람이 찍은 점의 물리적 오차0을 인증한 것은 아니다.','',
      '## 다음 설계의 우선순위','',
      '현재 결과를 근거로 단순 선 모듈 추가·손실 균등화·학습 연장만 반복하는 것은 우선순위가 낮다. 먼저 부모 모델이 보지 않은 별도 합성 원본에서 자연스럽게 발생하는 잔차를 수집하고, 가림/잘림과 기준점 품질을 함께 넓혀야 한다. 부모의 점만 GT 주변으로 옮기는 대신 영상에서 부모 전체 문맥까지 일관되게 생성한 관측으로 검증하는 것이 다음 가설이다.','',
      '그 데이터에서 보정 방향/크기와 영상 근거로 점별 오차 감소를 구분할 수 있는지 먼저 확인해야 한다. 구분이 가능할 때에만 이를 점·선 보정 헤드에 연결하고, 새 세션 holdout에서 큰 오류 개선과 작은 오류 보호를 함께 검증한다. gradient surgery는 같은 학습 파라미터를 공유하는 여러 손실의 충돌을 줄이는 방법인데, 이번 단계는 동결 부모와 단일 적용률 손실이므로 지금 관측한 일반화 실패에 대한 직접 해결책으로 확인되지 않았다.','',
      '이번에 구현·검증한 수정은 일부 악화를 완화했으나 요구한 안정성은 미해결이다. 운영 모델을 교체하지 않았고33개 실험용 통합 체크포인트를 export에 보존했다. 점과 선이 같은 이미지에서 왔다는 사실만으로 융합이 불가능하다는 결론을 낼 수 없다. 현재 증거는 새 선 특징이 좋은/나쁜 보정을 실사에서 안정적으로 구분하는 데 충분히 기여하지 못한다는 범위다.']
    report='\n'.join(lines)+'\n'
    (RUN/'REPORT_KO.md').write_text(report);(ROOT/'_docs/notes/pallet_cause_v20_results.md').write_text(report)
    design='# V20 설계와 단계별 계획\n\n최종 구조: p = p0 + sigmoid(g(point context, line evidence, direction, length)) * frozen_delta.\n\n초기 원인 분리 이후 두 후속 단계는 탐색적이다. 단계 사이 평가 결과를 확인했으며 독립 최종 holdout은 없다.\n'
    for f in ['PROBE_PROTOCOL.json','PROTOCOL.json','RESUME_PROTOCOL.json','CF_PROTOCOL.json','QUALITY_PROTOCOL.json']:
        design+='\n## '+f+'\n\n```json\n'+json.dumps(read(RUN/f),ensure_ascii=False,indent=2)+'\n```\n'
    (RUN/'DESIGN_KO.md').write_text(design);(ROOT/'_docs/notes/pallet_cause_v20_design.md').write_text(design)
    body=[];inside=False
    for line in lines[1:]:
        if line.startswith('|'):
            if line.startswith('|---'):continue
            if not inside:body.append('<div class="scroll"><table>');inside=True
            body.append('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if inside:body.append('</table></div>');inside=False
        if line:body.append('<h2>'+html.escape(line[3:])+'</h2>' if line.startswith('## ') else '<p>'+html.escape(line)+'</p>')
    if inside:body.append('</table></div>')
    gallery=[]
    for stage,f in [('최종 맞는 점 증강','GALLERY_QUALITY.json'),('방향 증강','GALLERY_CF.json'),('초기2×2','GALLERY.json')]:
        for r in read(RUN/f)['rows']:gallery.append(dict(r,id=stage+' / '+r['id']))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V20 원인 분석과 수정 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}select{padding:8px;margin:6px;font-size:15px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{color:#596579}img{width:100%;height:auto}a{color:#0369a1}</style><h1>점·선 보정은 왜 이미 맞는 점을 악화시키는가?</h1><section><b>'''+verdict+'''</b><p>33모델 · 3시드 · 84,000업데이트 · 합성 학습 / 실제 재사용DEV</p><p>맞는 점 증강으로 잘림20% 14.28→12.86px. 그러나 코너 가림 12.90→13.89px, 큰 잘림 20.80→22.61px로 악화됐습니다. 약한 잘림 개선은 새 선 특징을 빼도 거의 같습니다.</p><p><b>최종점 = 기준점 + 학습 적용률 × 고정 보정 벡터</b></p><img src="effects.png"></section><section><h2>좋은 보정과 나쁜 보정에 주는 적용률</h2><img src="alpha.png"><p>평균 적용률을 낮추는 데는 성공했지만 필요한 보정만 유지하는 구분은 부족합니다. 아래 GT 기반 구분은 사후 진단이며 추론 입력이 아닙니다.</p></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p></section><section><h2>원본 GT · 선 샘플 위치 · 보정 화살표</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p>고정seed1의 SHA 예시와 사후 최대 개선/악화 예시를 구분합니다. 평균을 대표하는 사례가 아닙니다. 원래 영상 좌표의 화살표이며 증폭하지 않았습니다. 샘플 위치는 모델이 읽는 곳이고 attention 중요도의 증명이 아닙니다.</p></section><section>'''+''.join(body)
    for f in ['REPORT_KO.md','DESIGN_KO.md','SUMMARY_FINAL.json','PROBE.json','POPULATION_DIAGNOSIS.json','DIAGNOSIS_QUALITY.json','AUDIT_FINAL.json','QUALITY_VIEW_AUDIT.json','REPRODUCTION.json']:
        page+='<a href="'+f+'">'+f+'</a> · '
    previous=(KIT.parent/'pallet_detail_v17_fixed/report17.py').read_text();js=previous[previous.index('const params=new URLSearchParams'):previous.index("for(const id of ['split','seed','metric'])document.getElementById(id).onchange")]
    page+='</section><script>const data='+json.dumps(s['results'],ensure_ascii=False)+',labels='+json.dumps(LABELS,ensure_ascii=False)+',names='+json.dumps(NAMES,ensure_ascii=False)+',gallery='+json.dumps(gallery,ensure_ascii=False)+';\n'+js
    page+="""for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · 기준점 오차 '+r.base_error_px.toFixed(2)+'px / 수정 후 '+r.corrected_error_px.toFixed(2)+'px';}sel.onchange=show;show();</script></html>"""
    (RUN/'index.html').write_text(page)
    write(RUN/'CONCLUSION.json',dict(status='PARTIAL_REPAIR_FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],evidence_gates=s['evidence_gates'],models=33,updates=84000,real_gt_used_for_training=False,quality_followup_exploratory=True,adoption=False,remaining_problem='Retention generalization and hard-gain preservation are unresolved; all point-line fusion is not disproven.'))
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY_FINAL.json'),source_sha256=sha(Path(__file__)),figures=len(gallery)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
