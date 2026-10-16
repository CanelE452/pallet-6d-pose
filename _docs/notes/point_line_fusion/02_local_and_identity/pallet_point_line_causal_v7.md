# 점·선 융합 원인 분리 추가 검증 — 완료

사용자 요청: 현재 실패 이유를 가설별로 검증하고, 해결되지 않으면 그 이유도 확인한다.

실행 코드: `scripts/research/pallet_point_line_causal_v7/`
결과: `data/pallet/results/pallet_point_line_causal_v7/index.html`, `REPORT_KO.md`

기존 v6 Hough 체크포인트 3개를 고정하고, 재사용 합성 calibration 253장에 대해
점 관측(new/GT/stock YOLO), 초기 자세(learned/SQPnP), 선(none/predicted/GT-best-mode/oracle),
반복(5/25)을 18개 조건으로 분리 검사했다. 전수 조합이 아닌 사전 명시한 원인 비교다.
원래 요청 256장 중 검출 매칭 성공 253장 조건부 진단이며 실사·새 holdout은 아니다.
Stock YOLO는 실제 새 추론을 수행했지만 공동 자세 입력의 코너 번호는 GT로 대응시켰다.
따라서 stock 교체 조건 전체는 특권 정보 진단이며 배포 가능한 개선 성능으로 해석하면 안 된다.

주요 결과 (프레임 평균 코너 오차):

- 실제 YOLO 점 + GT 코너 대응 + PnP, 선 없음: 2.956px.
- 같은 점·초기 자세 + 예측선: 11.770 / 11.665 / 13.009px.
- 같은 점·초기 자세 + 정확한 GT선: 1.976px. 개선 여지는 있으나 GT 정보다.
- 정확한 GT 점 + PnP, 선 없음: 사실상 0px; 예측선 사용: 11.074 / 10.734 / 12.157px.
- 새 점만 정답으로 교체하고 학습 초기값 유지, 선 없음: 94~99px → 0.95~1.94px.
- 기존 점에서 PnP로 초기화만 교체해도 세 seed 모두 예측선 결합 오차가 악화.
- 기존 점/초기값/선 유지, 반복만 5→25회: 한 seed 소폭 개선, 두 seed 악화.
- GT로 가장 좋은 기존 선 후보를 골라도 정확한 YOLO 점에서 10.68~11.16px.

후속 탐색: 선 가중치 0/.001/.01/.1/1 + 가중치1의 25회 반복, 같은 CPU 관측을 공유.
.1에서도 악화하고, .01/.001은 결합하지 않은 결과에 가까우며 세 seed의 일관된 개선 없음.
CPU/GPU 관측 재계산으로 full-weight seed3 평균이 약0.26px 다르므로 가중치 효과는
CPU 실험 내부에서만 비교한다. 비트 동일 재현이나 미세한 개선의 의미를 주장하지 않는다.

해석: 새 점의 정확도 손실뿐 아니라, 부정확한 선이 좋은 점을 움직이는 문제가 확인됐다.
상대 후보 mass가 절대 정확도 확률은 아니며, 모든 역할 활성/상대 신뢰도 부재는 관측과 맞는다.
역할 대응·선 위치·여러 선의 공동 일관성 중 하나만을 유일한 원인으로 확정하지 않는다.
가시성·amodal 라벨 불일치와 실사 전이는 아직 직접 검증하지 않았다.
과거 uncertainty fusion의 제한적 DEV 개선도 존재하므로 모든 융합이 실패했다고 일반화하지 않는다.

NumPy 독립 수치 감사 18,216개 프레임·조건 PASS. 최대 raw 오차 재계산 차이 3.64e-5px.
정답 점 대조군 P90<.01px, PnP 실패0, v6 코어·모델 SHA 유지. 학습/GT 수정 없음.
HTML 링크8개 검사, 사례 이미지2장 육안 확인. Chrome 창0x62003e4 active/IsViewable 확인.
디스코드에는 사용자 승인 범위인 완료 문구만 발송했고 영수증을 저장했다.

실행:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_causal_v7/run.py --device cuda:0
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_causal_v7/weight_probe.py
MPLCONFIGDIR=/tmp/pallet-causal-v7-matplotlib /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_causal_v7/report.py
```

원인 확인은 완료했지만 현재 구조의 성능 개선을 달성한 것은 아니다. 기본 YOLO는 보존한다.
