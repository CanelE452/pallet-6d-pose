# 점 주변 영상 특징을 학습하는 DHT — 완료

사용자 요청: 점을 연결하지 않고 점 주변 영상 특징과 점 위치를 함께 이용해 선을 학습한다.

코드: `scripts/research/pallet_point_attention_v9/`
결과: `data/pallet/results/pallet_point_attention_v9/index.html`, `REPORT_KO.md`

원래 YOLO 점을 train1775장 실제 추론하고 calibration253/synth_val505는 동일 R0의
기존 실제 예측을 재사용했다. GT 점 대체 없음. YOLO 모델과 출력좌표는 고정했다.
ROI 영상특징48채널과 점 주변 Gaussian map3채널을 함께 받는 공간·채널 gate를 학습했다.
map은 점 순서에 불변이며 점쌍/연결선은 없다. gate=1+.5tanh, 초기1, 범위.5~1.5.
v8의 frozen 후보점수 조정과 달리 실제 영상특징 가중치를 학습했다.

none/points/donor 3조건×3seed, 각2000update/B8/AdamW1e-4, 총18000update.
seed내 동일143900파라미터/초기state SHA/학습순서 SHA/선 CE 손실/optimizer.
기존 v6 trunk와 DHT를 같은 선 손실로 추가 학습했다. 새 점/자세 head는 쓰지 않는다.
none은 zero map, donor는 다른 이미지 점을 현재 ROI에 정규화해 넣는다.

| synth_val505 역할 포함 선 평균오차px | seed1 | seed2 | seed3 |
|---|---:|---:|---:|
| 같은 추가 학습, 점 정보 없음 |41.516|38.810|38.138|
| 점 주변 영상 특징 학습 |40.815|38.475|38.551|
| 다른 이미지 점으로 학습 |41.494|38.786|38.173|

역할 오차1.69%/0.86% 개선,1.08% 악화. 공간 배치3.86%/2.72% 개선,2.57% 악화.
프레임별 seed평균 후 paired bootstrap 역할 차이-.207px,95%구간[-.761,+.353].
사전 모든seed5%개선 조건 미통과. 현재 구조의 안정적인 성능 개선을 입증하지 못했다.

같은 points모델의 calibration 입력교체 역할 오차:
correct41.177/46.191/46.592, zero43.347/48.983/47.898,
donor44.902/50.529/50.347. 실제 점 활용과 gate 변화는 확인했다.
입력 제거의 OOD효과 가능성이 있으므로 전체개선 판단은 별도학습none/donor에 따른다.
시각화는 feature gate채널평균의 점입력 유무 차이이며 픽셀 중요도/정답확률이 아니다.

독립산술 감사9099개 조건·프레임 PASS, 최대차이1.14e-13px.
초기 기존DHT출력 일치/point순서불변/유한gradient/실제8update smoke 검사 통과.
smoke의 calibration가중치는 버렸고 main학습에 사용하지 않았다.
18000step 유한loss/gradient, 초기값/파라미터/순서 바인딩, 원래YOLO점 보존 확인.
모든 추론파일 저장 후 GT채점, inference GT파일 접근0개.
재사용 합성 조건부 평가505/512,253/256으로 새 독립holdout/실사/최종자세 개선이 아니다.
이 선 오차를 과거41.49px 점 오차와 직접 비교하면 안 된다.

작은gate의 표현력/24x24 특징해상도/가시경계와구조선차이는 아직 미검증 가설이다.
이 구조를 기존YOLO 개선 모델로 채택하지 않는다. 추가학습/평가 기반 설정선택 없음.

HTML 참조8개 검사, 사례2장 육안확인. Chrome0x62003ea active/IsViewable 확인.
디스코드 완료문구 전달 확인. 최초 파일참조 인수를 넣은 호출은 자동검토가 거절했지만,
알림 소스를 확인한 뒤 evidence 인수를 생략해 결과파일을 읽지 않는 완료문구만 전송했다.

실행:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_attention_v9/prepare.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_attention_v9/train.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_attention_v9/evaluate.py
MPLCONFIGDIR=/tmp/pallet-attention-v9-matplotlib /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_attention_v9/report.py
```
