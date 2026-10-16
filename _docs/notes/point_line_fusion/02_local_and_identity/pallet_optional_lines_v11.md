# V11: 실제 이미지, 가림과 잘림에서 선택적 선 추가 검증

2026-09-10 완료. 사용자 요청: V10의 점 보정이 실제 영상에서도 유효한지, 기존 점 보정을 유지하는 선 추가와 DHT 공동학습이 가림·잘림에 도움이 되는지 실험.

## 구조와 통제

원래 YOLO와 V10 local refiner를 고정했다. 최종 출력은 `V10 local point + sigmoid(gate) * A * residual`이다. 추가 residual 마지막 층은 0으로 초기화하여 초기 출력과 extra-off 출력 모두 V10 local과 정확히 같다. 기존 local 보정을 선 방향으로 제한하지 않는다.

- local_extra: 추가 영상 기반 점 보정, A=I, 선 context=0.
- lines_frozen: 고정 DHT 후보와 점별 attention으로 A=sum(w*n*n^T).
- lines_joint: 동일 구조이며 DHT도 점 손실로 함께 학습.

각 3개 시드, 2,000스텝, batch16. 원본1775 합성 + SHA 선택384장의 네 가림/잘림 변형에서 검출된1451장 =3226 관측. 총9모델·18,000업데이트. 학습 약120초. 실제 학습/모델 선택/GT 수정 없음. V10 local/directional도 추가 학습 없이 모든 조건에 평가.

추가 refiner58,852개 파라미터, joint DHT143,900개 추가 학습 파라미터. 총 학습 가능 용량이 같다는 주장은 하지 않는다. joint의 실제 점 손실 DHT gradient 비영/유한성, 모든 최종 시드 체크포인트의 차이를 확인.

## 평가 계약

실제297장/9세션은 기존 HELD_OUT 목록의 재사용 DEV. `keypoint_annotations`의 직접 클릭1251점만 주평가. PnP 투영점은 제외하고 원래 camera-facing ID를 유지한다. 새 인덱스 매칭/GT pose 사용 없음. K와 실물 치수는 관측 메타데이터로만 사용한다.

합성은 기존 검출 조건으로 선택했던505장 재사용. 현재 추론은 두 도메인 모두 GT 매칭 없이 최고 confidence>=.25를 사용한다. 각 변형 이미지를 새로 YOLO 추론하여 점·상자·특징을 추출한다. 모든 모델이 정확히 같은 관측과 검출 실패를 공유한다.

중앙 세로 가림20/40%는 원본 검출 상자 폭 기준. 실제 잘림20/40%는 좌우 한쪽에서 상자의 해당 비율까지 이미지를 잘라 다시 추론하고, 카메라 주점/GT 좌표를 이동한다. 원본 클릭은 화면 밖으로 나가도 평가 대상으로 유지하며 outside 층을 따로 보고한다. 원래 reflect100 padding은 잘림 경계에도 유지한다. 정규화 분모는 원본 대각선, 누락 검출은 normalized failure1.

## 핵심 결과

아래는 실사 검출 프레임별 점 평균 오차의 3개 시드 평균(px).

| 조건 | 검출 | 원래 YOLO | V10 local | local_extra | lines_joint |
|---|---:|---:|---:|---:|---:|
| 원본 |295/297|5.817|5.534|5.506|5.478|
| 중앙가림20% |291/297|6.251|5.949|5.924|5.880|
| 중앙가림40% |261/297|7.608|7.170|7.125|7.075|
| 잘림20% |289/297|8.652|8.691|8.827|8.714|
| 잘림40% |280/297|17.172|16.789|16.519|16.704|

원본에서 V10 local은 YOLO 대비3.15~5.73% 개선, 모든 시드 같은 방향. joint의 local_extra 대비 원본/중앙가림 추가 이득은 각 시드1% 미만으로 사전 실용성 기준 미달. 같은 joint 모델에 다른 이미지의 선을 넣은 원본/중앙가림 비교는 시드별 부호가 섞이며 CI가0을 포함한다. 따라서 작은 이득을 올바른 선의 인과 효과로 확정하지 않는다.

중앙가림40%에서 실제로 가려진 수동 GT는63개, 검출된 프레임의 점은54개뿐이다. 이 점들의 joint-local_extra 오차 차이는 평균+0.218px로 오히려 악화했다. 전체 프레임의 작은 이득은 가려진 점 복구를 뜻하지 않는다.

## 추가 표적 가림 진단

중앙 가림의 숨겨진 수동 점 수가 적어, 이미 학습한 모델 그대로 예측된 상단 코너0/1/4/5 중 SHA로 하나를 골라 box diagonal20% 크기 사각 패치를 배치했다. GT를 배치에 사용하지 않았다. 이는 주실험 결과 이후 등록한 사후 진단이며 새 학습/튜닝이 아니다.

실사284개 클릭점이 가려졌고 검출된280점을 평가했다. 가려진 점 평균 오차:

- 기존 YOLO16.446px.
- V10 local17.040px (기존보다+0.594px; 세션95% CI +0.294~+0.840).
- local_extra17.037px.
- lines_joint17.045px (local_extra보다+0.008px; CI -0.004~+0.024, 시드 부호 혼재).

잘림40%의 화면 밖571점에서도 joint가 local_extra보다 평균+0.425px 악화(CI +0.110~+0.705). 화면 안 점의 작은 이득이 화면 밖 실패를 해결하지 못한다.

## 검증·한계·결론

주실험84,590개 프레임/모델/조건 행을 독립 재계산. additive 좌표 재구성 최대 차이0.000061px, A 재구성0.000000255, 지표 차이 약7e-15. GT 변환 정확 일치. 라벨 SHA와 원본 해상도, train/eval 이미지 SHA 비중복, 학습 소스 고정 확인. 추가 진단도 preserved base/additive 경로 검산 완료.

JSON만 있는 실제 주석 폴더184장의 이미지 경로를 같은 세션/프레임의 원본 RGB로 해결한 기록은 IMAGE_LOCATOR_REPAIR.json에 있다. 원래 protocol 모집단 SHA와 경로 해결 후 모집단은 같은 ID 집합임을 역재구성해 확인했다. 숫자 직렬화와 import 이름 충돌은 학습 전 수정했다. 현재 새 YOLO 추론은 V10 예전 캐시와 작은 차이가 있으며 연산 설정 차이가 가능한 원인이다. 정확한 원인은 분리하지 않았으므로 현재 동일 입력 baseline만 비교한다.

자연 가림 라벨은 실제297장 모두 unknown, 자연 잘림 표시는1장뿐이다. 인공 스트레스 결과를 자연 가림/새 현장/최종6D 개선으로 주장하지 않는다. 손 클릭의 독립 재어노테이션은 수행하지 않았다. 이 수치는 예전41.49px와 다른 모집단/지표다.

결론: 실제 원본에서 점 보정의 유효성은 재현. 선의 실용적 추가 이득과 가려진 코너 복구는 미확인. 현재 보정은 표적 가림과 일부 잘림에서 악화하므로 전 조건에 무조건 적용할 근거는 부족하다. 다음 설계는 기존 local 보정 자체의 관측 가능성에 따른 억제, 화면 밖 점의 별도 amodal 구조 추론을 고려해야 한다. Hough의 원리적 불가능성을 증명한 결과는 아니다.

## 산출물

- 코드: `scripts/research/pallet_optional_lines_v11/`.
- 결과: `data/pallet/results/pallet_optional_lines_v11/`.
- HTML: `index.html`; 원시 수치 `SUMMARY.json`, `TARGETED_SUMMARY.json`, `CORNER_DIAGNOSTICS.json`.
- 실행 순서: prepare.py → prepare.py --phase lines → train.py → evaluate.py → evaluate.py --phase score → audit.py → targeted_occlusion.py → report.py.
- Python: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`; GPU 단계는 prepare/extraction, line cache, train, inference, targeted diagnostic.
