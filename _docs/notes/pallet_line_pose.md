# Pallet point–line architecture against the paper YOLO26n baseline

**2026-09-08 최종 완료.** 실제 학습·합성 전용 선택·실사 9개 평가·독립 감사를
완료했다. 개별 네 지표는 개선 구간을 확인했지만 P90·회전은 미확정이며,
사전등록 `overall_accuracy_improved=false`다. 아래 제안은 실행 전 가설 기록이고,
최종 관측 결과는 3절을 따른다.

## 1. 제안

[소비처] 기존 논문용 YOLO26n보다 정확한 팔레트 모델을 만들 수 있는지 결정하고, 구현·학습·동일 조건 비교의 근거를 남긴다.
[문장] YOLO의 공간 특징과 예측 점을 함께 사용하는 학습형 선 분기 및 불확실성 가중 결합이, 기존 검출 성능을 유지하면서 논문 평가 계약의 keypoint와 6D pose 정확도를 개선하는지 검증한다.

사용자 목표: “그러면 이거를 이용해서 기존 yolo26jn 논문용의 성능을 뛰어넘을수 있어? 그리고 파렛트용 아키텍쳐를 구현할수있나?” 문맥과 모델 정본상 YOLO26n을 뜻하는 것으로 진행한다. 구현뿐 아니라 실제 학습·추론·평가까지 수행한다. 검증 결과 없이 개선이나 논문 기여를 확정하지 않는다.

현재 근거: 이전 DHT 불확실성 실험은 실제 YOLO26n R0에서 DEV52의 관측8corner 평균 9.40→8.97px를 보였지만, 논문319장·9keypoint·6D 평가와 다른 계약이었다. 논문 baseline checkpoint SHA는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`. 학습은 G38+legacy P0/TEX 55,980장으로, 인터넷 모델만 학습했다는 옛 설명을 이 모델에 적용하지 않는다.

설계 방향: 기존 YOLO 검출·점 경로와 공간 특징을 보존하고, 예측 점이 만드는 측면8선을 초기값으로 삼는 학습형 residual line voting 분기를 추가한다. 각 후보 선을 따라 실제 특징을 샘플링하며, 선 분포와 점 anchor를 이용한 미분가능 결합으로 코너를 반환한다. 학습과 추론 모두 예측 점을 조건으로 쓰고 GT crop/GT pose/실사 치수로 후보를 고르지 않는다. 원래의 별도 VGG 추론 결합은 참조이며 새 아키텍처의 대체 구현으로 간주하지 않는다. 정확한 후보 범위·손실·노출 예산은 합성 데이터의 단위와 학습 경로를 확인한 뒤 실행 전에 고정한다.

판정 지표: 논문 정본 scorer의 9keypoint median/P90, MAIN 6D rotation·translation median·IoU3D median·symmetry-aware ADD AUC, 검출 AP/negative 동작 보존과 실제 추론시간. 같은 입력·padding·threshold·GT reference를 쓰고 모든 지표를 함께 보고한다. Synthetic calibration/selection만으로 승격을 주장하지 않으며, real 평가에서는 규칙을 다시 맞추지 않는다. PAPER_EVAL_ALL_POS319 및 NEG2689는 반복 사용된 DEV이며 독립 final test가 아니다.

예상 실패: 분포 전이 실패, 낮은 자세의 선 겹침, 보이지 않는 구조선과 물리적 경계 혼동, 후보 범위 밖의 초기 점, 점과 선의 상관으로 잘못된 과신, 선 offset이 6D translation을 손상함. 과거 Direct-Hough 전역 descriptor 및 PnP loss 실패를 이번 공간 선 누적 분기와 구분하되 같은 실패 원인이 나타나면 기록한다. 학습 분기가 실제 gradient를 받는지와 zero-fusion의 baseline 일치를 먼저 확인한다.

중단·계속 기준: 실제 오류·배선 문제는 수정 후 같은 목적의 검증을 계속한다. 성능이 나쁘면 실패를 숨기거나 지표를 바꾸지 않는다. 동일 가설의 실패를 세 번 누적해 무한 튜닝하지 않고 원인을 재평가한다. 최종 목표를 구현 스모크나 일부 DEV 개선으로 축소하지 않는다.

## 2. 선행 근거

- [Deep Hough Transform, ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/779_ECCV_2020_paper.php): 후보 선을 따라 영상 특징을 누적하고 선 매개변수 공간에서 학습한다.
- [HAWP, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xue_Holistically-Attracted_Wireframe_Parsing_CVPR_2020_paper.pdf): 선과 junction의 공동 검출 및 선 후보 검증을 다룬다. 여기서 제안하는 팔레트 분기가 이 모델의 재현이라는 뜻은 아니다.
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26): Pose26의 RLE 지원은 공식 문서에서 확인하되, 실제 구현은 설치된8.4.60과 동결 checkpoint로 고정한다.
- 내부 근거: `dht_uncertainty_fusion.md`, `_docs/audits/accuracy_root_cause_v1/HOUGH_IMPLEMENTATION_AUDIT.md`.

## 3. 결과

`image_joint / geometry_joint / image_line_only × seed1,2,3`의 **9개 실제
6,000-step 학습**과 합성4,020장 logits, 분리된 보정·선택·heldout 평가를 완료했다.
실사 규칙 조정 없이 각 모델을 동일 DEV319장에 실제 추론하고 원본 논문 평가기로
2D와 MAIN6D를 평가했다. 최종 checkpoint와 선택 SHA를 검증했으며 driver는 종료됐다.

주실험은 실행 전 정한 `image_joint` 그대로다. 다음 값은 원본 R0와3seed 평균이다.
구간은13세션 대응 bootstrap10,000회의 primary−R0 차이95% 구간이다.

| 지표 | R0 → image_joint | 차이 95% 구간 | 개선 확인 |
|---|---:|---:|---|
| 9점 pooled median, px ↓ | 6.6157 → 6.0409 | [-0.9715,-0.3349] | 확인 |
| 9점 P90, px ↓ | 38.6700 → 37.1011 | [-1.9000,+0.3917] | 미확정 |
| 회전 median, ° ↓ | 2.2625 → 2.1126 | [-0.4334,+0.0124] | 미확정 |
| Translation median, cm ↓ | 7.8969 → 7.5730 | [-1.3720,-0.0732] | 확인 |
| IoU3D median ↑ | 0.6032 → 0.6301 | [+0.0043,+0.0472] | 확인 |
| ADDsym AUC ↑ | 0.4285 → 0.4490 | [+0.0119,+0.0303] | 확인 |

여섯 평균은 모두 개선됐으나 P90과 회전의 세션 구간이0을 포함한다. 따라서 모든
2D/6D 조건을 요구하는 `keypoint_gain_confirmed`, `pose_gain_confirmed`,
`overall_accuracy_improved`는 false다. `complete/PASS=true`는 실행·증거의
유효성이다. 2D 개선을 곧바로6D 개선 또는 종합 우월성으로 읽지 않는다.
독립 원본 지표·seed 표본표준편차(ddof1) 재계산은 일치했고, 세션 구간 최대 차이는
1.67e-16이었다. Pose coverage319/319, box AP·score·순서·비선택 인스턴스와
중심점을 보존했다.

`geometry_joint` 대비 영상 특징을 쓰는 주실험은 frame9점 mean에서 추가
0.4544px 개선했다. 하지만 `image_line_only` 대비 주실험의 pooled median 차이는
-0.0291px, 세션95%구간 [-0.0643,+0.0851]이며 여섯 지표 모두 두 군 간 확정된
개선이 없다. Joint 코너 손실 자체가 선 손실만 사용하는 것보다 더 낫다는 근거는
부족하다. 각 군은 합성으로 고른 이동 상한이 다를 수 있어 전체 파이프라인 비교다.

사후 `DIAGNOSIS.json`에서는 visibility>0인9점의 **frame mean**을 사용했다.
매칭311프레임 중248개 개선,63개 악화였다. 정답으로 나눈 쉬움(≤5px)91장은
3.622→3.444px, 중간((5,10]px)85장은7.168→6.742px, 어려움(>10px)135장은
42.975→42.120px였다. 마지막 구간도 평균 개선은 있지만 큰 초기 오류를
복구했다고 볼 수준은 아니다. 미매칭8장은 분모에 남고 이 난이도 구간은 운영
gate가 아니다. DAY/NIGHT·plastic/wood와13세션의 frame mean은 모두 평균 개선했다.
선 GT 지원2418개 frame/role 쌍의 endpoint 거리는14.065→13.432px였다.
Height18.754→18.010px, depth9.312→8.793px로, 두 종류 모두 잔여 오류가 있다.
이 선은 amodal cuboid support이며 실제 보이는 물리 edge나 attention 정답이 아니다.

추론 비용은 별도26프레임 반복 측정의 **seed별 median 평균9.4028→17.5855ms**다.
BGR 입력→전처리→YOLO→선 분기/결합→원본2D좌표 반환을 포함하고 파일 decoding과
PnP는 제외한다. Negative2689장의 모든 후보는 baseline cache에서 복사·재비교했다.
이는 출력 보존 증거이며 새 모델로 negative 전체를 재추론한 실험은 아니다.
DEV319와13세션은 반복 사용한 모집단으로 독립 final 일반화 결론은 내리지 않는다.

완료 근거: [SUMMARY](../../data/pallet/results/pallet_line_pose_v1/SUMMARY.json),
[VERDICT](../../data/pallet/results/pallet_line_pose_v1/VERDICT.json),
[RUNTIME](../../data/pallet/results/pallet_line_pose_v1/RUNTIME.json),
[DIAGNOSIS](../../data/pallet/results/pallet_line_pose_v1/DIAGNOSIS.json),
[독립 최종 감사](../../data/pallet/results/pallet_line_pose_v1/INDEPENDENT_FINAL_METRIC_AUDIT.json),
[HTML](../../data/pallet/results/pallet_line_pose_v1/index.html),
[완료 기록](../../data/pallet/results/pallet_line_pose_v1/COMPLETION.json).
HTML 자동 열기와 Discord HTTP204 전송을 확인했다.

## 4. 실행 전 고정한 학습·판정 계약

`data/pallet/results/pallet_line_pose_v1/TRAIN_PROTOCOL.json` SHA
`35ccbb2e198774b33d3f5deda60564a8237c478c772c9bb7a63d5cae79b3bbb3`.
YOLO 파라미터·BN은 고정하고 P3/P4의 실제 선상 특징을 사용하는 19,810개 파라미터의 분기를 학습한다.
예측 측면8선마다 ±12°의13각도×bbox대각선±8%의17offset+null 후보를 사용한다.
선별32위치에서 특징을 샘플링하고 분포의 불확실성으로 점 anchor와 결합한다.
image_joint / geometry_joint / image_line_only의3구조×seed1,2,3을 각각6,000step,
batch16, AdamW lr0.001로 학습한다. geometry control은 영상 특징을 제거하고,
line-only control은 코너 손실을 제거한다. 초기 checkpoint와 추가 학습 예산·표본 순서는 대응한다.

55,980개 train record를 모두 캐시에 보존하며, top1예측과 IoU≥0.5로 매칭된 유효 GT만
손실에 사용한다. 검증4,020장은 시나리오 단위로 calibration1,004 / selection1,031 /
heldout1,985장으로 분리했다. temperature는0.5/1/2/4, 결합λ는0/0.0625/0.25/1/4,
이동상한은 없음/원본대각선1%에서 합성 데이터로만 결정한다.
미검출·미매칭·다른 GT객체도 선택 점수의 실패 분모에 포함한다.
Temperature 방식의 참고는 [Guo et al., ICML2017](https://proceedings.mlr.press/v70/guo17a.html)이며
이 분포가 물리적인 가시성이나 attention이라는 의미는 아니다.

`DECISION_PROTOCOL.json`은 main 학습·실사 새모델 추론 이전에 판정 조건을 구체화했다.
주실험은 image_joint로 고정한다. 2D의median/P90 및6D의rotation/translation/IoU3D/ADD AUC가
각각3seed평균에서 개선되고 세션 paired-bootstrap95%구간도 개선 방향을 지지해야
전체 정확도 우위를 확정한다. 평가 coverage와 검출·negative 결과는 보존해야 한다.
지표가 엇갈리면 그 차이를 보고하고, 실제 결과를 본 후 기준이나 arm을 바꾸지 않는다.

사전 검증: 원본 논문 evaluator에baseline을 통과시켰을 때2D/MAIN6D 수치 모두차이0.
실제 합성128장 캐시의 모든 후보·좌표·score와 CUDA 정규화 입력은 원본 YOLO와 정확히 같다.
NumPy CPU정규화와 CUDA정규화 사이최대2.98e-8산술차이는 별도 기록했으며 uint8입력은 같다.
중단·재개·미커밋 행 복구·손상 거부를 검증했다. 합성32장100step smoke의 고정batch 총손실은
5.4707→4.5420, 선손실5.4037→4.4410, 코너손실0.06696→0.10104였다.
총손실 감소와 영상 adapter까지의 유효gradient는 확인했지만 이 결과를 코너 정확도 개선으로
해석하지 않는다. 이후 본9회 학습·실사 비교까지 완료했고 최종 결과는 위3절에 있다.

## 5. 실행 후 수치 감사 수정

실사9회와 runtime이 모두 끝난 뒤 집계가 공식 CSV의6자리 반올림 때문에 중단됐다.
`repairs/csv_precision_001/`에 원본 aggregate 소스·실패·계약을 보존하고2D
median/P90 **일치검사만** atol5.1e-7/rtol0으로 교정했다. 실제18개 비교 최대
4.7872e-7px와 허용/거부 경계검사를 확인했다. 6D 엄격도, 원본 지표, bootstrap,
모델·선택·판정은 변경하지 않았다. 별도 `resume_reporting.py`로 집계→보고서→
완료만 재개했고 학습·추론은 반복하지 않았다.

`repairs/report_sd_001/`은 HTML에서 seed 표준편차를 ddof0으로 표시하던 부분을
SUMMARY의 ddof1과 맞춘 별도 감사 수정이다. 원본 표시와 소스를 보존하고 보고서를
재렌더했으며 `INDEPENDENT_REPORT_SD_AUDIT.json` PASS를 확인했다.
`ACTUAL_VISUAL_QA.json`도319×9개 좌표·8개 역할·JS 예외0건으로 PASS다.
브라우저 표시를 다시 확인했고 Discord는 기존204 기록을 유지해 중복 전송하지 않았다.
학습·추론·통계판정은 불변이다. 최종 렌더 상태는 해당 감사와
`REPORT_RENDER.json`을 따른다. 원본 `DRIVER_CONTRACT.json`을 바꾸지 않았으므로
수정된 소스로 원래 driver를 무작정 재실행하면 SHA 검사에서 거부된다.
완료 산출물과 [추론 사용법](../../scripts/research/pallet_line_pose_v1/README.md)을
먼저 사용하고, 재현 실행은 수정 이력을 포함한 별도 계약으로 준비해야 한다.
