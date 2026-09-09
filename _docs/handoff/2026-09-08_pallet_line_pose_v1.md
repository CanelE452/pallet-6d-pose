# Pallet line pose v1 — final handoff, 2026-09-08

**전체 실행 완료, driver 종료.** 구현·실제 학습·합성 전용 선택·실사9개 평가·
독립 수치 감사·HTML 자동 열기·Discord HTTP204 전송을 완료했다.
`complete/PASS=true`, 사전등록 `overall_accuracy_improved=false`다.
미완료 학습이나 재개할 GPU 작업은 없다. 이 기록은 이전 중간 상태를 대체한다.

## 결과와 해석

주실험은 실행 전에 정한 `image_joint`의 seed1,2,3이며 실사에서 좋은 군/seed를
고르지 않았다. R0와 비교한 원본 evaluator 수치의3seed 평균:

| 지표 | R0 → image_joint | 세션 대응95% 차이 구간 | 개선 확인 |
|---|---:|---:|---|
| 9점 median, px ↓ | 6.6157 → 6.0409 | [-0.9715,-0.3349] | 확인 |
| 9점 P90, px ↓ | 38.6700 → 37.1011 | [-1.9000,+0.3917] | 미확정 |
| 회전 median, ° ↓ | 2.2625 → 2.1126 | [-0.4334,+0.0124] | 미확정 |
| Translation median, cm ↓ | 7.8969 → 7.5730 | [-1.3720,-0.0732] | 확인 |
| IoU3D median ↑ | 0.6032 → 0.6301 | [+0.0043,+0.0472] | 확인 |
| ADDsym AUC ↑ | 0.4285 → 0.4490 | [+0.0119,+0.0303] | 확인 |

모든 평균은 개선됐지만 네 지표만13세션 대응10,000회 bootstrap 구간까지
개선을 지지한다. P90·회전이 엄격한 사전등록 조건을 만족하지 못하므로
`keypoint_gain_confirmed`, `pose_gain_confirmed`, `overall_accuracy_improved`는
모두 false다. 개별2D 개선을 종합6D 우월성으로 바꾸어 말하지 않는다.
독립 지표·3seed 표본SD(ddof1) 재계산 일치, primary CI 최대차1.67e-16 PASS.

사후 GT 설명 지표인9점 frame mean에서는 매칭311프레임 중248개 개선/63개
악화였다. Baseline≤5px91장3.622→3.444px, (5,10]px85장7.168→6.742px,
>10px135장42.975→42.120px. 어려운 큰 실패가 해결됐다고 보기는 어렵다.
미매칭8장은 분모에 남으며 GT 난이도는 운영 gate가 아니다. GT 지원2418개
frame/role의 선 endpoint 거리14.065→13.432px(height18.754→18.010,
depth9.312→8.793). DAY/NIGHT·plastic/wood·13세션의 frame mean은 평균 개선했다.

Geometry-only 대비 image_joint는 frame mean0.4544px 개선했다. 하지만
image_line_only 대비 joint pooled median 차이는−0.0291px,
95%구간[-0.0643,+0.0851]이며 여섯 지표 모두 joint 추가 기여는 확정되지 않았다.
각 군의 합성 선택 규칙까지 포함한 비교다. GT 난이도·선오차는 설명용이며
재선택·추가 학습·운영 threshold로 사용하지 않았다.

주실험의 seed별 latency median 평균은 **9.4028→17.5855ms**다. 고정26프레임의
실제 반복 BGR→전처리→YOLO→선 분기·결합→원본2D좌표 시간이며 파일 decoding과
PnP는 제외한다. Pose coverage319/319와 box AP·score·후보 순서·비선택 인스턴스·
중심점 보존을 확인했다. Negative2689장은 baseline 후보를 복사하고 전수 재비교한
것으로 새 모델 negative 전체 재추론이나 속도 실험이 아니다.

PAPER_EVAL_ALL_POS319/negative2689와13세션은 **재사용 DEV**이고
`held_out_final=false`다. 독립 final 일반화 시험 또는 baseline 종합 우월성으로
확대하지 않는다. 이전 DEV52/8corner DHT 결과와 논문9point 수치를 혼합하지 않는다.

## 고정 구현·학습 계약

- 코드: `scripts/research/pallet_line_pose_v1/`; 결과: `data/pallet/results/pallet_line_pose_v1/`.
- 논문 R0: `challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt`, SHA `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`.
- Frozen YOLO P3/P4 spatial features +19,810개 파라미터의 local DHT-inspired 선 분기. DOPE/원본 DHT 논문 전체 재현은 아니다. Predicted-point-conditioned13각도×17offset+null,32 samples, 미분가능 불확실성 가중 점 결합.
- Camera-facing0123, height `(1,2),(3,0),(5,6),(7,4)` / depth `(0,4),(1,5),(2,6),(3,7)`. 폭 방향4선은 제외. Amodal cuboid support이며 물리 edge 가시성·attention GT를 뜻하지 않는다.
- 실제3arms(image_joint,geometry_joint,image_line_only)×3seeds×6,000steps,batch16. 합성60,000행 cache 모두 보존; train55,980 중 실제 eligible55,915, calibration1,004/selection1,031/heldout1,985. 9개 최종 checkpoint와 optimizer step6000 감사 PASS.
- 합성만으로 보정·선택한 T=1, λ=.25(모든 군); 이동 상한은 geometry_joint만 원본 대각선1%, 나머지 없음. SELECTION을 실사 전에 동결하고 heldout도 선택 후 평가했다. 실제 실사319×9회 추론·논문2D/MAIN6D 완료.
- Source validation은 기존 R0/탐색에서도 사용된 합성 계보여서 새 분기의 heldout이라는 한계가 있다. 실제 R0의 G38+legacy P0/TEX55,980/val4,020 출처를 다른73,916/realFT157 명세와 혼합하지 않는다.
- Python `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`, 설치 Ultralytics8.4.60. 원본 이미지 reflect100/imgsz640/conf.001, topconfidence 선택, IoU≥.5와 visibility>0는 평가에만 사용. GT를 추론 입력으로 제공하지 않는다.

고정 SHA:

| 파일 | SHA256 |
|---|---|
| TRAIN_PROTOCOL.json | `35ccbb2e198774b33d3f5deda60564a8237c478c772c9bb7a63d5cae79b3bbb3` |
| SOURCE_MANIFEST.json | `feaa24075d31c4227e3397b7450a19a0f1d3a73dc0203d12a8ca5b927eb59789` |
| SELECTION.json | `06c27494ba4a4928a0c928e7f108f0b507fc5c078b9888cc398fec402295af76` |
| SUMMARY.json | `b0bbe8e373c3ab7f0b6098b171c8bc869285c42fae2579b0e2a93f440016ee0a` |
| VERDICT.json | `091674475dda9d9a2e529fcf30f91387ccaff60a2e48088c8b255186c2c89d59` |
| DIAGNOSIS.json | `85bce04812f783734dad2e4ff657e5e783aa68f6411ac4abecec888ce23a0837` |

## 감사 수정과 재실행 주의

원래 driver session94884/PID142967는 종료됐다. 실사9개 평가와 runtime이 끝난 뒤
집계의 CSV 소수6자리 반올림 일치검사에서 멈췄다. `repairs/csv_precision_001/`에
원본 aggregate(6ce51528…)와 실패·계약을 보존하고,2D median/P90 **일치검사만**
atol5.1e-7/rtol0으로 바꿨다(새 aggregate SHA d8b78c67…).18개 검사 최대차
4.7872e-7px와 경계검사 PASS,6D 엄격도·수치·bootstrap·모델·선택·판정은 그대로다.
`resume_reporting.py`는 완료된 단계 SHA를 확인하고 집계→보고서→완료만 재개했다.
재학습·실사 재추론은 없었다.

별도 `repairs/report_sd_001/`은 HTML seed SD 표시를 ddof0→ddof1로 맞추고
보고서를 재렌더한 감사 수정이다. `INDEPENDENT_REPORT_SD_AUDIT.json` PASS,
`ACTUAL_VISUAL_QA.json`도319×9저장 좌표·8역할·브라우저 전환·JS예외0건 PASS다.
원본 보고서·소스를 보존했고 학습·추론·SUMMARY/VERDICT 통계와 판정은 불변이다.
최신 HTML SHA는 `367b15e707ae21f2059d11b25160e772b529568043e1cc477e6b9d33edc1599c`.
브라우저 표시를 다시 확인했으며 Discord는 기존 HTTP204 receipt로 중복 전송을
방지했다. 최신 완료와 렌더링 증거는 해당 repair 및 `COMPLETION.json`,
`REPORT_RENDER.json`을 따른다.

원본 `DRIVER_CONTRACT.json`은 수정하지 않았다. 따라서 수정된 소스로 원래
`driver.py`를 그대로 실행하면 SHA 검사에서 거부된다. **완료 폴더에 무작정
재실행하지 않는다.** 완료 산출물을 읽거나 [README 추론 API](../../scripts/research/pallet_line_pose_v1/README.md)를
사용한다. API는 complete final6000 checkpoint와 고정SELECTION의 SHA 일치를
요구하며, seed1 예시는 좋은 seed 선택을 의미하지 않는다. 새 재현 실험이 필요하면
수정 이력을 포함한 별도 계약·결과 폴더를 준비한다. 새 실험은 이번 완료의 미진행
단계가 아니다. 원본 데이터·모델·선택·논문 evaluator 변경이나 commit 요청은 없다.

## 정본 산출물

[SUMMARY](../../data/pallet/results/pallet_line_pose_v1/SUMMARY.json),
[VERDICT](../../data/pallet/results/pallet_line_pose_v1/VERDICT.json),
[RUNTIME](../../data/pallet/results/pallet_line_pose_v1/RUNTIME.json),
[DIAGNOSIS](../../data/pallet/results/pallet_line_pose_v1/DIAGNOSIS.json),
[독립 최종 감사](../../data/pallet/results/pallet_line_pose_v1/INDEPENDENT_FINAL_METRIC_AUDIT.json),
[HTML 보고서](../../data/pallet/results/pallet_line_pose_v1/index.html),
[전체 완료·브라우저·Discord 기록](../../data/pallet/results/pallet_line_pose_v1/COMPLETION.json).
설계·해석은 [notes](../notes/pallet_line_pose.md), 날짜별 실행 기록은
[history](../history/2026-09-08.md)를 따른다.


## 당시 실행 경과 — 종료된 작업의 역사

아래는 과거 snapshot이며 현재 실행 중인 작업을 뜻하지 않는다. 현재 상태는 이
문서 첫 절과 완료 산출물을 따른다. 모든 원래 stage log와 실패 receipt는 보존했다.

- 실행 전: source/좌표/계보 감사, R0 pass-through2D/MAIN6D 차이0, model15/readout8/inference8 CPU 검사,128합성 feature smoke·손상/재개 감사 PASS. 실제32frame100step smoke는 total5.4707→4.5420이나 corner0.06696→0.10104로 악화했으며 정확도 이득으로 해석하지 않았다. SmokeGPU session29573은 완료됐다.
- 01:44:25 KST: 원래 driver session94884, PID142967, cache child142972 시작. 첫512/60,000행은14.65초에 commit, GPU636MiB. DRIVER_CONTRACT가14개 source를 동결했고 cache 약74GB를 추출했다. 당시 NOT_RUN HTML은 준비 화면이었다.
- 02:12:19 KST: cache60,000행 PASS(1668.83초; detected59,999/matched59,921), train worker192010 시작. Eligible train55,915/55,980이며 제외65행도 source/cache 분모에 보존했다.
- 첫 image_joint_seed1: 실제6,000step 및4,020validation logits 완료. 고정 train probe total5.6096→3.2159, corner0.20306→0.20027; 학습 진단일 뿐 일반화 증거는 아니었다. 다음 geometry_joint_seed1이350/6000일 때 중간 인계를 남겼다.
- Seed1 세 군3/9완료 후 seed2 진행. FIRST_TRAINED_MODEL_AUDIT는 실제19,810changed parameters,17개 optimizer state step6000, train-only96,000노출,4,020validation 유한성·SHA·분리 PASS. 이후 seed1+2의6/9완료, image_joint_seed3의50/6000 당시 중간 인계를 남겼다.
- 03:11:43 KST: 모든9학습과 validation logits 완료(TRAIN stage3564.1초). SELECTION SHA06c27494…af76을 실사 전에 고정하고 heldout1,985장을 평가했다.
- 03:12:15 KST: real_evaluation 시작. 첫 image_joint_seed1 실제319장 추론·allcandidate parity PASS·MAIN6D 완료 이후 나머지8군과 runtime을 마쳤다.
- 이후 원래 driver는 집계 CSV 저장 정밀도 검사에서 중단됐다. 감사 수정과 별도 reporting resume로 집계·HTML·최종 알림을 완료했다. 원래 session/PID는 재개 대상이 아니며 종료됐다.
