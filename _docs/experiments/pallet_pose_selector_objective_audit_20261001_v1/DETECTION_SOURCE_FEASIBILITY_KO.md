# 현재 R0의 source 검출 후보와 비팔레트 감독 가능성

**기존 source 5,120장에는 대상 위치를 선호하도록 감독할 근거가 있지만, 낮은 IoU 후보를 모두 비팔레트로 가르칠 근거는 없다.** 현재 R0가 저장한 모든 7,173개 후보를 확인했다. IoU≥0.5 기준 최고 score 선택은 5,113장에서 대상 박스와 매칭되고, 잘못 선택했지만 맞는 대안이 남아 있는 경우는 TRAIN 4장·VAL 2장뿐이다. 별도 1장은 검출이 없다. 이 자료만으로 새 객체 분류기나 검출 학습이 실사의 콘 혼동을 안정적으로 고친다고 주장하지 않는다.

감사 대상은 **단일 RGB와 치수**, 기존 내부 카메라 K를 사용하는 계약이다. 시간 정보·다른 프레임·새 실사 GT·실사 cone box supervision은 사용하지 않았다. 아래 결과는 source 후보의 구성과 감독 의미에 대한 진단이다. 전체 실사 T/R 개선 결과나 새 배포 선택기의 성능이 아니다.

[전체 수치와 ID별 진단 JSON](DETECTION_SOURCE_FEASIBILITY.json), [재현 코드](../../../scripts/research/pallet_pose_selector_objective_audit_20261001_v1/detection_source_feasibility.py)를 함께 남긴다. 기존 픽셀 예측·checkpoint·source 라벨은 수정하지 않았으며, 새 neural forward·fit·backward는 각각 0회다.

## 1. 어떤 후보와 정답을 비교했는가

현재 R0 checkpoint SHA는 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7`이다. 기존 [source 예측 lock](../pallet_pose_union_selection_20261001_v1/SOURCE_PREDICTIONS_LOCK.json)의 R0 영수증·metadata·protocol·runtime amendment와 5,120개 개별 예측 파일 SHA를 확인했다. 과거의 다른 Y0 checkpoint 출력을 현재 R0 결과로 사용하지 않았다.

모집단은 원래 selector TRAIN 4,096장과 VAL 1,024장이다. G38 2,946장, P0 1,099장, TEX 1,075장이다. 원본 파일 경로의 `images/val` 문자열이 이 연구의 TRAIN/VAL 역할을 정하지 않는다. 역할은 고정된 `SOURCE_INPUTS.json`에서 읽었다. **C1 1,497장과 기존 pose 적격성 검사에서 제외된 부적절한 C2 기하 1장도 포함한 전체 5,120장**을 유지했다. 객체 박스 감사에 6D label 적격성 필터를 적용하지 않았다.

후보는 준비된 RGB 좌표계의 end-to-end one2one 출력 중 **score>0.001, max_det=300** 계약을 통과한 저장 후보다. 이 경로는 IoU NMS를 수행하지 않는다. 모든 프레임에서 원래 선택 index가 최고 score의 첫 index와 같았다. 후보 0개 1장, 1개 3,915장, 2개 이상 1,204장이고 최대 25개다. 이 풀은 threshold 이전 dense anchor 전체가 아니며, 누락된 후보가 dense tensor에도 없다는 뜻은 아니다. [현재 extractor](../../../scripts/research/pallet_line_pose_v1/features.py)와 [앞선 후보 계약 감사](../pallet_pose_joint_recovery_20261001_v1/ASSOCIATION_AUDIT_KO.md)에 실행 경로가 정리돼 있다.

5,120개 source YOLO 라벨의 SHA와 내용도 다시 확인했다. 모두 class 0 한 줄, 정규화 bbox 4개 값, 9점×3개 값으로 구성되며 bound source manifest와 정확히 일치한다. 다만 [라벨 생성 코드](../../../challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py)가 정하는 bbox는 다음 의미다.

- 코너 8개와 중심점의 투영 중 **준비된 영상 안에 있는 점**의 min/max envelope다. 실제로 보이는 물체의 segmentation bbox가 아니다.
- `v=2`는 화면 안에 있다는 뜻이다. 가려지지 않았다는 별도 가시성 정답이 아니다.
- 4,859장은 9점이 모두 준비된 영상 안에 있고, 261장은 일부 점이 밖에 있다. 1,792장의 bbox는 원본 영상 경계를 넘어 100px 반사 패딩에 걸친다.
- 저장 bbox와 해당 투영점 envelope의 최대 차이는 0.001160px로, 6자리 정규화 좌표 저장 정밀도와 일치하는 작은 차이다.

따라서 source IoU는 **이 단일 대상 주석에 대한 위치 일치도**다. 모든 scene object의 semantic annotation이 아니며, 실사의 모든 코너를 감싼 참조 박스 계약과도 가장자리 처리에서 차이가 있다. [원래 source pose 계약](../pallet_pose_union_selection_20261001_v1/SOURCE_CONTRACT_KO.md)의 물리 좌표·축·대칭 검증을 객체/비객체 정답의 증명으로 확대하지 않았다.

## 2. 최고 score와 전체 후보 풀

| 진단 | TRAIN 4,096 | VAL 1,024 | 전체 5,120 |
|---|---:|---:|---:|
| 저장 후보 수 | 5,608 | 1,565 | 7,173 |
| 선택된 후보 IoU≥0.5 | 4,091 | 1,022 | 5,113 |
| 하나 이상의 후보 IoU≥0.5 | 4,095 | 1,024 | 5,119 |
| 선택은 IoU<0.5, 매칭 대안은 있음 | 4 | 2 | 6 |
| 매칭 후보가 없음 | 1 | 0 | 1 |
| 매칭 후보와 IoU<0.5 후보가 함께 있음 | 180 | 52 | 232 |
| 위 경쟁 후보 중 score≥0.1이 있음 | 12 | 6 | 18 |
| 선택된 후보 IoU≥0.75 | 4,062 | 1,014 | 5,076 |
| 하나 이상의 후보 IoU≥0.75 | 4,086 | 1,024 | 5,110 |

IoU≥0.5 선택 성공은 99.863%다. 이 수치는 detector의 source T/R 정확도나 실사 성능을 뜻하지 않는다. source-only 재순위 학습에 필요한 **강한 현재 오선택 사례가 적다**는 관찰이다. 0.75 진단은 위치 정밀도에 따른 차이를 공개하기 위한 추가 고정 구간이며, 결과가 좋은 기준으로 0.5를 바꾸거나 방법 선택 gate로 사용하지 않았다.

아래 구간은 descriptive bin이다. 학습 label이나 runtime threshold를 새로 정한 것이 아니다.

| 후보의 source 대상 박스 관계 | 정의 | TRAIN | VAL | 전체 | 원래 선택됨 |
|---|---|---:|---:|---:|---:|
| 매칭 | IoU≥0.5 | 5,314 | 1,471 | 6,785 | 5,113 |
| 대상 박스와 분리 | intersection=0 | 233 | 42 | 275 | 2 |
| 작은 내부 부분처럼 보이는 기하 proxy | IoU<0.5, intersection/candidate area≥0.8, candidate/target area≤0.5 | 32 | 18 | 50 | 1 |
| 그 밖의 낮은 IoU 겹침 | 위 구간 이외 IoU<0.5 | 29 | 34 | 63 | 3 |

**275개를 ‘명백한 비팔레트 배경’, 50개를 ‘정답 part negative’라고 부를 수 없다.** 전체 대상의 일부, truncation·localization 차이, 패딩 속 복제 모습, 다른 실제 팔레트, 비팔레트 구조물이 섞일 수 있다. GT가 대상 하나라는 이유로 다른 실제 팔레트의 부재를 증명할 수 없다. 비선택 후보 2,054개 중 1,672개는 이미 대상과 IoU≥0.5다. 이를 전부 negative로 다루면 같은 대상의 대안 검출을 억제하게 된다.

선택이 매칭되지 않은 7개 ID는 JSON에 전부 공개했다. 6개의 대안 존재 사례 중 2개는 선택 IoU가 0.480501·0.499310으로 기준 근처다. 이를 육안 검증 없이 실사의 cone part confusion과 같은 semantic 실패로 세지 않았다. 검출이 없는 1장은 `TEX__shard_04_f0110`이며 그대로 포함했다.

## 3. 패딩과 pose 유효성으로 semantic label을 대신할 수 있는가

| 후보 구간 | 전체 | 중심이 반사 패딩에 있음 | box 면적의 과반이 패딩에 있음 | score≥0.1 | 한 개 이상 positive-depth pose |
|---|---:|---:|---:|---:|---:|
| 매칭 | 6,785 | 9 | 9 | 5,568 | 6,785 |
| 대상 박스와 분리 | 275 | 161 | 162 | 6 | 275 |
| 작은 내부 부분 proxy | 50 | 0 | 0 | 3 | 50 |
| 그 밖의 낮은 IoU 겹침 | 63 | 12 | 12 | 11 | 63 |

특히 VAL의 분리 후보 42개 중 39개는 중심이 반사 패딩에 있고, 42개 전부 score<0.1이다. 기존 실사 8장 감사에서는 오선택 중심이 모두 원본 내부였다. 이 차이는 source의 후보 경쟁이 실사의 콘 혼동을 충분히 대표한다고 보기 어려운 근거다. 매칭 후보도 9개가 패딩 중심이므로 패딩 제거를 무손실 규칙으로 추천하지 않는다.

모든 후보에 기존 runtime `candidate_record`를 적용했다. 입력은 **예측 2D점 + source K + 치수**뿐이며 source 정답 pose·정답점은 PnP에 주지 않았다. 각 후보의 W/D 가설 중 한 개라도 이용 가능한지와 코너 8개 양의 깊이를 확인했다. 7,173개 전부, 그중 낮은 IoU 388개 전부가 조건을 만족했다. 유한 PnP 해나 양의 깊이는 그 RGB 부분이 팔레트라는 증명이 아니다.

최소 재투영 오차의 구간별 중앙값도 JSON에 남겼다. 이 사후 분포로 cutoff를 맞추지 않았으며 T/R 오차도 새로 계산하지 않았다. 기존 PnP 기반 selector와 detector 의미 분별을 동일시하지 않는다.

## 4. 실제 RGB 확인의 범위

아래 그림은 예전 source inference 전에 **RGB SHA 순서로 고정한 TRAIN 4장·VAL 4장**이다. 이번 결과로 사례를 고르지 않았고, 원본 RGB와 R0 예측만 표시한다. 8개 패널을 직접 확인했다. 주변 물체·배경·화면 경계와 반사 패딩이 존재함을 확인할 수 있지만, 이 8장만으로 388개 낮은 IoU 후보 전체의 semantic identity를 판정할 수 없다. 그림에 보이는 치수/pose 적격성 표시는 앞선 단계 정보이며 이번 객체 박스 감사의 제외 기준이 아니다.

![결과를 보기 전에 고정한 source RGB 8장](../pallet_pose_union_selection_20261001_v1/figures/01_fixed_source_smoke.png)

G38/P0/TEX의 원시 주석을 source별로 한 개씩 추가 열어 구조를 확인했을 때에도 target `objects`와 scene placement/context count 정보는 있지만, 모든 화면 물체의 exhaustive semantic box 정답은 없었다. 저장소의 generic renderer에 distractor나 숨길 pallet asset 규칙이 있다는 사실만으로 세 데이터 계열의 모든 입력이 그 규칙으로 생성됐다고 인증할 수 없다. 이 감사에서는 새로운 수동 source semantic label도 만들지 않았다.

## 5. 기존 negative9K가 제공하는 것과 남는 한계

기존 `negative_synth_v1_train` 9,000장의 RGB 존재와 라벨을 전수 확인했다. 모든 라벨은 `object_present=false`, `pose_valid=false`, `objects=[]`, `keypoints=[]`, `structural_lines=[]`이며 record의 sample ID도 일치한다. 이는 **semantic absence를 선언한 영상 단위 label**이다. 양성 영상 안의 어느 후보가 팔레트의 일부인지 가르치는 box/part label은 아니다. `target_dimensions` 역시 찾으려는 규격이라는 query spec이지 비팔레트 구조물의 실제 치수 GT가 아니다.

| source negative TRAIN 구분 | 개수 | 생성 출처 메타데이터 |
|---|---:|---|
| N0 matched empty, 재사용 | 3,600 | `legacy_unavailable` |
| N1 structural hard, 재사용 | 785 | `legacy_unavailable` |
| N1 structural hard, 신규 생성 | 2,365 | `generated` |
| N2 pallet-like hard, 신규 생성 | 2,250 | `generated` |
| 합계 | 9,000 | 미확인 출처 4,385 / 생성 메타데이터 있음 4,615 |

원 README는 N0의 background/HDRI/distractor asset ID가 없어 배경 팔레트 배제를 전수 증명할 수 없고, 128장 QA에서 관찰하지 못했지만 full exclusion은 **UNVERIFIED**라고 명시한다. 실제 records에는 N1 재사용 785장도 `legacy_unavailable`로 기록돼 있었다. 따라서 9,000개 empty label의 형식 검증을 9,000개 영상의 독립 semantic 검증으로 보고하지 않는다. `generated`라는 메타데이터 값도 이번 감사에서 각 배경의 모든 픽셀에 팔레트가 없음을 다시 증명했다는 뜻은 아니다.

정확한 로컬 근거는 `data/pallet/training_data/paper_release/negative/extracted/negative_synth_v1_train/{README.txt,records.jsonl,index.csv,labels/}`다. README·records·index SHA와 9,000개 라벨 순서 digest는 감사 JSON에 남겼다. 원 데이터 전체를 문서에 복제하지 않았다. 이 범위에서 현재 `970a0913…` R0에 인증된 negative9K 전체 후보 cache는 확인하지 못했다. 과거 다른 checkpoint의 mining 점수를 현재 실행 결과로 대체하지 않았고, cache를 새로 만들기 위한 추론도 하지 않았다.

## 6. 이미 해 본 접근과 구분

| 선행 근거 | 확인된 내용 | 이번 결과와의 관계 |
|---|---|---|
| [hard negative 계획](../../../challenge/yolo_pose_one_model/hard_negative_v1/PHASE_A/METHOD_SPEC.md)와 [이전 결과 감사](../pallet_pose_joint_recovery_20261001_v1/ASSOCIATION_AUDIT_KO.md) | negative9K에서 1,900장 mining 후 stock/focal negative 학습을 이미 수행. positive suppression으로 STOP | ‘negative를 처음 추가해 보자’는 제안은 새로운 개입이 아님. 당시 Y0 SHA `37f904…`, no-reflect LetterBox640 mining 계약으로 현재 R0와 다름 |
| [기존 within-image pairwise audit](../../../challenge/yolo_pose_one_model/p26_pairwise_signal_audit/FINAL_PAIRWISE_SIGNAL_AUDIT.md) | 구 Y0의 raw anchor TRAIN5,000/VAL1,998. hard delta≤2는 313/131개, 그중 IoU상 duplicate가 95.2%/87.0%. gate 실패로 pairwise loss 30ep 중단 | source hard pair 희소성과 duplicate 편중은 이미 발견됐음. 과거 문서의 NEAR/FAR ‘진짜 distractor’ 표현을 exhaustive semantic 증명으로 재승인하지 않음 |
| 이번 현재 R0 감사 | 전체 5,120장·postprocessed 7,173개, 낮은 IoU 경쟁 score≥0.1 영상은 18장. source top1 오선택에 matched alternate는 6장 | 다른 checkpoint와 후보 계약에서 현재 실제 구성을 다시 확인한 것. 과거 raw-anchor 수치를 이번 풀 수치로 섞지 않음 |

negative-only 영상이 과거 augmentation에서 양성 영상과 절대 섞이지 않았다고도 단정하지 않는다. 기존 계획은 YN negative의 19.7%가 mosaic으로 양성과 섞였다고 적고 있다. 그래도 그것이 현재 콘 부분과 같은 후보에 대한 독립적인 object-versus-part annotation을 제공했다는 증거는 아니다.

## 7. 가능한 개입과 아직 충족되지 않은 조건

현재 bound source bbox로 만들 수 있는 것은 **주석 대상의 위치를 더 잘 설명하는 후보를 선호하는 감독**이다. 후보를 분리·부분·중복으로 구분해 자료를 검토할 수 있고, 원래 최고 score 대신 모든 후보를 보는 데이터 경로도 실제 존재한다. 그러나 semantic 비팔레트 label이 없고 강한 현재 source 오선택이 TRAIN 4장뿐이므로, 곧바로 새 classifier가 안정적 실사 T/R을 개선할 것이라는 근거는 부족하다.

과거와 구별되는 검토 질문은 **“현재 R0가 팔레트가 있는 단일 영상의 비팔레트 부분에 반응하는 source 사례를, 실제 생성 asset/instance 근거로 식별할 수 있는가”**다. 출처가 확인된 기존 합성 자산에서 양성 물체와 비팔레트 구조물을 함께 보여 주되, 모든 실제 팔레트·가림·반사 패딩의 역할을 명시한 instance 근거가 있어야 한다. 이는 가능성을 확인할 다음 자료 조건이며, 이 감사에서 새로운 생성·학습·선택기를 승인하거나 실행했다는 뜻이 아니다. 기존 negative9K의 empty label이나 IoU<0.5만으로 이 조건을 충족했다고 볼 수 없다.

source-only 객체 재순위는 새 실사 GT를 학습에 쓰지 않을 수 있지만, 원래 후보 풀에 매칭 후보가 없는 실사 5장을 복구하지 못한다. 후보 생성·좌표 localization의 변화가 필요한 문제와 기존 후보 선택 문제를 분리해야 한다. 또한 올바른 박스는 올바른 W/D·회전·키포인트를 보장하지 않으므로 전체 T/R·P90·실패·clean 보존 검증은 그대로 필요하다. 이번 source 감사로 전체 173장 평가 모집단을 줄이거나 기록별 규칙·cone 색 threshold를 만들지 않았다.

이번 VAL source label은 진단에 사용됐고 과거에도 검토된 자료다. 이후 이를 untouched confirmation으로 부를 수 없다. 기존 학습 모델·실사 routing을 변경하지 않았으며, 이 보고서는 **검출 개입의 데이터 근거를 확인한 자료**로 고정한다.

## 8. 재현과 검증

코드는 원 source input과 라벨 SHA, 모든 R0 cache SHA를 대조한 후 기존 OpenCV PnP만 실행한다. source reference pose에 대한 T/R scoring은 수행하지 않는다. 실행 과정에 실사 reference 경로 차단 hook을 설치했다. 모든 5,120개 ID와 target bbox 관계, 388개 비매칭 후보의 기하 통계, source/type/split 집계, 입력·코드 SHA를 JSON에 남겼다. 파일 읽기 목록은 공개 문서 크기를 줄이기 위해 개수와 정렬된 경로 digest로 보존했다.

```bash
MPLCONFIGDIR=/tmp/pallet-stability-mpl python -m scripts.research.pallet_pose_selector_objective_audit_20261001_v1.detection_source_feasibility
```

기존 산출물이 있으면 덮어쓰지 않고 실패하도록 작성했다. 전체 bbox 진단에 대한 추가 패딩 통계 확인으로 CPU 계산을 두 번 수행했으며, 결과가 같은 core count는 유지됐다. 최종 실행의 PnP 호출은 후보당 1회, 총 7,173회다. 이것은 모델 forward 7,173회가 아니다. 두 실행 모두 GPU inference·학습·실사 GT 읽기는 0회였다. 원 phase 문서와 결과는 수정하지 않았다.
