# V196: 학습에서 뺀 합성 자료의 끝부위 가림 판독

**새 학습·추론 없이 저장된 모든 source HOLDOUT을 확인했다. 합성 내부에서도 가림을 놓치므로 실사 외관 차이만으로 실패를 설명할 수 없다.** 원 query·L·native0..7을 유지했다. 현재 실사 평가·publicheld·COMMON·기존 VAL은 이 진단에서 읽지 않았다.

15 triplet은 6개 base에서 나온 BEFORE/COVER/SHAM 대조다. 전체 360행 중 직접 weak 라벨은 D30/E15의 45행이고 나머지315행은 UNKNOWN이다. 입력 U는0이다. 실제 도구로 root00–07, 독립 검토자08–14를 나누어 15개 contact를 모두 보았다. 360개 crop을 각각 도구로 봤다는 뜻은 아니다.

| 모델 | CM 실제D→D/E/U | CM 실제E→D/E/U | D recall | E recall | E precision | D/E/D 완전 반응 |
|---|---|---|---:|---:|---:|---:|
| C56 | [28, 2, 0] | [11, 4, 0] | 93.3% | 26.7% | 66.7% | 3/15 |
| C57 | [28, 2, 0] | [8, 7, 0] | 93.3% | 46.7% | 77.8% | 6/15 |

C56은11세트, C57은8세트에서 BEFORE/COVER/SHAM을 모두 D로 답했다. 두 모델 모두1세트에서는 세 변형을 모두 E로 답했다. 학습327행 완전 적합이라는 V194 기록과 구분해야 한다. 이45행은 자연 실사 native 정답이 아니라 구성한 finite-end weak 과제다.

## 입력과 표적에서 확인한 사실

모든15세트에서 COVER는 표적 끝부분 관련 RGB 샘플을 바꾸었고, SHAM은 그 subset을 보존했다. 따라서 입력에 가림 변화가 전혀 없었다는 설명은 이 저장 대조와 맞지 않는다. 다만 변화한 샘플이20–30개라는 것은 보간 지지 수이며, 사람이 읽을 수 있는 신호량·물리 면적·모델 특징 보존량을 측정한 수치는 아니다.

연속 end enclosure는96좌표에서 대략1×1–2픽셀이고, query 중심과 enclosure 중심 거리는 0.22–32.39px(중앙값5.53px)다. **이 거리는 합성 표적 대응을 보는 진단값이며, 실사 키포인트의 참조 좌표 오차가 아니다.** enclosure·정답 위치는 표시·감사에만 쓰고 모델 입력으로 추가하지 않았다.

실제 그림에는 목재와 비슷한 색의 작은 가림을 놓치는 경우와, 회색 패치가 뚜렷하고 query에 가까워도 놓치는 경우가 함께 있다. 색 대비 하나·query 거리 하나·SAM 유무 하나를 단독 원인으로 결론내릴 수 없다. native4/6/7은 이45개 known 정답에 없지만 원 입력 native8개는 모두 유지했다.

| triplet | base/native | 끝 관련 샘플 | q→finite-end 거리96px | C56 B/C/S | C57 B/C/S | 그림 |
|---|---|---:|---:|---|---|---|
| V189_BASE042_NATIVE1 | 42/1 | 25 | 7.22 | DED | DED | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_00/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE050_NATIVE1 | 50/1 | 20 | 4.73 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_01/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE050_NATIVE2 | 50/2 | 20 | 6.02 | DDD | DED | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_02/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE074_NATIVE0 | 74/0 | 25 | 0.84 | DDD | DED | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_03/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE074_NATIVE1 | 74/1 | 24 | 3.48 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_04/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE074_NATIVE2 | 74/2 | 20 | 9.13 | DDD | DED | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_05/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE074_NATIVE5 | 74/5 | 25 | 0.22 | DDD | DED | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_06/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE075_NATIVE0 | 75/0 | 20 | 2.71 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_07/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE075_NATIVE1 | 75/1 | 30 | 5.53 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_08/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE075_NATIVE2 | 75/2 | 25 | 9.14 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_09/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE075_NATIVE5 | 75/5 | 25 | 1.18 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_10/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE083_NATIVE0 | 83/0 | 30 | 12.67 | DED | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_11/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE083_NATIVE1 | 83/1 | 20 | 32.39 | DDD | DDD | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_12/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE083_NATIVE2 | 83/2 | 20 | 27.57 | EEE | EEE | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_13/CONTACT_SELECTED_WEAK_END.png) |
| V189_BASE104_NATIVE3 | 104/3 | 25 | 2.48 | DED | DED | [원 crop](all360_source_holdout_viewer_v1/TRIPLET_14/CONTACT_SELECTED_WEAK_END.png) |

## 해결된 부분과 남은 문제

계보를 묶은 사전 source 분할, 모든 학습 손실의 TRAIN 전용 draw, 동일 input·라벨·고정0 문턱으로 두 모델을 비교하는 절차, crop/예측/라벨의 대응은 확보했다. 저장 source 결과를 재현했고, 현재45행에 입력 누락은 없다. 이것이 모델의 실제 판별 능력을 보장하지는 않는다.

남은 문제는 새로운 source 장면에서의 가림 관계 일반화, finite-end 감독과 사람이 판단한 native D/E의 의미 대응, 작은 부위와 문맥을 모델이 실제로 읽는지다. 물리 GT가 없다는 이유로 기존 사람 D/E 평가를 지우지 않는다. weak 끝부위를 강한 native 물리 정답으로 승격하지도 않는다.

후속 작업은 TRAIN source만으로 native→원query→의도한 부위→감독 영역의 연결을 먼저 점검하는 것이다. 잘못된 대응과 입력에서 읽을 수 없는 감독을 구분한 뒤, 허용된 source 개발자료에서만 한 가지 가설을 정하고 설정을 사전 동결해야 한다. 이번 HOLDOUT과 반복 노출 실사는 진단 결과로 보존하며, 이를 이용한 문턱·체크포인트·모델·데이터 선택이나 실패행 재학습은 하지 않는다. 새 학습·PnP·위치 보정은 이 단계에서 실행하지 않았다.

[전체 원 crop 갤러리](all360_source_holdout_viewer_v1/PORTABLE_GALLERY_KO.md) · [360행 대응표](all360_source_holdout_viewer_v1/ALL360_HOLDOUT_TARGET_MAPPING.csv) · [표적 근거](all360_source_holdout_viewer_v1/ALL15_TARGET_ASSOCIATIONS.json) · [수치 JSON](ROOT_RESULTS_KO.json) · [독립 저장결과/그림 검토](independent_saved_viewer_review_c/REVIEW.json) · [root 그림 검토](ROOT_ACTUAL_FIXED8_SOURCE_CONTACT_VISUAL_QA.json)
