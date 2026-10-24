# V197: 원 query와 학습 부위의 대응 진단

**학습용 자료에서 감독 부위와 원 입력의 관계를 확인했다. 새 모델·학습·예측·PnP는 실행하지 않았다.** 현재 head는 RGB96·valid·고정 crop 상대XY와 선택적 frozenSAM을 받으며, native 번호·전체8점 좌표·감독 끝점 좌표를 직접 받지 않는다. 이 구조 사실은 기존 코드 검토에 근거한다.

원 query·L·native0..7과 weak 라벨을 그대로 두고, 점수를 열지 않은 상태에서 전체 TRAIN109개 대응 metadata를 확인했다. 고정 hash로 선택한12세트의 원 crop과 같은 영상의 전체문맥을 함께 보았다. 이 기록은 분류 성능 검증이나 독립 human gold 작성이 아니다.

| 직접 감독 출처 | triplet / D·E | 원본 범위 / component | canonical 제공·UNKNOWN | q→후보영역 중심96 최소·중앙·최대 |
|---|---|---|---|---|
| 구성적 합성 finite-end | 82 / 164·82 | 41 bases / 41 | 82·0 | 0.445·6.759·33.709 |
| 공개실사 RGB-weak + YCB 차폐 | 27 / 54·27 | 12 frames / 1 | 0·27 | 4.027·11.709·35.614 |

이 거리는 crop96의 q중심과 저장된 후보 enclosure 중심 사이의 설명값이다. **물리 키포인트 좌표 오차나 좌표 보정 정답이 아니다.** 구성적 합성의 enclosure 폭/높이는 약0.918–1.567 /1.024–2.036이고, 공개실사는6.651–19.170 /8.539–23.093이다. 후보 영역의 정의·불확실성이 다르므로 물리 면적이나 난이도의 직접 비교로 해석하지 않는다.

| native | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| known triplet | 35 | 26 | 13 | 22 | 6 | 7 | 0 | 0 |

native6/7에는 direct known 감독이 없지만 원native8 입력을 유지했다. 공개실사의 canonical27개 UNKNOWN도 유지한다. 없는 대응을 permutation이나 GT 끝점으로 채우지 않고, UNKNOWN만을 이유로 모든 가시성 학습을 차단하지 않는다.

## 실제 본 화면과 읽을 수 있는 범위

선택12=합성8+공개실사4이며 288행(D24/E12/UNKNOWN252)이다. 입력 U24(8.33%)를 유지했고 모두 선택된 공개실사96행 안에 있다. 이는 전체 TRAIN109의 가용성 비율이 아니다. 같은 exact TRAINpacket의 전체문맥36개가 모두 존재했고 다른 영상으로 대체하지 않았다.

Root는 고정00–07의 contact8장+BEFORE문맥8장, 독립 검토자는08–11의 contact4장+BEFORE문맥4장을 실제 도구로 보았다. 12개 contact와12개 BEFORE문맥을 본 범위이며, 288개 crop이나36개 문맥 전부를 눈으로 검사한 것은 아니다. PNG 원문 대응과 crop ROI는 별도 독립 검산으로 확인했다.

공개실사 Q0041_D3·Q0033_D2·Q0381_D2에서는 원 query 중심과 표시된 끝 후보 영역이 떨어져 보인다. 이는 q나 라벨이 틀렸다는 판정이 아니다. literal 중심 픽셀의 소속과, 의도한 팔레트 끝부분의 차폐 여부가 서로 다른 질문임을 보여 준다. 전체문맥은 해당 팔레트 부위를 이해하는 데 도움을 주지만 physical/native 정체성을 인증하지는 않는다.

합성 BASE020·BASE082에서는 작은 가림이 목재와 비슷하게 보이고, BASE095에서는 배경과 비슷한 작은 윤곽 변화로 보인다. BASE118·BASE057·BASE089처럼 색이 구분되는 사례도 있다. 픽셀 변화가 있다는 사실과 사람이 부위를 식별할 충분한 단서가 있다는 주장을 구분한다. 이번 화면에는 예측이 없으므로 이 요인이나3×3 표현 압축을 모델 실패 원인으로 확정하지 않는다.

## 현재 판단과 다음 확인

직접 감독의 출처·부위·입력 대응을 명시했고, 원좌표·누락·UNKNOWN을 보존한 채 비교할 화면을 확보했다. 공개실사 direct27쌍은12개 프레임·1개 저장 component에 속하며 자연가림 정답27개나 독립27영상이 아니다. 가림 관계 일반화와 사람 native D/E에 대한 전이는 여전히 미해결이다.

다음은 학습용 근거만으로, 배포 때 얻는 native 번호와 초기8점 문맥을 head에 전달할 수 있는지 별도 계약부터 확인한다. 원 crop·query·L·키포인트 순서·기존 라벨은 유지하며 감독 끝점·GT pose를 입력으로 추가하지 않는다. 번호 조건과8점 조건의 가용성을 분리해 검토하고, 증강된 한 점과 다른 점을 임의로 같은 추론 결과처럼 합치지 않는다. 이 보고서는 새 모델·자료·문턱을 채택하거나 새 학습을 실행한 기록이 아니다.

[고정12 원 crop·전체문맥](fixed_train_target_contract_viewer_v1/PORTABLE_GALLERY_KO.md) · [HTML 갤러리](fixed_train_target_contract_viewer_v1/index.html) · [전체109 대응](fixed_train_target_contract_viewer_v1/ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json) · [선택288행](fixed_train_target_contract_viewer_v1/SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv) · [결과 JSON](ROOT_RESULTS_KO.json)

[독립 저장·PNG·실제뷰 검토](independent_saved_train_viewer_review_b/REVIEW.json) · [독립 해석](independent_train_target_interpretation_a/KO.md) · [root 실제뷰](ROOT_ACTUAL_FIXED8_TRAIN_CONTACT_AND_CONTEXT_VISUAL_QA.json) · [기존 head 코드 검토](../query_visibility_grouped_source_holdout_target_readout_20261024_v196/architecture_code_review_from_b/KO.md)

[기존 V194 source 출처·학습 계약](../query_visibility_source_group_split_direct_end_training_20261024_v194/ROOT_REPORT_KO.md) · [기존 V193 공개 source 제어·권한](../query_visibility_source_end_pairs_opposite_sham_20261024_v193/source_pair_visual_readout_v2_caption_fit/PORTABLE_GALLERY_KO.md). 기존 source credit을 유지하며 새 라이선스 검사를 했다고 주장하지 않는다.
