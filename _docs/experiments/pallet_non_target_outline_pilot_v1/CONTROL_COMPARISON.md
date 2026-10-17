# 같은6장 저장 N0·PoseFix 비교

현재 pilot의 고정6장에 **이미 저장된** N0·POSEFIX_RAW 결과를 결합했다. 각 방법의 분모는6장이고 모두 REMAINDER143에 속한다. 새 NN·PnP·최적화·후보 생성·참조 채점은0회이며, 기존 오차 행을 읽어서 결합했다. 사례 선정은 사후 설명용이므로 전체 성능 순위나 일반화 추정으로 해석하지 않는다.

N0는 **N3+SubPix→기존 공통 F**다. N3 단독, SubPix 단독, 최소 B/I 후단 또는 원 전체 e라는 뜻이 아니다. 원 alias는 `pallet_minimal_bi_ablation_v1/reportmaker.py`의 설명과 `regression.py`의 `N0=frame[arms][N3_SUBPIX][coarse_pose]`, 기존 봉인 baseline pose에서 확인했다.

POSEFIX_RAW는 synthetic-only PRIOR1의 seed1·last6000 checkpoint 첫 RAW pass를 기존 W/D selector+F로 처리한 저장 출력이다. checkpoint SHA는 `b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581`이다. cap·SubPix를 적용하지 않았다. 이번6장은143장 저장 보충분에 모두 포함되며 PF+SubPix의 공통95장과 섞지 않았다. 과거 F 안에서 수행된 자세 계산은 이번 호출이 아니다.

원 `ROW_METRICS.csv`와 `BASELINE_PREDICTIONS.jsonl.gz`, `FIXED_REFERENCES.json`의 SHA를 기존 SCORE_SEAL과 대조했다. 원 E0와 이번 BASELINE_E의 T/R은6장 모두 원 허용치1e-9에서 일치한다(최대차 8.8817841970012523e-16). 성공·가용성 bool은 exact하게 같다. 이번 evaluator가 사용한 reference SHA도 원 점수의 fixed reference와 같다. PoseFix143의 원 prediction/protocol seal, seed·step·RAW/cap/SubPix 설정 및 REMAINDER 참조 binding도 확인했다. 따라서 아래 수치는 동일한 기존 physical/C2 참조 기준으로 비교 가능하다. 카메라·참조의 독립 물리 정확성이 검증됐다는 뜻은 아니다.

아래 각 셀은 **위치cm / 회전°**이며 `✓`는 같은 pose에서≤5cm AND≤5°다. 두 오차가 다른 후보에서 온 값은 없다. 전체 precision·가용성·원 status의 결측은 [CSV](CONTROL_COMPARISON.csv)에 보존했다. N0/PoseFix의 과거 accepted/fallback flag는 기록되지 않아 빈칸으로 두며 가용성과 등치하지 않는다.

| ID | 원 e | 새 지지 처리 | mass 대조 | N0 | PoseFix RAW |
|---|---:|---:|---:|---:|---:|
| plastic_night_01:038630 | 99.985 / 88.238 | 99.985 / 88.238 | 80.714 / 88.370 | 3.848 / 2.745 ✓ | 7.295 / 1.500 |
| wood_day_01:005286 | 76.857 / 88.270 | 6.127 / 10.178 | 3.673 / 10.603 | 20.238 / 5.440 | 15.769 / 3.996 |
| wood_night_01:030698 | 29.389 / 19.116 | 29.389 / 19.116 | 29.389 / 19.116 | 0.665 / 2.235 ✓ | 0.614 / 1.815 ✓ |
| plastic_day_01:020955 | 20.662 / 12.681 | 30.310 / 15.639 | 30.310 / 15.639 | 25.066 / 74.432 | 13.724 / 74.206 |
| plastic_day_01:014704 | 4.952 / 0.609 ✓ | 4.952 / 0.609 ✓ | 4.952 / 0.609 ✓ | 3.956 / 1.189 ✓ | 17.001 / 0.568 |
| wood_night_01:030898 | 4.946 / 3.772 ✓ | 4.946 / 3.772 ✓ | 4.946 / 3.772 ✓ | 6.245 / 5.052 | 3.258 / 4.145 ✓ |

| 방법 | 가용/전체 | 성공/전체 |
|---|---:|---:|
| BASELINE_E | 6/6 | 2/6 |
| SUPPORT_AWARE_RERANK | 6/6 | 2/6 |
| MASS_MATCHED_CONTROL | 6/6 | 2/6 |
| N0 | 6/6 | 3/6 |
| POSEFIX_RAW | 6/6 | 2/6 |

이6장에서 N0는3/6, PoseFix RAW는2/6이고 새 두 경로는 각각2/6이다. N0가 성공한 `plastic_night_01:038630`과 `wood_night_01:030698`은 새 지지 처리로 회복되지 않았다. 반대로 원 e가 성공한 `wood_night_01:030898`은 N0에서 실패한다. 따라서 특정 cone 입력만으로 기존 모든 실패나 N0/e 사이의 대응·관측 손익이 해결됐다는 근거는 없다.

기존 [PoseFix 정의 감사](../pallet_statewise_correspondence_diagnosis_v1/POSEFIX_TRANSFER_AUDIT.md), [행별 저장 오차](../pallet_statewise_correspondence_diagnosis_v1/metrics/ROW_METRICS.csv), [기존 점수 seal](../pallet_statewise_correspondence_diagnosis_v1/metrics/SCORE_SEAL.json)를 함께 확인할 수 있다. 이 비교를 selector 입력으로 되돌려 넣지 않았으며 예측 봉인 후의 보고용 결합이다.
