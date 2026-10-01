# Source VAL 평가 코드 독립 검토

2026-10-01. `scripts/research/pallet_pose_union_selection_20261001_v1/source_val.py`를 `seal_training.py`, 공유 `train.py` scoring 함수 및 `source_features.py`의 metric/gate 정의와 대조했다. 이번 검토는 코드와 이미 동결된 계약 JSON만 읽었다. 학습·이미지 추론·실제 VAL routing/오류 채점·실사 GT 읽기를 실행하지 않았다.

## 검토 결과

수학·후보 선택·전체 분모·비교군 구성은 최종 설계와 일치한다. 발견한 reference-chain·guard 보완이 담당 구현자에 의해 반영됐으며, 수정본을 다시 읽어 확인했다. **검토 범위 내 남은 실행 차단 결함은 없다.** 이 문서는 실행 성공 또는 T/R 개선 판정이 아니다.

최종 검토본 `source_val.py` SHA256: `867ac51c6e43808cc5af4661a885b5d5860612a9d7b5be2a1486210555bf0c3b`. 대조한 `seal_training.py` SHA256: `897a52c84b827fb25e46d34608a88da4af02e6d7ee71c12915988952c55e3f5e`.

| 검토 항목 | 확인 내용 |
|---|---|
| 여섯 최종 모델 | R0_ONLY/UNION × seeds1·2·3의 집합을 정확히 요구한다. 6fits·총1980updates·각330steps·final epoch30·checkpoint/trace 상태 hash·paired 초기값/정규화/순서 hash를 확인한다. seed/epoch 선택 경로가 없다. |
| GT 이전 routing | 기본 source reference guard를 닫은 채 모든 checkpoint·입력·후보 pose를 검증하고, 여섯 모델의1024 choices와 routing lock을 저장한다. score는 이 잠금을 다시 검증한 뒤에만 reference 접근을 연다. 실사 reference guard는 계속 닫혀 있다. |
| 후보 mask와 whole pose | 학습과 같은 `score_candidates`/`select_candidates`를 사용한다. invalid 후보는 선택 불가이며 동률은 R0·가설 이름 순서다. 선택한 expert/가설의 R와 t를 함께 가져오며 양축을 다른 후보에서 합치지 않는다. |
| 0후보 fallback | 두 arm 모두 동일한 저장 R0+GEO whole pose를 사용한다. 그것도 unavailable이면 실패로 남긴다. 단순히 pose를 반환했다는 사실은 정확하다는 뜻이 아니다. |
| 참조 좌표계 | 적격 C2/proper-rigid VAL1024만 연결하며 table stem·split·K·dimensions를 재확인한다. `R_reference=renderer.R@diag(1,-1,-1)`, t 불변, C2 full-rotation geodesic은 동결 source 계약과 일치한다. |
| 실패 포함 분모 | 모든1024행을 유지한다. 실패는 T/R 모두 +∞이며 full-population quantile와 conditional summary를 구분한다. JSON null의 상태를 별도 기록하고 NPZ는 +∞를 보존한다. |
| 45개 필수 판정 | UNION3seed × {paired R0_ONLY, R0_GEO, paired DIVERSE_GEO} × {T median 엄격 개선, R median 엄격 개선, T P90≤1.05배, R P90≤1.05배, 실패 수 비증가}다. 하나라도 실패하면 실사 routing을 승인하지 않는다. |
| protocol 일치 | 최종 source_val의1024행·3seed·3비교군·final-only·fallback 규칙은 seal_training이 쓰는 필드와 일치한다. 코드·입력·feature/prediction lock·runtime amendment·최종 checkpoint를 hash로 연결한다. |

## 봉인 전에 발견하고 수정 확인한 사항

1. 최초 검토본은 score에서 현재 historical `SYNTHETIC_SPLIT_LOCK.json`을 읽고 그 파일의 geometry/records hash를 신뢰했다. 수정본은 routing 검증 뒤 동결 SOURCE_CONTRACT의 history binding을 먼저 검증하고, geometry/records binding도 같은 계약과 일치시킨다. 중복 binding의 충돌도 거부한다. 이렇게 reference와 history가 함께 바뀌어도 동결 계약과의 불일치를 검출한다. 검토자는 두 JSON에 저장된 기대 binding들이 실제로 같음을 비교했으며 raw reference 파일은 열지 않았다.
2. 최초 guard는 raw geometry·SYNTH reference 접근을 막았지만 현재 `SOURCE_VAL_METRICS.npz`와 `SOURCE_VAL_GATE.json`에는 명시적 차단이 없었다. 수정본은 두 quality artifact도 차단하고, 신규·idempotent score 모두 routing 검증이 끝난 뒤 접근을 연다. 따라서 결과를 먼저 열 수 없다는 약속을 audit hook으로도 강제한다.
3. routing 전 protocol 입력 hash 검증은 이미 만들어진 TRAIN label container를 byte 단위로 읽는다. 수정본 disclosure는 **VAL label 값을 열거나 채점하지 않음**과 TRAIN label SHA 검증을 구분한다. TRAIN label 값은 routing 계산에 사용하지 않는다.

파일 AST 검사는 통과했다. 실제 source VAL 수치에 관한 결론은 모든 fit·routing lock 이후 생성되는 결과에서 별도로 판단해야 한다.
