# 학습 실행 전 독립 코드 검토

2026-10-01. 검토 대상은 새 `train.py`·`common.py`, 기존 FULL 학습 구현, 최종 `INPUTS.json`, 완료된 CPU preflight와 새 checkpoint의 추론 로더다. 검토자는 새 GPU fit이나 optimizer update를 실행하지 않았다.

**판정: 현재 두 조건·세 seed의 사전 고정 학습 실행을 막는 코드·schema 오류를 발견하지 않았다.** CPU preflight도 실제 PASS 영수증으로 확인했다. 이 판정은 실행 경로와 비교 계약의 검토이며 T/R 개선 판정이 아니다. 실제 GPU 실행과 최종 trace 대조는 이후 증거가 필요하다.

## 원래 FULL과 같은 부분

| 확인 항목 | 코드 및 실물 확인 |
|---|---|
| 초기화 | 각 fit이 같은 PRIOR1을 strict load한다. 초기 canonical state hash는 `08de12a4c9299c585c983ed7c86860f6ac1f692261f1b93eda472c4a3566e99d`이며 CPU preflight에서도 확인했다. |
| 학습 범위 | 기존 `AdaptedPoseFix(FULL)`·trainable 계약 재사용. Convolution 68,508,681개가 학습 대상이며 BN 통계·affine은 동결한다. |
| optimizer | 기존 TFAdam을 직접 import한다. lr1e-4, 기본 betas(.9,.999), eps1e-8이며 새 weight decay를 추가하지 않는다. 매 fit 새 optimizer다. |
| task loss | 기존 `T.parts`를 재사용한다. 유효 crop support의 heatmap CE와 expectation 좌표 L1, 중심8 제외 처리가 같다. |
| accumulation | real8/source8을 micro2 네 번씩 처리하며 각 micro loss에0.25를 곱한다. 원래 convolution L2는 real branch 네 번에만 들어가므로 update당 한 번의 기존 가중치다. source branch에서 L2를 중복하지 않는다. |
| source replay | 같은 기존 source_rows(300,8), TRAIN row 검증, heldout256 분리를 사용한다. `SourceData`가 prediction box crop과 synthetic target를 분리한다. |
| 고정 checkpoint | 정확히300step의 마지막 checkpoint만 저장하며 중간 평가 또는 DEV best 선택이 없다. 기존 fit/START/trace가 있으면 덮어쓰지 않는다. |

BN은 단순히 최초 `eval()`을 호출하는 데 그치지 않는다. 기존 wrapper의 `train()`이 매번 BN을 다시 동결하며, 새 loop는 step마다 BN hash를 검사하고 fit 종료 때 전체 frozen state를 검사한다. optimizer에는 실제 requires-grad 파라미터만 포함되는지 identity set으로 확인한다.

## 의도적으로 달라진 부분과 짝 비교

실사 구성만 SINGLE251 대 DIVERSE251로 바뀐다. 두 군은 같은 seed에서 같은 real **index** 배열을 사용하지만 해당 index의 사진은 다르다. seed1/2/3은 같은 pretrained 초기화에서 학습 순서와 source corruption을 바꾸는 반복이다. 초기 가중치를 서로 다른 무작위 값으로 바꾼3회가 아니다.

실제 입력의 seed별 real order는 `default_rng(6400+seed).integers(0,251,(300,8))`와 exact equality로 확인한다. source row 순서는 모든 군·seed에서 같고 source corruption RNG만 `7102+seed`다. `corrupted()`는 전달한 numpy Generator를 사용하고 target·validity의 공유 원본을 변조하지 않도록 points/valid를 복사한다. 같은 seed의 두 군이 같은 source RGB·row·교란을 받았는지는 최종 trace의 `source_ids`, `source_rows`, `source_corrupted_points_sha`로300step 전부 대조한다.

SINGLE251 감독 코너2008개, DIVERSE2511913개이고 최소 지원 코너는 각각8/4개였다. preflight의 최소4코너 조건과 실제 입력이 일치한다. 기존 loss의 mask 평균 방식도 유지되므로 유효 감독 수 차이가 실효 감독량에 반영된다. 이를 동일 코너 수 비교 또는 recording 수만의 순수 효과로 설명하지 않는다.

## CPU preflight가 실제 확인한 것

[PRETRAIN_TESTS.json](PRETRAIN_TESTS.json)은 PASS이며 다음은 보고서 문구만이 아니라 현재 코드의 실행 assertion과 영수증을 확인했다.

- 실사 pair/image binding, source·checkpoint·target recipe binding과 seed별 index 순서를 검증한다.
- 양쪽 crop의 같은 target mask, 유효 좌표 유한성, index8 제외, inverse-affine 원영상 target≤1e-4px를 확인한다.
- source row가 TRAIN이고256 heldout과 겹치지 않는다.
- wrapper와 원래 PRIOR1의 같은 입력 출력이 CPU에서 exact equality다.
- center target를 크게 바꾸고 valid를 True로 바꿔도 task loss가 동일하다.
- 실제 backward 후 학습 대상에 finite nonzero gradient가 있고 frozen parameter에는 gradient가 없다.
- backward 이후 BN/frozen state와 초기 model state가 그대로이며 optimizer update는0회다.

전처리·source target mapping 검토는 [DATA_PREPARATION_FINAL.json](DATA_PREPARATION_FINAL.json)과 [TRAINING_AUDIT_KO.md](TRAINING_AUDIT_KO.md)의 전체502 pair 검사도 함께 사용했다. 평가 annotation 좌표를 학습 입력으로 읽는 경로는 새 학습 함수에 없다. 다만 기존 Replay teacher의 수동 실사 감독 계보는 존재하므로 방법 전체를 완전한 real-GT-free로 표현하면 안 된다.

## Checkpoint와 추론 호환성

새 학습은 `L.A.original_state(m)`으로 wrapper의 `model.` prefix를 제거한 canonical state를 `final.pt['state']`에 저장한다. 새 `infer.py`는 bare `CORE.PoseFixPallet9()`를 만들고 새 state에 `model.` prefix가 없음을 확인한 뒤 strict load한다. canonical state hash와 로드 후 전체 tensor equality도 검사한다. 기존 FULL125의 wrapper checkpoint와 새 checkpoint의 저장 형식 차이를 각 경로에서 명시적으로 처리하므로 이 부분의 schema 불일치는 없다.

최종 fit 이후에는 실제 저장 파일 hash·strict load·출력 보존 검사가 수행되어야 한다. 현재 검토는 아직 존재하지 않는 새 fit의 직렬화 성공을 미리 입증하지 않는다.

## 검토 중 보완한 추적 범위

최초 TRAIN_CODE_LOCK은 주요 모듈을 잠갔으나 동적으로 사용하는 `SourceData`, `corrupted`, `FeatureDataset`의 정의 파일을 모두 직접 포함하지 않았다. root 요청에 따라 원래 lock을 수정하지 않고 [TRAIN_TRANSITIVE_CODE_LOCK.json](TRAIN_TRANSITIVE_CODE_LOCK.json)을 추가했다.

CPU에서 SourceData를 실제 구성한 뒤 저장소 안에서 import된104개 코드 파일과 위 세 정의 파일을 binding했다. 새 optimizer update·model forward는0회다. root의 실행 wrapper가 이104개 binding을 GPU 학습 직전에 검증하도록 연결한다. 이 영수증은 전이 import 코드 추적을 보완하며 source cache의 모든 대형 ndarray를 새로 전수 재추출했다는 의미는 아니다. SourceData 생성에서 기존 cache manifest/completion·shape/dtype·record-index·partition 계약을 확인한다.

검토 산출물만으로 전체 개선 목표를 완료 처리하지 않는다. 모든 fit의 종료 영수증, matched-source trace, 동일 T/R 계약 평가, 세 seed와 recording 안정성 판정이 다음 검증 단계다.
