# 재현 유효성 정정 — seed 설정은 달랐지만 실제 난수 흐름은 같았다

판정: `NOT_RUN_EFFECTIVE_TRAINING_VARIATION`. 명목 seed43 네fit은 실제로 완료됐지만 **독립 학습 난수 변화의 재현 실험이 아니라 결정론적 재실행**이었다. 수치 재현 증거만 인정하며, C의 선택을 추가 정당화하거나 성능 변동/재현성을 추정하는 증거로 쓰지 않는다. [정정 JSON](REPLICATION_VALIDITY_CORRECTION.json)에 입력·checkpoint·trace·설치 코드·실행 로그 SHA를 남겼다. 기존 `FINAL_SELECTION.json`, fit 결과와 비용은 덮어쓰지 않는다.

## 실행 검증에서 놓친 점

seed 설정값 변경을 실제 데이터/증강 난수 stream 변경으로 간주했고, **fit 전에 loader generator 차이 및 실제 입력 stream 차이를 확인하지 못한 실행 검증 누락**이다. 프레임워크의 동작만 탓하는 설명으로 대체하지 않는다.

기존 trainer는 `args.seed=43`을 정상 수신했고 로그에도43/workers2가 남았다. 설치 Ultralytics `engine/trainer.py:134`는 `args.seed + 1 + RANK`로 전역 초기화를 한다. 그러나 같은 설치본 `data/build.py:348`의 DataLoader generator는 `6148914691236517205 + RANK`라는 별도 고정 seed를 사용하며 `args.seed`를 받지 않는다. `seed_worker`는 이 generator에서 유래한 `torch.initial_seed()`를 NumPy/Python random에 전달한다. 현재 workers2·동일한 사전학습 R0·동일 목록 조건에서 실제 순서와 증강 흐름이 그대로였음을 trace와 최종 tensor로 확인했다. 다른 모든 설정에서도 seed가 무효라는 일반화는 하지 않는다.

## 독립 확인한 동등성

| 비교 | 명목 설정 | 실제 비교 결과 |
| --- | --- | --- |
| C RAW42 ↔ recipe RAW43 |42→43 |320batch 전체 trace SHA 동일,879/879 state tensor 정확 일치 |
| C REF42 ↔ recipe REF43 |42→43 |320batch 전체 trace SHA 동일,879/879 state tensor 정확 일치 |
| OLD baseline RAW42 ↔ baseline RAW43 |42→43 |879/879 state tensor 정확 일치 |
| OLD baseline REF42 ↔ baseline REF43 |42→43 |879/879 state tensor 정확 일치 |

Tensor 검사는 CPU에서 각 checkpoint의 float `state_dict`를 `torch.equal`로 직접 비교했다. 저장 container SHA가 다른 것은 tensor 차이라는 뜻이 아니다. C의 complete trace에는 모든 sample 이름/순서·RGB/box/support/좌표 digest와 가림 계획이 포함되어 있고 두 arm 모두320batch 전체가 같았다. **과거 baseline42의 전체 trace를 같은 형식으로 대조했다고 주장하지 않는다.** 그 비교는879개 tensor와 저장된 예측/metric 동등성에 제한한다.

따라서 baseline42→43의 관측 오차 변화0, C42→43의 관측 오차 변화0은 학습 난수 변화에 대한 낮은 분산의 추정치가 아니다. 실제 난수 개입이 달라지지 않은 결과다. 같은 출력에서 계산한 paired/LORO 통계도 수치상 재실행일 뿐 독립적인 추가 표본이 아니다.

## 비용과 남은 범위

baseline RAW/REF + recipe RAW/REF **4fit/1,280update를 모두 실행 비용으로 계산**한다. 각 fit은320update 완료됐고 재현 목적을 달성하지 못했다고 비용을 삭제하지 않는다. 네fit의 계측 GPU 시간은 각각68.0897/69.8264/69.1697/69.2387초다. 정정 감사 자체는 GPU0/fit0/update0이며 CPU tensor 비교 process wall 약5.25초였다.

기존 승인 Wood 적용 두fit과 평가·보고만12fit 상한 안에서 마무리한다. 이 정정을 이유로 다시 seed를 바꾸어 재시작하거나 추가4fit을 실행하지 않는다. 최종 상태는 독립 난수 재현이 미완료인 `PARTIAL_BUDGET` 의미를 유지하며 모델 승격 근거로 쓰지 않는다.

향후 별도 승인된 재현 작업에서는 fit 전에 의도한 seed 변화가 실제 sample order와 변환 RGB/가림 계획의 trace를 바꾸는지 확인해야 한다. 동시에 같은 seed의 RAW/REF끼리는 공통 RGB/순서/계획을 유지하는지 확인한다. 이 사전 조건을 확인하기 위한 새 학습은 이번 배치에서 수행하지 않는다.
