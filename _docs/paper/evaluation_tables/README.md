# Paper Evaluation Tables

이 폴더는 논문용 결과표의 **빈 템플릿 정본**이다. 결과가 확정되기 전에는 수치를 채우지 않는다.

## 사용 규칙

1. 현재 `current_real_dataset`의 128/140 positive와 2,689 negative는 **DEVELOPMENT** 데이터다. 최종 논문 성능 수치를 여기에 채우지 않는다.
2. 최종 수치는 `_docs/paper/current_real_dataset/FINAL_TEST_REQUIREMENTS.md`를 만족한 **새로운 frozen final test**에서만 채운다.
3. 메인 pose 지표는 다음 5개로 고정한다.
   - Box AP50:95
   - ADD(-S) AUC
   - Rotation median (deg)
   - Translation median (cm)
   - Yaw median (deg)
4. `5cm5deg`는 paper-facing main metric으로 사용하지 않는다.
5. 다른 논문 방법과의 공정한 controlled comparison에서는 가능한 한 **동일한 학습 이미지 membership, 동일한 supervision budget, 동일한 final test, 동일한 평가 구현**을 사용한다.
6. 단, 각 방법의 optimizer/lr/augmentation까지 억지로 동일하게 강제하지 않는다. 각 방법에 합리적인 native training recipe를 허용하되, target-specific CAD·real supervision·bbox-at-inference 같은 추가 정보는 표에 명시한다.
7. published dataset의 원 논문 수치를 우리 in-house 결과와 직접 숫자 비교하지 않는다. 같은 final in-house dataset에서 재평가한 값만 같은 정량 비교표의 주 비교 대상으로 사용한다.
8. Real-FT 모델은 target-free method와 동급 baseline이 아니라 **supervised upper bound**로 표기한다.

## 모집단 ID — 한 행에 두 모집단을 섞지 않는다

```
DEV_POS140          140    reviewed clean real DEV (FT leak 미제외)
COMMON_DEV_POS128   128    140 - FT overlap 12. 모든 controlled DEV 비교는 이것만 쓴다
DEV_NEG2689       2,689    real negative DEV
FINAL_POS              0   미동결 — untouched final membership 없음
FINAL_NEG              0   미동결
```

`DEV_POS140`은 migration·selector 진단에만 쓴다. 논문 final 표는 동결된
`FINAL_POS` / `FINAL_NEG` 로만 채운다.

## 현재 gate 상태

> **POSE 열은 CANONICAL_POSE + SELECTOR + SYMMETRY + FINAL TEST 가 모두
> 통과할 때까지 비워둔다.**

canonical migration 과 yaw-180 대칭은 명시적 동치류로 통과했다. 그럼에도 열이
비어 있는 이유는 W/D-parity selector 진단과 untouched FINAL membership 이 아직
통과하지 못했기 때문이다.

세부 표는 `RESULT_TABLE_TEMPLATES.md`에, controlled comparison / native-setting
reference / architecture-only 의 구분은 `COMPARISON_PROTOCOL.md`에 있다.
