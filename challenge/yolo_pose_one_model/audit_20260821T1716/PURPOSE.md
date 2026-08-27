# PURPOSE — Object-Frame Contract Audit → V2 Data Audit

## 1. 최상위 목표

평가 대상 pallet OBJ 와 target-specific positive 학습 **없이**, generic synthetic
pallet family 만으로 **unseen real pallet 의 monocular 6DoF pose** 를 추정할 수
있는지 논문 수준으로 검증한다.

inference trick 을 더 붙이는 것이 목적이 **아니다**.

## 2. 소비처

- **논문 심사자** — target-free generalization 이 실제인지 판단.
  심사자가 던질 첫 질문: *"당신들이 보고한 6DoF pose 는 고정된 물체 좌표계
  기준인가, 아니면 시점마다 재정의되는 좌표계인가?"*
- **deployment** — 고정된 좌표계에서 실제 pose 가 사용 가능한지 판단.
  포크리프트는 "카메라에 가까운 면 기준 pose" 로는 삽입 축을 정할 수 없다.

## 3. 행동별 WHY / 예상 결과

### AXIS CONTRACT AUDIT (PHASE B)
- **WHY A** — 현재 pose metric 이 실제 fixed object-frame pose 인지 확인.
- **WHY B** — 90° gross error 가 model semantic failure 인지 evaluator
  axis-switch 인지 분리.

```
예상 결과   evaluator 가 per-frame GT dimensions_m 로 3D model 을 만들고,
            그 라벨이 (1.1,1.3)/(1.3,1.1) 두 종 W<->D 스왑이므로
            frame-dependent axis 정보가 들어갈 가능성이 높다
목적 지지   지지 — 이 계약이 틀리면 아래 모든 수치의 의미가 바뀐다
최상위 도달 닿음 — "fixed object-frame 6DoF 를 재고 있는가" 가 논문 주장 자체
독자 첫 질문 "그 dimension 이 배포 때 어떻게 주어지는가?"
```

### V2 DATA AUDIT (PHASE C)
- **WHY A** — frame 수가 아니라 **independent topology coverage** 가 늘었는지.
- **WHY B** — bad-keypoint population 이 어려워한 projection/appearance 영역을
  실제로 포함하는지.

```
예상 결과   기존 audit(V2_READINESS.json)이 V2_NOT_RENDERED / rendered_images=0
            이라고 기록했으므로 감사 자체가 성립하지 않을 가능성이 높다
목적 지지   부분 — 없다는 것을 확정하고 필요 사양을 넘기는 데까지만
최상위 도달 안 닿음 — 데이터가 없으면 일반화 주장에 기여 0
독자 첫 질문 "그래서 V2 는 존재하는가?"
```

### V2 TRAIN (PHASE D/E)
- **WHY A** — CORRECT_BOX_BAD_KP 를 감소시키는지 검증.
- **WHY B** — topology/appearance coverage 확대가 end-to-end pose 까지
  연결되는지 검증.

```
예상 결과   PHASE B 또는 C 게이트에서 막혀 실행되지 않을 가능성이 높다
목적 지지   불가 (이번 실행에서는)
최상위 도달 안 닿음
독자 첫 질문 "왜 안 돌렸는가?"
```

## 4. 이번 실행에서 하지 않는 것

새 학습 0 · 새 inference trick 0 · 새 데이터 생성 0 · 기존 파일 수정/삭제 0 ·
commit/push 0.
