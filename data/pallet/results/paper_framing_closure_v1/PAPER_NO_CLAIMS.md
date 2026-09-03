# 주장하지 않는 것

> 정본 금지 문장 목록은 `_docs/paper/final/PAPER_CLAIM_LOCK.json` 의
> `forbidden_sentences` 다.  전부 유효하다.  아래는 **이번 작업으로 새로 추가되는**
> 금지 항목이다.

## 새로 금지되는 문장

```
"YOLO 검출 상자를 pose 계산에 함께 쓰면 translation 이 좋아진다"
    C1 이 반증했다.  ΔIoU3D -0.0660, ΔADDsym -0.0492, session CI 둘 다 0 배제.
    게다가 잘못된 semantics 로 맞추던 V1 S2(-0.0216) 보다 **올바른 semantics 가
    더 나쁘다**

"구조선(line) 을 회전에 붙이면 pose 가 좋아진다"
    L3 가 두 seed 모두 반증했다.  ★ 단 "lambda 가 클수록 나빠진다" 라고 쓰지 않는다 —
    seed1 은 lambda 3.0, seed2 는 lambda 1.0 이라 seed 와 lambda 가 confounded 다.
    허용 표현: "두 historical seed-specific line-fusion 구성이 모두 강한 YOLO
    기준선을 개선하지 못했고, 더 큰 악화는 seed-1 구성에서 나타났다"

"시간축 일관성을 쓰면 pseudo-label 이 좋아진다"
    정식 모집단 계약 아래 적격 centre 가 0 개다.  실패했다고도, 성공했다고도
    말할 수 없다.  POPULATION_LIMITED 라고 적는다

"depth 로 pseudo-label 을 보정하면 정확도가 오른다"
    Gate 1 은 실행되지 않았다.  센서 검증은 NOT_READY_FOR_GATE1 이다

"실제 GT 로 미세조정하면 되므로 우리 방향이 옳다"
    REALFT 는 통제된 비교가 아니라 상한선이다.  같은 열 블록에 넣지 않는다

"6D pose 가 개선됐다"  /  "yaw 오차가 줄었다"  /  "self-training improves 6D pose"
    lock 의 금지 문장 그대로.  6D 표가 생겼다고 해서 이 금지가 풀리지 않는다.
    표가 생긴 것과 차이가 갈리는 것은 다른 문제다

"temporal 방법이 실패했다"
    POPULATION_LIMITED 이다.  적격 centre 가 0 개라 성공도 실패도 측정되지 않았다

"depth 보조 monocular 방법"
    Gate 1 이 실행되지 않았다.  main method 에 depth 는 없다

"확증된 6D 개선" / "held-out 6D 개선"
    독립 확증 모집단이 존재하지 않는다
```

## 이번 결과에 붙는 필수 단서

```
모든 V1B 수치        PAPER_EVAL 319 는 개발에 반복 사용된 셋이다.
                    held-out · independent · final · confirmed 금지
C1 · L3             POST_STOP_EXPLORATORY_CORRECTION.  새 confirmatory study 가 아니다
ranking 구간         점추정을 본 뒤 계산됐다.  구간은 Tier B
positive-session     negative 쪽 변동을 덮지 않는다.  "session-clustered" 라고
  cluster 구간       단독으로 부르지 말 것
wood line 결과       line 모델의 wood 종횡비 노출량은 확인하지 않았다
lighting 부분모집단   manifest 가 319 중 120 장만 라벨한다.  나머지 199 는 UNLABELLED
```

## 아직 답이 없는 질문 — 없는 답을 지어내지 않는다

```
axis selector       실제 이미지에서 0.59~0.65, gate 는 0.95.  단일 프레임 기하는
                    소진됐다.  appearance 기반 단서 또는 다중 프레임 관측이 필요하다
독립 확증 모집단      존재하지 않는다.  이 논문의 어떤 수치도 확증이 아니다
closed-loop 평가     없다.  2D 픽셀 오차도 6D pose 도 삽입 성공률이 아니다
```
