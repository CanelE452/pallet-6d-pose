# 그림 계획

> 정본은 `_docs/paper/final/FIGURE_PLAN.md` (Figure 1 파이프라인 / 2 주요
> trade-off / 3 라벨 품질 대 retention / 4 실패 모드) 이다.  네 그림 모두 유지한다.
> 이번 작업은 **새 그림을 만들지 않고**, 기존 그림에 붙일 데이터와 한 개의 부록
> 후보만 정한다.

## 기존 네 그림 — 변경 없음

이번에 새로 나온 것들(V1B · temporal · depth)은 전부 음성이거나 모집단 제한이라
주 그림의 메시지를 바꾸지 않는다.

## Figure 2 에 더할 수 있는 것 (선택)

Figure 2 는 "라벨 품질은 올라가는데 student 는 안 좋아진다" 는 trade-off 를
보인다.  이번 결과가 그 축을 **한 칸 더 밀 수 있다**.

```
현재     pseudo-label 품질  ->  student 2D 국소화        (개선 없음)
추가 가능 pseudo-label 품질  ->  student 2D  ->  6D pose  (역시 개선 없음)
```

같은 그림 안에 6D 를 세 번째 패널로 붙이면 "2D 만 보고 판단한 것 아닌가" 라는
반론이 그림 하나에서 닫힌다.

★ 2026-09-04 확정.  `_docs/paper/final/FIGURE_PLAN.md` 에 3 패널 계층으로 반영했다
(A 검출/랭킹 · B 2D 국소화 · C 하류 6D).  새 추론 없이 기존 frozen 결과만으로
그릴 수 있다.

## 부록 그림 후보 F-A1 — no-train formulation 스크린

한 장으로 여덟 arm 의 ΔIoU3D 와 ΔADDsym 을 세션 클러스터 구간과 함께 보인다
(V1 S1/S3/S4 + V1B C1/L2/L3/L4, 기준선 0 에 수직선).

```
왜 유용한가   "bbox 나 구조선을 붙여봤나" 라는 리뷰어 질문에 대한 답이 문단이
             아니라 그림 하나가 된다
왜 부록인가   POST-STOP 탐색이고 전부 음성이다.  본문 주장을 만들지 않는다
데이터 출처   paper_fast6d_screen_v1/screen/PAIRED_BOOTSTRAP.json
             paper_fast6d_screen_v1b/screen/PAIRED_BOOTSTRAP.json
```

## 모든 그림 공통 규칙

`_docs/paper/final/FIGURE_PLAN.md` 의 "Rules for all figures" 를 그대로 따른다.
추가로 이번 트랙 것에는 **개발 모집단**임을 캡션에 명시한다.
