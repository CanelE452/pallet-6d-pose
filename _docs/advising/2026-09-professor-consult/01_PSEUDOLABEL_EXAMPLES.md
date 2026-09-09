# 01 — pseudo-label 이 잘 된 예 / 안 된 예

## 목적

숫자만으로는 "필터가 무엇을 통과시키는지" 가 안 보인다.  통과·기각된 라벨을
GT 와 겹쳐 그려 실패 모드를 눈으로 확인한다.

## 어떤 모집단에서 뽑았나 — 그리고 왜

[확인] **PAPER_EVAL plastic positive 194 장**에서 뽑았다.  self-training pool 이 아니다.

```text
pool 프레임        manual GT 가 없다 -> "잘 됐다/안 됐다" 를 눈대중으로만 말할 수 있다
PAPER_EVAL 194    manual GT 가 있다 -> GT 대비 실제 픽셀 오차로 good/bad 를 정의할 수 있다
```

[확인] `M4_FRAME_RECORDS.json` 은 **같은 teacher**(R0, sha 970a0913...)와
**같은 frozen 필터**(lock sha 57c9b939...)를 GT 있는 194 장에 적용한 기록이다.
그래서 여기서 뽑은 예시는 self-training 이 실제로 쓴 것과 같은 teacher·같은 규칙이다.

★ 한계: 학생이 실제로 학습한 pool 프레임 그 자체는 아니다.  같은 teacher·같은 필터의
**대리 모집단**이다.  이 점을 상담에서 먼저 말할 것.

## 선정 규칙 (눈대중 아님)

보여주는 필터는 논문의 proposed 인 **F4_PROPOSED** 다.
194 장 중 검출 194, 통과 142, 기각 52.

```text
good_accepted      통과 · GT 대비 최대 코너 오차가 가장 작은 6 장
bad_accepted       통과 · GT 대비 최대 코너 오차가 가장 큰 6 장        ★핵심
rejected_correct   기각 · 최대 오차가 가장 컸던 6 장                   (기각이 옳았다)
rejected_costly    기각 · 최대 오차가 가장 작았던 6 장                 ★기각의 비용
```

**왜 median 이 아니라 max 인가.**  주된 실패는 90도 축 순열이라 median 은 작게 남고
한 코너만 폭발한다.  median 으로 고르면 이 실패를 놓친다.

## 뽑힌 범위

```text
범주                장수   GT 대비 최대 코너 오차 범위
──────────────────────────────────────────────────────
good_accepted        6      3.8 ~ 4.4 px
bad_accepted         6     96.9 ~ 315.7 px
rejected_correct     6    263.9 ~ 425.2 px
rejected_costly      6      5.9 ~ 10.1 px
```

## 그림 읽는 법

```text
초록      manual GT cuboid
노랑/파랑  teacher 예측 (노랑 = 앞면 0-3, 파랑 = 뒷면 4-7, 회색 = 깊이 변)
빨강 선    GT 와 예측이 가장 많이 어긋난 코너
배너      frame id · GT 대비 median/max 오차 · 20px 초과 코너 수 ·
          box_conf · s_reproj · s_remove · s_flip  (전부 기록된 값)
```

## 예시 파일

```text
figures/pseudolabel_good_accepted_01..06.png
figures/pseudolabel_bad_accepted_01..06.png
figures/pseudolabel_rejected_correct_01..06.png
figures/pseudolabel_rejected_costly_01..06.png
manifests/PSEUDOLABEL_EXAMPLES.csv    (프레임별 점수·오차 전부)
manifests/PSEUDOLABEL_EXAMPLES.json
```

## 관찰 요약

- [확인] **통과했는데 틀린 라벨이 존재하고, 그 틀린 방식이 90도 축 순열이다.**
  대표 예: `eval_pallet07:1778652166837872128` — box_conf 0.927, s_reproj 0.0169,
  s_remove 0.0233, s_flip 0.0499 로 **모든 기하 점수가 작은데** GT 대비 median 267.9 px.
  상자는 팔레트를 정확히 감쌌고 코너 라벨만 돌아갔다.
- [확인] good 쪽은 GT 와 예측이 거의 겹친다 (max 3.8~4.4 px).  필터가 맞을 때는 확실히 맞다.
- [확인] 기각된 52 장 중에도 max 5.9~10.1 px 짜리 정확한 라벨이 있다 — 기각에는 비용이 있다.
- [확인] 필터 전체 성능은 precision 0.5915 / recall 0.8235 (`M4_FILTER_QUALITY.json`).
  즉 **통과시킨 것의 약 41% 가 20px 기준으로 부정확하다.**

## 해석 주의점

- [추정] "기하 점수가 작은데 틀렸다" 는 것은 현재 필터가 쓰는 세 신호
  (reprojection / keypoint-removal / flip consistency)가 **90도 축 순열에 둔감**하다는
  뜻으로 읽힌다.  이건 해석이고, 신호별 분리 실험으로 증명한 것은 아니다.
- [확인] 이 그림들은 **논문 주장이 아니라 진단**이다.  본문 성능 표에 넣지 않는다.
- [확인] precision 0.5915 는 194 장 plastic 기준이다.  319 장 표와 섞지 말 것.
