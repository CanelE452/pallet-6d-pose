# framing 결정

## 결정 한 줄

> **실험 탐색을 끝낸다.  논문은 "개선했다" 가 아니라 "무엇이 옮겨가지 않는지를
> 측정했다" 로 쓴다.**

## 왜 지금 끝내는가 — 소진의 근거

같은 벽에 서로 다른 다섯 방향에서 부딪혔고, 전부 같은 답이 나왔다.

```
방향                          시도                                 결과
──────────────────────────────────────────────────────────────────────────────
선택 규칙                     V1~V5 (naive · confidence · reproj ·  2D 미세 국소화
                             removal · full consistency · 가중치)   개선 없음
teacher 강화                  FAST A/B/C, STRONG teacher            개선 없음
pose 계산단 기하 신호          V1 S1·S3·S4 (translation refit ·      전부 음성.
                             bbox gate · square ROI)               S3 는 선택을 0 개 바꿈
                             V1B C1 (관측 semantics bbox)          -0.066, 더 나빠짐
                             V1B L2·L3·L4 (line 융합, 두 seed)      전부 음성
추가 센서 · 시간축             depth Gate 0/0B/센서검증               NOT_READY_FOR_GATE1
                             temporal pilot + closure              적격 모집단 0
현장 정합 적응                 A8_DAY_ONLY 를 site 정합 88 프레임에서   해소된 개선 없음.
                             평가 (recording cluster 7)          네 지표 모두 구간이 0 포함
```

여기에 더해 **모집단이 소진됐다**.  PAPER_EVAL 319 는 다섯 선택 트랙, 세 teacher
probe, 두 no-train 스크린이 모두 개발용으로 소비했다.  같은 셋을 상대로 설계한
어떤 변형도 확증 증거가 될 수 없다 — `EXPERIMENT_STOP_LOCK.json` 이 이미 그렇게
적어 두었고 이번 작업은 그 판단을 뒤집을 이유를 찾지 못했다.

## 핵심 framing 문장

> Pseudo-label reliability does not necessarily translate into fine geometric
> localisation or downstream 6D pose.

artifact 가 지지하는 범위는 `PAPER_MAIN_CLAIMS.md` 마지막 절에 정확히 적었다.
요약하면 **앞의 세 마디는 측정됐고, "왜" 는 해석이다.**

## 논문이 실제로 기여하는 것 세 가지

```
1  합성 감독만으로 실제 도메인 팔레트 키포인트 추정이 이미 강하다는 것을
   검출·랭킹·2D 국소화·6D pose 네 층에서 한 계약 아래 측정했다

2  가짜 라벨의 **신뢰도를 높이는 것**과 student 의 **미세 기하 정확도**가 분리된다는
   것을, 선택 규칙 다섯 · teacher 세 · 계산단 기하 여덟 갈래로 반복 확인했다.
   부정 결과이되 좁게 재현된 부정 결과다

3  근사 정사각 산업용 팔레트에서 단일 프레임 기하는 footprint 축을 못 가른다는
   구체적이고 이전 가능한 발견 — 실패가 깨끗한 90도 교환이고 비용이 회전 85도다.
   selector 실측 0.59~0.65, gate 0.95
```

## 결정 상태 — 2026-09-04 canonical sync 로 전부 해소

```
D1 ★ pose 상태          RESOLVED.  POSE_METRICS_STATUS = REPORTABLE,
                       can_claim_6d_improvement = false.  6D 표는 본문에 들어간다.
                       PAPER_CLAIM_LOCK.json 은 amendment 됐고 historical first pass 는
                       삭제하지 않고 보존했다

D2   ranking 구간        PARTIALLY RESOLVED.  frame-level 구간은 계산했다
                       (R5-R0 AUROC +0.00318 [+0.000092, +0.006898], 0 배제).
                       session-cluster 구간은 negative 행에 session_id 가 없어
                       여전히 계산 불가 — 이건 데이터의 성질이지 미결정이 아니다

D3   site-matched       RESOLVED.  소규모 arm(A8_DAY_ONLY)은 **이미 평가됐다**
                       (88 프레임, recording cluster 7).  네 지표 모두 구간이 0 포함.
                       full-site 2,227 수량 확대 학습은 NOT_RUN_AND_NOT_PLANNED.
                       "유일하게 남은 실험" 이라는 표현은 폐기한다

D4   wood pose          RESOLVED.  wood 125 는 pose 표에 **포함**된다.
                       평가 전용 POSE_EVAL_OBJECT_CONTRACT 를 쓰므로
                       OBJECT_GEOMETRY_REGISTRY.json 재발행은 하지 않는다.
                       ALL 319 = plastic 194 + wood 125

D5   6D 그림            RESOLVED.  Figure 2 를 3 패널 계층으로 그린다
                       (A 검출/랭킹 · B 2D 국소화 · C 하류 6D).
                       새 추론 없이 기존 frozen 결과만으로 그릴 수 있다
```

새 미결정 사항은 **artifact 부족 두 건**뿐이고, 둘 다 계산이 아니라 데이터의 성질이다.

```
BLOCKED_MISSING_ARTIFACT   ranking 의 완전한 session-cluster 구간
                           (negative 2,689 행에 session_id 없음)
BLOCKED_MISSING_ARTIFACT   필터 품질 지표의 신뢰구간
                           (FILTER_SEPARABILITY.json 에 항목별 배열 없음)
```

## 다음에 하지 말아야 할 것

`PAPER_NO_CLAIMS.md` 의 새 금지 목록과
`FAST_6D_SCREEN_V1B_LOCK.json` 의 `forbidden_after_seeing_results` 를 따른다.
특히 **실패한 arm 을 파라미터 탐색으로 구조하지 않는다** — line lambda, bbox
margin, DOPE padding, crop 비율 전부 해당한다.
