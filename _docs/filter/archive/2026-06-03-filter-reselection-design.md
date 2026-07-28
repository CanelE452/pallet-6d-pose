# Filter Re-selection (데이터 많은 환경) — Design Spec

작성일: 2026-06-03
대상: DOPE self-training pseudo-label filter 재선정
관련: `_docs/filter/2026-04-11_selection.md`, `_docs/history/2026-05-28.md`

## 1. 문제 정의와 핵심 가설

### 4월 실패 메커니즘
- 4월 selection은 `capture0403middle` 440f 중 n_good 15~27개라는 **극소 GT**로 P/R 측정.
- 이 작은 데이터에서 RANSAC c≥6이 F1 1등(P=0.71, R=1.0)이라 채택. recall에 후한 선정.
- 실제 pool이 커지자(night 9134f 등) 같은 c≥6 조건이 절대 통과량 폭증 → FP 절대수 폭증.
- 5-28 발표에서 `ransac` 필터로 R1 학습 시 모든 도메인 R0 대비 하락(indoor -13.4pp 등).
- `ransac_loo`(strict)로 통과량을 줄이니 +12~38pp 회복. → "느슨한 필터 = 많지만 noisy한 PL = 학습 저하" 실측 증명.

### 핵심 가설
> 데이터가 많아진 환경에서는 필터의 **precision(통과 PL 순도)이 recall보다 압도적으로 중요**하다.
> 절대 PL 수량은 이미 충분하므로, 적게 통과시켜도 깨끗한 필터가 downstream에서 이긴다.

### 평가 기준 (4월과 다른 점)
- 4월: F1 1등 채택 (recall 후함).
- 이번: **통과 PL 수 ≥ 30 충족 후 precision 내림차순** 랭킹. 그리고 **P/R 랭킹이 실제 downstream 향상과 일치하는지 명시적 검증**(4월엔 이 검증을 안 해서 빗나감).

## 2. 결정 사항 (사용자 합의)
- **대상 파이프라인**: DOPE self-training (발표 셋업 확장)
- **평가**: 2단계 — P/R proxy 스크리닝 → 상위만 실제 R1/R2 downstream
- **범위**: 1차 R1 빠르게(3도메인 R0→R1) → 승자만 R2
- **필터 후보**: 기존 + 새 4방향 + 논문 서베이발 1~2

## 3. 필터 후보 라인업

### A. 기존 (baseline, 코드 존재)
| ID | 필터 | 역할 |
|----|------|------|
| `ransac` | RANSAC c≥6 (4월 선정) | 음성 대조군 (데이터 많을 때 망가짐 재현) |
| `ransac_loo` | RANSAC + LOO τ=0.05 | 양성 baseline (5-28에서 이긴 strict) |
| `cf_strict_v2` | canonical strict | 비교용 |

### B. 새 필터 (이번 구현)
| ID | 필터 | "많지만 noisy" 대응 |
|----|------|------|
| `topk` | quality 상위 top-K / 백분위 | reproj·consensus 점수 정렬 후 절대 수량 상한 |
| `strict_sweep` | c≥7/8 × reproj 3px × LOO 강화 | 기존 축을 더 빡세게 |
| `conf_geo` | DOPE belief peak conf × geometric consistency | 복합 게이트 |
| `adaptive` | per-domain 백분위 threshold | 도메인별 품질분포 차이 자동 반영 |

### C. 논문 서베이 (Phase 0)
- 검색: pseudo-label filtering / confidence-based selection / uncertainty for self-training / 6D pose self-training
- 찾을 아이디어: confidence calibration, agreement/consistency(TTA 일관성), curriculum threshold, domain-balanced selection, uncertainty(ensemble/MC-dropout)
- 산출: `_docs/filter/2026-06-02_survey_pseudolabel_filtering.md`, 4방향에 1~2 후보 추가

총 후보 ≈ 8~9개.

## 4. 2단계 평가 파이프라인

### Stage 1 — P/R proxy 스크리닝
- 도구: `filter_pr_eval.py` 확장 (도메인 인자화 + 새 필터 dispatch)
- GT 평가셋: indoor=`capture0403middle`(440f, AprilTag), night=`_eval_sets/night_combined`(90f, manual)
- **outside는 GT 부재** → Phase 0에서 `capture0403noapril` GT 확인, 없으면 Stage1 생략 후 Stage2에서만 평가
- 선정 규칙: 통과 수 ≥30 후보 중 precision 상위 2~3 + baseline `ransac_loo` + 대조군 `ransac`

### Stage 2 — 실제 downstream
- 선정 ~5필터 × 3도메인 R0→R1 학습 (5-28 명령 패턴)
- 지표: NN<20px per-frame %(메인) + PCK + reproj
- 매트릭스: 필터 × 도메인 R1 표

### 핵심 검증 (4월 교정)
- Stage1 P/R 랭킹 ↔ Stage2 downstream 향상 상관 산점도
- P/R 1등이 downstream에서 지면 "P/R proxy 신뢰 불가, downstream이 정답" 명시 문서화

### outside GT 부재 처리
- Stage1에서 outside 못 재면 Stage2(R1 NN<20px)로만 판정. indoor/night 승자가 outside R1에서도 이기는지 cross-check.

## 5. 실험 매트릭스·산출물·일정

### Phase 0 (반나절)
- 논문 서베이 문서
- 인벤토리(PL pool·GT·anchor weight), outside GT 확정
- `filter_pr_eval.py` 도메인 인자화 + dispatch, `geometric_filter.py` 새 필터 4종 구현

### Phase 1 — Stage1 (반나절)
- 후보 × {indoor, night} P/R → 선정 ~5개
- 산출: `filter_pr_screening.csv` + P/R scatter

### Phase 2 — Stage2 R1 (1~1.5일)
- ~5필터 × 3도메인 R0→R1 + 평가
- P/R↔downstream 상관 산점도
- 산출: R1 매트릭스 표 + 상관 그림

### Phase 3 — R2 승자 (반나절)
- R1 best 1~2필터 × 3도메인 R2 → 발표용 풀 매트릭스

### 최종 산출물
- `_docs/filter/2026-06-03_filter_reselection.md` — 전체 결론
- 필터 × 도메인 × R0/R1/R2 매트릭스 표 (5-28 스타일)
- P/R vs downstream 상관 그림, 필터별 향상 막대그래프
- 갱신: `config/stage3_selftrain.yaml`, `_docs/filter/README.md`, history

## 6. 비목표 (YAGNI)
- YOLO26-pose 파이프라인 필터는 이번 범위 밖
- learning-based quality estimator는 long-term, 이번 제외
- 4월 23개 후보 전수 재비교 안 함 (선별된 8~9개만)
