# Docs 마이그레이션 TODO — master 기준 정렬

> 생성: 2026-06-08 | 트랙: camera-facing 0123 (논문용, v1/v2 제외)
>
> ## ⚠️ Canonical 규칙 (drift 방지 — 필독)
> - **canonical 결정 = `paper_strategy_master.md` (repo 루트) + `metric_split_lock.md` (frozen protocol).**
> - 아래 docs를 고칠 때 **lock된 결정(thesis/claim/split/metric/baseline/venue)을 복붙하지 말 것.**
>   대신 master의 해당 섹션을 **가리켜라** (예: "split protocol은 `metric_split_lock.md §3` 참조").
> - 복붙하면 master가 바뀔 때 docs가 drift남. docs는 "설명/맥락", master는 "확정값". 역할 분리.
> - `[LOCKED]` 값을 docs에서 재서술해야 할 때는 반드시 출처(`master §x`)를 같이 적는다.

---

## Step 0 — lock된 파일 repo 배치 (공짜, 모든 게 이걸 참조)

- [x] `metric_split_lock.md` → repo 루트 배치 (2026-06-08, Downloads에서 복사. 재생성 아님)
- [x] `session_inventory_v2.py` → repo 루트 배치 (2026-06-08). 원본(Codex 세션)이 이 PC에
      분실 확정 → "재생성 금지" 전제 깨짐 → 문서화된 v2 스펙(numeric-stem 제외 / metrics 기반
      viability·A? / 보수적 signal+singleton 폐기 / frame∩·session∩ 분리 / decision rule)으로
      **재구성**. 4개 수정분 검증 완료. 원본 복구 시 diff 대조 권장.
      ⚠️ **실행 전 CONFIG 경로 수정 필요** — `DOMAINS` 의 unlabeled 풀 경로(outside 9894/
      night 9134)가 추정값이라 실제와 안 맞음. 실제: indoor=`data/pallet/raw_data/capture0403*`,
      forklift=`data/outside/forklift_raw_*`. unlabeled 대용량 풀 위치는 사용자 확인 필요.

---

## Track 1 — 문서 마이그레이션 (에이전트 위임 가능, 병렬)

> 데이터/GPU 불필요. 텍스트만으로 닫힘. master를 가리키게(복붙 금지).

### 게이트 (충돌 명확 → 먼저)
- [x] **B1** `_docs/experiments/related_work.md` — (2026-06-08 에이전트 완료, 검증됨)
      옛 RANSAC/23-candidate/"~10K vs 30-50K" 제거 → master §10 세트 + 2.1/2.2/2.3 분리구조 +
      Self6D++=reference. 본 연구 수치는 master 가리킴(복붙 안 함). 남은 [SLOT]: 각 논문 서지정보
      + venue IF 수치(최신 JCR) — 사용자가 채울 것.
- [ ] **B5** `_docs/method/evaluation.md` — PnP A/B/C만 있음 → metric battery 4-layer
      (detection/keypoint/pose/operational)는 `metric_split_lock.md §4` 가리킴 +
      SQPnP config `§5` + domain dims W/D swap 주의.

### 서사 정렬
- [ ] **B2** `_docs/method/overview.md` — thesis를 master §1.2(selection 아닌
      **suppression**) + §1.3 역할분담표(convention=enabler, 필터=main, ST=main experiment)에
      맞춰 갱신. 값은 master 가리킴.
- [ ] **B3** `_docs/method/step2_geometric_filter.md` — "3d-expert 위임 예정" 상태 해소.
      diag/ratio/size-aspect 구체화 + A4(naive geometry) 방어 + suppression claim
      (precision 불허, gross-reject 72%) 반영. 임계값은 `metric_split_lock.md` 가리킴.
- [ ] **B4** `_docs/method/step3_selftraining.md` — master §8 R2-collapse를 hard-PL에만
      한정 + Mean Teacher 실패모드 분리.
- [ ] **B6** `_docs/models/paper_base.md` — master §13 SLOT 고정값(synthetic:PL mixing/
      epochs/input res/per-domain W·D·H) 확정되면 반영. (값 확정 전엔 SLOT 표기 유지)

### 새 실험 문서 (master에 있으나 experiments/ 인덱스에 없음)
- [ ] **C1** de-risk 4-arm (master §9, outside R0→R1) — thesis 생사 사전점검.
- [ ] **C2** gate-1 baseline matrix (master §6, A0~A9) — 기존 `filter/B2_filter_selftraining.md`
      확장/대체.
- [ ] **C3** quality-quantity sweep 곡선 = main figure (master §7) — 기존 C2를 "한 점→곡선"
      재정의(gate-1과 병합).
- [ ] **C4** `_docs/experiments/README.md` 인덱스 갱신 (C1~C3 + venue §2 반영).

---

## Track 2 — 진짜 critical path (사용자만 가능, 데이터/GPU 필요)

> 문서 고치는 건 진도처럼 느껴지나 blocker는 이 둘로만 풀린다.

- [ ] **session_inventory_v2.py 실제 데이터에 실행** → master §3.6 채우기
      (outside/night/indoor/forklift 세션수 + unlabeled∩test 겹침) → split 분기(A/B/C) 확정
      → Table 1 close. (단 step 0의 A2 파일 확보가 선행)
- [ ] **paper_base 학습 상태 확인** — 멈춰있으면 A1~A3·B1재확인·C1·C2·D1·D2 전부 직렬 정지.
      master §11 "단일 실패점".

---

## 진행 순서 (요약)
```
step0: metric_split_lock.md 배치[done] + session_inventory_v2.py 확보[blocker]
  │
  ├─ Track1 (에이전트): B1 → B5 → B2/B3/B4 → C1~C4   ← 병렬로 돌림
  └─ Track2 (사용자):   세션 인벤토리 실행 + paper_base 상태 확인  ← 실제 blocker
```
