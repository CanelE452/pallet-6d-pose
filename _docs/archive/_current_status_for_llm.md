# 현재 상황 — fable5 핸드오프용 (2026-06-10 세션)

> 이 섹션은 **이번 세션에서 verify 로 확인한 실제 상태**다. 아래 docs 전문(master / metric_split_lock / _docs/*)보다
> **우선**한다 — docs 일부(특히 master §0·§3.6·§11)는 이번 검증으로 stale 판정됨. 충돌 시 이 섹션이 최신.

## ⛔ 지금 막혀있는 핵심 (먼저 읽을 것 — 이걸 풀고 싶어 fable5 에 묻는 것)

**한 줄: 시도 중인 geometry-filtered self-training 이, honest held-out base 가 real 에서 너무 약해서
믿을 PL 도 깨끗한 final-test 도 안 나와 막혀 있다.**

```
시도하는 것: CoordDOPE 합성 base → 2D 기하 필터로 strict pseudo-label 선별 → self-training 으로
            real 도메인 적응(처음 본 파렛트 일반화). 필터의 main contribution 으로 논문화.

막힌 지점 (전부 이번 세션 verify 로 확정):
─────────────────────────────────────────────────────────────────────────────
[P1] base 가 honest 로 너무 약함 (★ 근본 원인)
     held-out paper_base 의 real GT 성능: good(<10px) = outside 9 / night 1 / forklift 4.
     detectable(≥6kp) 도 outside 51/129, night 42/90 (검출 ~40-50%). indoor 더 붕괴(440중 353 <6kp).
     → self-training PL 소스가 이 약한 base 라, 필터가 골라낼 "깨끗한 PL" 자체가 거의 없음.
       (메모리도 일관: paper_base good% ~6-7%, threshold sweet spot 없음, 필터 천장=base 코너정확도.)
     → 필터를 "고품질 선별(selection)" 못 팔고 "구조적 무효 PL 억제(suppression)"로 후퇴한 이유가 이것.

[P2] 기존 self-training(R1) 이 누수 — inductive claim 못 함
     paper_r1 이 unlabeled 풀 전체에서 뽑은 696 PL 로 학습됨. GT 세션이 전부 그 풀 안(7/7, 6/6).
     → final-test 세션을 R1 이 (PL 통해) 봤을 수 있음 = transductive 만 유효, inductive 무효.
     → 풀려면 final-test 세션 정하고 그 세션 빼고 R1 재학습해야 함(아직 안 함).

[P3] 토대 의심 — paper_base 가 폐기된 v8 데이터로 학습됐을 가능성 (★ 최우선 확인)
     weights/paper_base/paper_base/header.txt 의 학습 data = mixed_v8_train + aug_*.
     "v8"=폐기된 object-frame v8(잘못된 convention)이면 camera-facing 전제가 통째로 깨짐.
     이름만 남은 건지 실제 v8 데이터인지 아직 미확인.

[P4] 평가 메트릭이 깨짐
     real 파렛트 치수 미지(canonical PnP) + 좌표계 문제로 ADD 불가 → PnP success+reproj 로 대체 중.
     honest good 이 희소(9/1)해서 어떤 메트릭을 써도 숫자가 약하게 나옴.

핵심 질문(fable5 에게): 이 약한 base + 누수 상태에서
  (a) self-training thesis 를 살릴 수 있나, 아니면 thesis/평가를 어떻게 재설계해야 하나?
  (b) P3(v8 의심)부터 어떻게 검증·격리하나?
```

---

## 0. 프로젝트 한 줄

팔레트 6D pose(=9 keypoint) monocular RGB 추정. CoordDOPE 합성 supervised → 2D 기하 필터로 strict
pseudo-label 선별 → self-training(UDA). 논문 트랙(camera-facing 0123, v1/v2 제외 일반화) vs 과제 트랙(v1/v2 과적합).
**canonical 전략 = repo 루트 `paper_strategy_master.md` + `metric_split_lock.md`.** 이번 세션은 그 master 기준으로
docs 정렬 + 실제 데이터 검증을 진행 중.

---

## 1. 이번 세션에서 한 것 (DONE)

```
✅ step0  metric_split_lock.md      Downloads → repo 루트 배치 (재생성 아님, lock 버전)
✅ step0  session_inventory_v2.py   원본(Codex 세션) 분실 확정 → 문서화 스펙으로 재구성 + 패치 + 실행
✅ B1     _docs/experiments/related_work.md   master §10 으로 정렬 (옛 RANSAC/UDA-COPE/PseudoFlow·
                                              "10K vs 30-50K" 제거, 2.1/2.2/2.3 구조, Self6D++=reference)
✅        _docs/_docs_todo.md       master=canonical 규칙(복붙금지·가리키기) + Track1/Track2 분리
✅        memory                    paper-strategy-master-canonical 기록
```

남은 docs 작업(Track1, 미완): B5 evaluation, B2 overview, B4 step3, B3 step2(필터=3d-expert 위임),
C1 de-risk 4-arm, C2 gate-1 matrix, C3 quality-quantity, C4 README. → `_docs/_docs_todo.md` 참조.

---

## 2. 진짜 데이터로 verify 한 결과 (★ master §3.6·§11 을 뒤집음)

### 2.1 세션 구조 (확정)
```
unlabeled 풀 = 번호 캡처 폴더 (dated 아님). 각 폴더 = cam_K.txt + rgb/ + depth/
  outside: data/outside/capturepallet01~11  (06=빈 폴더 → 실질 10세션, rgb 8715장)
  night:   data/night/capturenight01~10      (10세션, rgb 9134장)
  forklift: data/outside/forklift_raw_20260528_163408 (얘만 dated)
  capturepalletcad: data/outside/capturepalletcad (rgb 1179장) — 실제 야외 사진(렌더 아님)
GT 평가셋 = data/_eval_sets/{outside_combined(129), night_combined(90)} — 평탄화돼 세션정보 상실.
  → frame_id 로 raw 세션 역추적해야 함 (session_inventory_v2.py 가 수행).
master §3.6 의 "outside 9894 / night 9134" = rgb+cad 포함 카운트. capturepallet 만이면 outside 8715.
```

### 2.2 honest 성능 (held-out paper_base, `per_frame_heldout_pretrain.json`, json 자체 good 플래그와 일치)
```
도메인     GT    detectable(≥6kp)   good(<10px)   master §3.6 주장      판정
outside    129   51                 9             "31 good"            §3.6 stale (낙관치)
night      90    42                 1             "30 good"            §3.6 stale
forklift   32    26                 4             —                    —
```
- **master §3.6 의 good 31/30 은 틀림. honest 는 9/1.** 그 31/30 은 ft_s2(누수) 또는 다른 metric 산물.
- **night 은 good 이 전 도메인 통틀어 단 1장.** indoor 도 검출 붕괴(별도, master §4: 440 중 353 <6kp).
- 이게 이 프로젝트의 핵심 난점: **honest held-out base 가 real 에서 매우 약함**(검출 ~40-50%, good ~7-9%).
  메모리 [ransac-loo-sweep-paperbase-no-sweetspot], [diag-filter-not-reliable], [flip-consistency] 와 일관 —
  "필터 천장 = base 코너 정확도. paper_base 직접 PL 소스 부적합."

### 2.3 split 판정 (정정됨)
- session_inventory 의 초기 출력은 BRANCH C(good 부족) 였으나 **판정 기준이 틀렸음**:
  final-test 적격 = good 이 아니라 **detectable**(모델이 검출해 GT 대비 오차 잴 수 있는 프레임).
- detectable = outside 51 / night 42 = 충분 + 다세션 → **session-level split(branch A) 가능.**
- good 희소(9/1)는 split blocker 가 아니라 **"base 가 약하다"는 결과·리스크 story** (master §11 리스크 확정).

### 2.4 누수 지형 + 기존 학습 상태 (★ critical)
```
paper_base = 이미 학습 완료   weights/paper_base/paper_base/final_net_epoch_0060.pth (06-06)
paper_r1   = 이미 학습 완료   weights/paper_r1_{outside,night}/final_net_epoch_0091.pth (06-06)
  → master §0·§11 "paper_base 미학습 = 단일 실패점" 은 stale. 학습 대기는 사라짐.

GT 세션 전부 unlabeled 풀 안 (outside 7/7, night 6/6 세션 overlap).
paper_r1_outside = output/pl_paper_r1_outside 의 696 PL 로 finetune (풀에서 필터 통과분).
  → 기존 R1 은 final-test 세션을 PL 통해 봤을 수 있음 = inductive 무효, transductive(appendix §3.2)만 유효.
  → inductive 하려면: final-test 세션 선택 → 그 세션 PL 제외 → R1 재학습(1라운드). 다세션이라 가능.
```

---

## 3. 지금 안 되는 것 / 미해결 (BLOCKER & 리스크)

```
🔴 base 가 honest 로 약함        held-out paper_base: good outside 9 / night 1 / forklift 4.
                                 self-training 헤드룸·필터 천장이 여기서 결정됨. (가장 큰 리스크)
🔴 기존 R1 = transductive only   696 PL 이 풀 전체에서 와서 final-test 세션 누수 가능 → inductive 위해 재학습 필요
⚠ paper_base 데이터 = mixed_v8_train  header.txt 의 학습 data 가 mixed_v8_train + aug_{squash,trunc,scale}.
                                 "v8"=폐기된 object-frame v8 인지, 이름만 남은 camera-facing 데이터인지 미확인.
                                 ★ 만약 진짜 object-frame v8 이면 camera-facing 전제가 깨짐 — 최우선 확인 대상.
⚠ paper_r1_outside NaN/Inf       학습 로그 epoch 61 에 "NaN or Inf found in input tensor" — 불안정.
⚠ capturepalletcad 처리 미정     GT 129 중 22장이 cad 세션(unmatched 22 = cad∩GT 22). cad=내 CAD 파렛트면
                                 "unseen 일반화" claim 과 충돌. 빼면 GT 22장 손실 → 사용자 결정 필요.
⚠ session_inventory 원본 분실     Codex 세션에서 만든 v2 가 이 PC 에 없어 재구성. 원본 복구 시 diff 대조 권장.
○ Track1 docs 정렬 미완           B2/B3/B4/B5/C1~C4 (master 가리키게). blocker 아님(백그라운드).
```

## 4. 지금 사용자에게 물어둔 결정 (열림)
1. **mixed_v8_train 정체 확인** — paper_base 가 폐기 v8 데이터로 학습됐는지. (제일 급함, 나머지의 전제)
2. **split 방향** — detectable 기준 session-level(branch A) + final-test 세션 R1 재학습으로 갈지, 아니면
   good 희소 감안해 evaluation metric 자체 재설계(detectable 위 honest error 보고).
3. **capturepalletcad** — GT 22장 포함한 채 둘지, 빼고 GT 107 로 갈지.

## 5. master 갱신 정책
- §0·§3.6·§11 stale 확인됐으나 **지금 고치지 않음.** session inventory 결과 + 누수 판정 + 확정 split 까지
  받아 "paper_base/R1 완료 + honest 수치 + 누수 판정 + split lock" 을 **한 번에** 반영(두 번 고치지 않기).

---
---

> 아래부터는 docs 전문 (요약 아님). 순서: paper_strategy_master.md → metric_split_lock.md → _docs/* 전체.
> ⚠️ docs 의 stale 부분은 위 §2~§3 이 최신. 특히 master §0·§3.6·§11, 그리고 _docs 의 "good 31/30" 류 수치.
