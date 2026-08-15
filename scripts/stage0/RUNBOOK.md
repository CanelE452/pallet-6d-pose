# Stage-0 RUNBOOK — paper-track GPU 입구 (probe go/no-go)

목적: paper_base 모델로 **locked pl_pool** 에 PL 을 뽑고, **도메인별 통과량**으로
self-training probe 를 갈지(go) / base 보강으로 빠질지(no-go) 결정한다.
필터 τ 는 **filter-val 에서만** 캘리브하고 **final-test 는 절대 열지 않는다.**

> ⚠️ 이 폴더 스크립트 중 GPU 추론을 하는 건 1·3번. **작성자(에이전트)는 실행하지
> 않았다.** GPU 는 사용자가 돌린다. CPU-safe 부분만 스모크 점검됨(아래 표시).

## 봉인 (절대 입력 금지)
final-test 세션 = `capturepallet09, capturepallet07, capturenight09, capturenight08`.
- 1번은 `pl_pool_frames.txt`(pool 세션만) 를 입력 → final-test 미포함.
- 3번은 filter-val 세션 화이트리스트로만 GT 를 모으고, final-test 세션이 섞이면
  `assert` 로 죽는다.

---

## 실행 순서

### 1) PL 추출 — `extract_pl_v1.py`  (GPU)
```
conda run -n pallet-pose python scripts/stage0/extract_pl_v1.py \
    --weights weights/paper_base/paper_base/final_net_epoch_0060.pth \
    --output_dir data/pallet/pl/stage0_paper_base
```
- 입력: `weights/paper_base/paper_base/final_net_epoch_0060.pth`,
        `data/pallet/eval_results/split_lock/pl_pool_frames.txt` (8031 frames).
- 출력: `data/pallet/pl/stage0_paper_base/{frameid}.png` + `{frameid}.json`
        (NDDS, `source_model='paper_base_ep60'`, `filter_scores`={diag_pass,
        diag_score, flip_score, n_detected, min_conf}),
        `_records.json`(→ 3번 입력), `_manifest.json`.
- ★ diag/flip 은 **gate 가 아니라 score** 로 기록 (τ 는 2·3번에서 결정).
- CPU 스모크: `--dry_run` (모델 없이 frame→path 해석 + manifest).

### 2) PL 눈검사 — `pl_sanity_overlay.py`  (CPU) — ★ 카운트(3번) **전에**
```
python scripts/stage0/pl_sanity_overlay.py \
    --pl_dir data/pallet/pl/stage0_paper_base \
    --out_dir data/pallet/eval_results/stage0_pl_sanity --n 30 --flip_tau 10
```
- 통과 PL 무작위 20~30 장 cuboid overlay → 파렛트 없는데 헛검출한 PL 혼입 눈검사.
- 저장: `data/pallet/eval_results/stage0_pl_sanity/*.jpg`.
- ★ **여기서 헛검출 비율을 먼저 추정**한다. pool 엔 파렛트 없는/극단근접 프레임이 섞여
  있고, base 가 배경에 헛검출하면 중심 근처 점이 몰려 diag 를 통과할 수 있다(대각선
  일관성만으론 안 걸림). → 3번 N 을 이 비율로 **할인**해서 읽어야 go/no-go 가 정확.

### 3) ★ 통과량 카운트 — `pl_pass_count.py`  (CPU, go/no-go)
```
python scripts/stage0/pl_pass_count.py \
    --records data/pallet/pl/stage0_paper_base/_records.json \
    --flip_taus 5 8 10 15
```
- 출력(도메인별 outside/night/ALL): `detectable(n_det≥6) / diag / diag∧flip` N.
- **사용자가 GPU 돌린 뒤 보고할 숫자 = 이 표** (단 2번 헛검출 비율로 할인 후).
- **미리 박은 임계 (숫자 보고 즉석 고민 금지):**
  ```
  diag∧flip (헛검출 할인 후) 도메인당:
    ≥ 30   → probe GO     (R1 한 런이 통계적으로 말이 됨)
    10~30  → 애매 → probe 스킵, base 보강 직행 (어차피 base v2 에서 다시 봄)
    < 10   → probe 불가 확정 = base v2 선결조건 증명
             (= self-training 이 필요한 이유 = 현 base PL pool 고갈 → 논문 §11 정보)
  ```
  > loo sweep 추세대로면 **< 10 (no-go) 가 유력**. no-go 는 실패가 아니라 예측 적중 —
  > 다음이 base 보강 하나로 깔끔히 좁혀진다.

### 4) τ 캘리브 — `tau_calibrate.py`  (GPU, filter-val ONLY)
```
conda run -n pallet-pose python scripts/stage0/tau_calibrate.py \
    --weights weights/paper_base/paper_base/final_net_epoch_0060.pth
```
- 입력: filter-val GT (outside p08,02,03,04,05 / night n06,07,05) = 87 GT frames.
- 출력: `data/pallet/eval_results/stage0_tau_calibrate/tau_calibrate.json` —
        `τ_diag × flip_tau` grid 의 good%/gross/med/N.
- 선택 기준: **good 안 죽이면서 gross 거르는** 지점 (good% 높고 gross 작고 N 안 붕괴).
- CPU 스모크: `--list_val_frames` (쓸 GT frame 목록 출력 + final-test 미포함 확인).

---

## ⛔ SLOT — 자리만 잡음, 값은 사용자가 확정
self-training round 를 돌리기 전에 아래를 동결할 것 (`config/stage3_selftrain.yaml`):

- **synthetic : PL mixing 비율** — `training.synthetic_ratio` (현재 0.5).
  메모리 `cropaug-synthetic-ratio-tradeoff`: synthetic↑은 truncation 강건↑/clean 정밀↓.  → SLOT(미확정)
- **finetune epoch** — `self_training.epochs_per_round`(현재 3) × `num_rounds`(10).
  메모리 `dope-finetune-cumulative-epoch`: train.py finetune 은 **누적 epoch**
  (EPOCHS=base+추가분 절대값). probe 는 1 round 만 우선.  → SLOT(미확정)
- **checkpoint 규칙** — filter-val best vs last. (probe = 1 round 이므로 last 로
  충분할 수 있으나, 다중 round 시 filter-val best 로 잡아야 누수 없음.)  → SLOT(미확정)
- **eval 산출물 필드 규칙(manifest 교훈)** — eval json 에 `model` / `weights` /
  `detectable_def`(= n_detected≥6) 를 항상 박아 모델·정의 추적. 1·3번은 이미
  weights 를 박지만, self-train eval 산출물에도 동일 규칙 적용.  → SLOT(규칙만 확정, 적용은 학습 시)

## 재사용한 기존 모듈 (기하 재발명 0)
- `data_prep/eval/filter_pr_camfacing.py`: load_model, extract_keypoints_from_belief,
  canonical_kp3d, hungarian_mean_dist, filt_diag.
- `data_prep/eval/filter_flip_consistency.py`: infer_flip_kp, flip_consistency_score, FLIP_PAIRS.
- `data_prep/eval/dump_pl_overlay.py`: overlay (cuboid 그리기).
- split-lock 산출물: pl_pool_frames.txt, split_assignment.json.
