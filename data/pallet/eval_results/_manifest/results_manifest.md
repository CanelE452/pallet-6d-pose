# Eval Results Provenance Manifest

생성: 2026-06-15 (사실/숫자 정리용. 가치판단 없음).
범위: `data/pallet/eval_results/` 하위 결과 디렉토리 — 어느 모델 × 어느 스크립트 × detectable 정의에서 나왔는지 귀속.

## ★ Provenance 정정 2줄 (사용자 지정)

1. "논문용 best = ft_s2" 는 **오귀속**. `dope_cropaug_ft_s2` 는 과제(challenge) 트랙 + 누수(leak) 모델이다.
   논문용 base = **`weights/paper_base/`**. (메모리: v1v2-challenge-only, dope-cropaug-truncation-success 참고.)
   → 아래 표에서 model 이 `dope_cropaug_ft_s2` / `*_s2` 인 결과(filter_pr_camfacing s2, filter_domain_analysis s2,
   filter_loo_sweep s2, filter_combo_9kp/perkp s2, pl_gt_diff s2)는 논문 base 숫자가 아님.

2. `aug_trunc` / `aug_scale` 데이터의 **3D 필드(cuboid/pose)는 stale** 하다. DOPE 로더는 `projected_cuboid`(2D)만
   학습에 쓰므로 **학습 레이블엔 무해**. 단 그 3D 필드를 평가/PnP 입력으로 쓰면 안 됨 (기록만).

---

## 결과 manifest 표

```
result_path                                              model(weights)                                 script                                              detectable_def                  비고
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
eval_results/eval_summary.json                           UNKNOWN (필드 없음; 마지막 evaluate_on_val 실행) evaluate_on_val.py                                  corner>=4 (미만 PnP/PCK skip)    weights 미기록. PCK order-free + PnP + Volume. cx/cy/fx 기본값(615,320,240). num_frames=200
eval_results/calibration_results.json (+_histdata.npz)   해당없음(추론 아님; 카메라 캘리브)              (calibration 도구; eval 스크립트 아님)              N/A                              카메라 intrinsic 캘리브 산출물
eval_results/phase1_R0/*.txt                             v8_ablation_A_coord/final_net_epoch_0065.pth    eval_nn_matching.py                                 NN-match: 매칭된 valid pred kp    ★v8 폐기 트랙 모델. per-kp & per-frame(denom=129). thresh=0.3
eval_results/phase1_R0_6d/*.txt                          v8_ablation_A_coord/final_net_epoch_0065.pth    (6D eval; "Loading model" 헤더)                     6D/PnP                           ★v8 폐기 트랙
eval_results/phase1_R0_camfacing/*.txt                   challenge_camfacing_scratch/final_net_epoch_0060 eval_nn_matching.py                                 NN-match 매칭 valid kp            challenge(camfacing) 트랙
eval_results/phase1_R0_challenge0123/*.txt               challenge0123/final_net_epoch_0060.pth          eval_nn_matching.py                                 NN-match 매칭 valid kp            challenge 트랙
eval_results/phase1_R0_challenge0123_6d/*.txt            challenge0123/final_net_epoch_0060.pth          (6D eval; "Loading model" 헤더)                     6D/PnP                           challenge 트랙
eval_results/phase1_R1_outside/*.txt                     r1_outside_ransac/final_net_epoch_0096.pth      eval_nn_matching.py                                 NN-match 매칭 valid kp            self-train R1 (ransac 필터)
eval_results/phase1_R1_outside_loo/*.txt                 r1_outside_loo/final_net_epoch_0096.pth         eval_nn_matching.py                                 NN-match 매칭 valid kp            R1 loo 필터
eval_results/phase1_R1_outside_loo_6d/*.txt              r1_outside_loo/final_net_epoch_0096.pth         (6D eval)                                           6D/PnP                           R1 loo
eval_results/phase1_R1_outside_loo_FIX_6d/*.txt          r1_outside_loo/final_net_epoch_0096.pth         (6D eval)                                           6D/PnP                           R1 loo (FIX 변형, 같은 weight)
eval_results/phase1_R1_outside_cf_strict/*.txt           r1_outside_cf_strict/final_net_epoch_0080.pth   eval_nn_matching.py                                 NN-match 매칭 valid kp            R1 cf_strict ep80
eval_results/phase1_R1_outside_cf_strict_ep70/*.txt      r1_outside_cf_strict/net_epoch_0070.pth         eval_nn_matching.py                                 NN-match 매칭 valid kp            R1 cf_strict ep70
eval_results/phase1_R1_night_loo/*.txt                   r1_night_loo/final_net_epoch_0096.pth           eval_nn_matching.py                                 NN-match 매칭 valid kp            R1 night loo
eval_results/phase1_R1_night_cf_strict_ep70/*.txt        r1_night_cf_strict/net_epoch_0070.pth           eval_nn_matching.py                                 NN-match 매칭 valid kp            R1 night cf_strict ep70
eval_results/phase1_R1_indoor_cf_strict/*.txt            r1_indoor_cf_strict/final_net_epoch_0080.pth    eval_nn_matching.py                                 NN-match 매칭 valid kp            R1 indoor cf_strict
eval_results/phase1_R1_indoor_F5/*.txt                   f5_noapril_ransac_loo_realonly/final_net_ep0096 eval_nn_matching.py                                 NN-match 매칭 valid kp            F5 변형
eval_results/phase1_R2_outside_loo/*.txt                 r2_outside_loo/final_net_epoch_0127.pth         eval_nn_matching.py                                 NN-match 매칭 valid kp            self-train R2 loo
eval_results/phase1_R2_outside_cf_strict_ep75/*.txt      r2_outside_cf_strict/net_epoch_0075.pth         eval_nn_matching.py                                 NN-match 매칭 valid kp            R2 cf_strict ep75
eval_results/phase1_R2_indoor_loo/*.txt                  r2_indoor_loo/final_net_epoch_0127.pth          eval_nn_matching.py                                 NN-match 매칭 valid kp            R2 indoor loo
eval_results/phase1_R2_night_loo/*.txt                   r2_night_loo/final_net_epoch_0127.pth           eval_nn_matching.py                                 NN-match 매칭 valid kp            R2 night loo
eval_results/phase1_R3_outside_loo/*.txt                 r3_outside_loo/final_net_epoch_0158.pth         eval_nn_matching.py                                 NN-match 매칭 valid kp            self-train R3 loo
eval_results/filter_pr/summary_ep65.json + per_frame     v9_ablation_A_coord/final_net_epoch_0065.pth    filter_pr_eval.py                                   n_detected>=? + RANSAC consensus  ★v9/v8 트랙. F11=consensus>=6, F17~F20=consensus>=4/5/7/8 sweep. reproj_thresh=5px. 440 frames
eval_results/filter_pr/summary_r1.json (+ t30/t50/t80)   selftrain_r1/final_net_epoch_0070.pth           filter_pr_eval.py                                   RANSAC consensus (F11 base=6)     self-train r1. t30/t50/t80 = conf threshold sweep
eval_results/filter_pr/summary_st8.json (+ t50)          mixed_v8_st_8only/final_net_epoch_0091.pth      filter_pr_eval.py                                   RANSAC consensus (F11 base=6)     ★v8 트랙(mixed_v8)
eval_results/filter_pr/summary_ep68_* / smoke            (tag별; 헤더 weights 확인 필요)                 filter_pr_eval.py                                   RANSAC consensus                  ep68/smoke 변형
eval_results/filter_pr_camfacing/summary_s2.json         dope_cropaug_ft_s2/final_net_epoch_0180.pth     filter_pr_camfacing.py                              n_detected>=6 (valid<6 → inf)     ★과제/누수 모델(정정1). 219 frame, consensus>=6, good=10px. 필터: none/conf/ransac/ransac_loo/cf_strict/fullkp/combo
eval_results/filter_pr_camfacing/summary_heldout_pretrain dope_cropaug_pretrain/final_net_epoch_0060.pth   filter_pr_camfacing.py                              n_detected>=6                     251 frame held-out. consensus>=6, good=10px
eval_results/filter_domain_analysis/summary_paper_base   paper_base/final_net_epoch_0060.pth             filter_domain_analysis.py                           n_detected>=6                     ★논문 base. good=10px. per-domain detectable/good
eval_results/filter_domain_analysis/summary_pretrain     dope_cropaug_pretrain/final_net_epoch_0060.pth  filter_domain_analysis.py                           n_detected>=6                     good=10px
eval_results/filter_domain_analysis/summary_s2           dope_cropaug_ft_s2/net_epoch_0180.pth           filter_domain_analysis.py                           n_detected>=6                     ★과제/누수 모델(정정1). good=10px
eval_results/filter_domain_analysis/_full_*.json         (summary_* 와 동일 tag 모델)                    filter_domain_analysis.py                           n_detected>=6                     per-frame 전체 덤프. ⚠top-level weights 키 없음(summary_* 에만 있음)
eval_results/filter_loo_sweep/sweep_paper_base.json      paper_base (via _full_paper_base.json 재사용)   filter_loo_sweep.py                                 detectable = >=6 corner & 9kp-able RANSAC consensus tau sweep(3~15px)+LOO. ⚠추론 안 함, _full_paper_base 재사용. ransac_c=6
eval_results/filter_loo_sweep/sweep_s2.json              dope_cropaug_ft_s2 (via _full_s2.json)          filter_loo_sweep.py                                 >=6 corner & 9kp-able             ★과제/누수(정정1). _full_s2 재사용
eval_results/filter_flip_consistency/flip_consistency_paper_base  paper_base/final_net_epoch_0060.pth    filter_flip_consistency.py                          detectable (diag 통과 pool)       ★논문 base. flip TTA 일관성. good=10px, gross=20px. taus sweep
eval_results/filter_combo_9kp/combo_9kp_s2.json          dope_cropaug_ft_s2 (via _full_s2.json)          filter_combo_9kp.py                                 detectable(>=6kp); pass: 9kp med  ★과제/누수(정정1). 추론 안 함, _full_s2 재사용. good=10px. min_pass=indoor20/outside20/night8/ALL30
eval_results/filter_combo_perkp/combo_perkp_s2.json      dope_cropaug_ft_s2 (via _full_s2.json)          filter_combo_perkp.py                               detectable(>=6kp)                 ★과제/누수(정정1). 추론 안 함, _full_s2 재사용. good=10px, min_pass=20
eval_results/filter_combo_9kp/{diag,fullkp,paper_base}_pass overlay 대상 모델(s2 위주)                  filter_combo_9kp_overlay.py / *_pass_all_overlay.py  (overlay)                        시각화 산출물(contact sheet)
eval_results/pl_gt_diff/pl_gt_diff_results.json          tag=s2 → dope_cropaug_ft_s2 계열               pl_gt_diff_analysis.py                              fullkp/diag/ratio 통과분 별 비교  ★과제/누수(정정1). gross=20px. PL vs GT 9kp 오차
eval_results/pnp_reproj_compare/comparison_summary.json  v9_ablation_A_coord/final_net_epoch_0065.pth    infer_and_filter_v4.py                              pnp_ok(76/118) + B/C 기하         ★v9/v8 트랙. current vs sigma_weighted vs reproj_guided PnP 비교. 118 img
eval_results/pnp_reproj_compare_loose/*.json             (위와 동일 계열, loose 임계)                    infer_and_filter_v4.py                              pnp_ok + B/C(완화)                B/C 임계 완화 버전
eval_results/v8_audit/audit_summary.json                 해당없음(라벨 audit; 추론 모델 아님)            v8_label_audit.py                                   reproj<=2px / area / y-order inv  ★v8 라벨 무결성 audit(mixed_v8/aug_squash 등). 모델 추론 아닌 GT라벨 검증
eval_results/squash_vs_nosquash/indoor_overlay           (overlay)                                       squash_vs_nosquash.py / squash_indoor_overlay.py    (overlay)                        squash 비교 시각화
eval_results/split_lock/split_assignment.json            해당없음(데이터 split 잠금)                     (metric_split_lock 관련)                            N/A                              평가 split 고정 기록
eval_results/downloads_infer{,_t01}/*.jpg                (추론 시각화 jpg; weights 미기록)               visualize_inference 계열                            (overlay)                        t01 = conf threshold 0.1 변형 추정
eval_results/internet_vis, view_manual, compare_manual   (시각화/수동비교; weights 미기록)               -                                                   (overlay)                        산출물 jpg 모음
eval_results/phase1_pl_overlays, phase2_pl_overlays      (PL overlay)                                    self_train 계열 overlay                             (overlay)                        pseudo-label 시각화
eval_results/phase2_extra_filter/{indoor,outside,night}_R*_cf  R0/R1 cf 모델별                          extra_filter_analysis.py / dump_extra_filter_overlay n_detected>=6 기반 extra filter   phase2 추가 필터 분석
```

---

## ⚠ detectable 정의가 스크립트마다 다른 지점 (분모 불일치 경고)

```
스크립트                       detectable(분모) 정의                                            근거(소스 라인)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
evaluate_on_val.py             corner >= 4 (4개 미만이면 PnP/PCK 불가로 frame skip)              407행 "corner 4개 미만 — PnP/PCK 불가" continue
eval_nn_matching.py            "Frames with predictions" = valid pred kp 1개라도 매칭된 frame    extract_keypoints threshold=0.3; NN 매칭된 valid kp만 거리계산
                               (명시적 >=6 게이트 없음. per-frame denom=GT 전체 129)              denom 표기: "Per-frame ... denom=129" (전체 GT)
filter_domain_analysis.py      n_detected >= 6                                                  179행 det=[r for r if r["n_detected"]>=6]
filter_pr_camfacing.py         valid corner >= 6 (미만이면 reproj=inf 처리)                      154행 if valid.sum()<6: return inf
filter_loo_sweep.py            >= 6 corner AND 9kp-able                                          201/206행 "detectable(>=6 corners, 9kp-able)"
filter_combo_9kp.py            n_detected >= 6 (=mean_match 가능)                                "detectable(>=6kp)"
filter_combo_perkp.py          mean_match_px not None (= >=6 corner)                             158행 det=[r if r["mean_match_px"] is not None]
filter_flip_consistency.py     diag 통과 pool 기준 (>=6 전제 후 diag pass)                       "detectable" = diag-pass pool
filter_pr_eval.py              n_detected (kps not None 카운트) + RANSAC best_consensus 임계      360행 n_detected; consensus 임계 F11=6 / F17~F20=4,5,7,8
```

핵심 불일치 3가지:
- **분모 임계 상이**: evaluate_on_val = **corner>=4** vs 대부분 필터 스크립트 = **n_detected>=6** (서로 다른 분모 → good%/pass% 직접 비교 금지).
- **eval_nn_matching** 은 명시적 detectable 게이트 없이 "예측이 있는 frame" 으로 집계하고 per-frame 분모는 GT 전체(129) 고정. → phase1_R* 의 "<Npx %" 분모(129)와 필터 good%(detectable 분모) 는 서로 다른 모집단.
- **filter_pr_eval** 는 RANSAC consensus 임계 자체를 sweep(F17~F20=4/5/7/8)하므로 같은 파일 안에서도 "통과" 정의가 여러 개. F11(=6)이 base.

추가 메타 주의:
- `filter_loo_sweep` / `filter_combo_9kp` / `filter_combo_perkp` 는 **추론을 다시 안 하고** `filter_domain_analysis/_full_{tag}.json` 을 재사용. 따라서 이 3개의 model 귀속 = 해당 `_full_{tag}` 를 만든 모델(summary_{tag}.json 의 weights 키).
- `_full_*.json` 에는 **top-level `weights` 키가 없음** — 같은 tag 의 `summary_*.json` 을 봐야 모델 확인 가능 (귀속 혼동 위험 지점).
- `eval_summary.json` 에는 **weights/model 필드가 전혀 없음** — 어느 모델 산출인지 파일만으로 복원 불가(마지막 evaluate_on_val 실행 산출).
```
