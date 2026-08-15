# 우분투 이전 가이드 (2026-06-01 작성)

윈도우 → 우분투 작업 환경 이전을 위한 manifest. 현재 활성 라인은 **challenge 계열 하나**이며,
구 mixed/pallet 라인과 실패작/실험 중복은 미이전한다. KEEP만 옮기면 약 12~13G.

## 이전 방법 개요

1. **코드**: 우분투에서 `git clone https://github.com/CanelE452/pallet-6d-pose.git`
2. **git에 커밋 안 된 코드(untracked)**: clone에 안 따라옴 → 아래 §3 목록을 직접 복사
3. **큰 파일(data/weights)**: gitignore 대상이라 clone에 없음 → 아래 §4 목록을 rsync/외장으로 복사

---

## 1. KEEP — 가져갈 weights (challenge 라인만)

```
weights/challenge_track/challengenight/                       최종 모델 (=doc challenge0123_ft_v2)
weights/challenge_track/challenge0123/                        v4 convention 베이스
weights/challenge_track/challenge0123_ft_manual/              정식 ft (6 day GT)
weights/selftrain/r1_outside_loo/                       eval/figure 스크립트 직접 참조
weights/misc/f5_noapril_ransac_loo_realonly/       eval/figure 스크립트 직접 참조
challenge/weights/baseline_v8_A.pth           ft init (별도 위치 — 꼭 포함)
```

## 2. KEEP — 가져갈 data

```
data/night/*                                  raw 캡처 (재촬영 불가, 활성 ft GT 소스)
data/outside/*                                raw 캡처 (재촬영 불가)
data/pallet/raw_data/capture0403middle        indoor 평가셋 + AprilTag GT
data/pallet/raw_data/capture0403noapril       필터 ablation 평가셋
data/pallet/raw_data/real_data                real 평가/pretrain
data/pallet/raw_data/models_usd               Isaac 입력 USD 모델
data/pallet/scan_cleanup/                      실제 팔레트 스캔 (pallet_full.obj/.blend)
data/_eval_sets/night_combined                활성 3-도메인 평가셋
data/_eval_sets/outside_combined              활성 3-도메인 평가셋
data/pallet/training_data/mixed_v8_train      현재 학습 base (57M, Isaac 재생성 수일)
```

## 3. KEEP — git 미커밋(untracked) 코드 — 직접 복사 필요

> 커밋하지 않으면 clone에 안 따라온다. rsync 대상에 반드시 포함할 것.

```
challenge/yolo_pose/                          yolo 학습 코드
challenge/_docs/models/                       challenge 모델 docs (분류 근거 문서)
scripts/data_prep/eval/plot_*.py              플롯 스크립트 다수
scripts/data_prep/eval/dump_*.py
scripts/data_prep/eval/eval_3_domains.sh
scripts/data_prep/eval/eval_6d_3_domains.sh
scripts/data_prep/eval/extra_filter_analysis.py
scripts/data_prep/eval/qualitative_panel.py
scripts/data_prep/eval/prototype_demo.py
scripts/data_prep/inference/extract_pl_v5.py
_docs/experiments/self_training/*.md
```

## 4. 미이전 (윈도우에 그대로 둠 — 필요시 나중에 개별 복사)

```
[weights — 구 라인/실패작/실험중복]
  pallet_category, pallet_v11, pallet_v11_far       구 mixed 라인
  challenge                                          구 v1/v2 convention
  challenge_ft_pallet07                              단일 도메인적응 실험
  r2/r3_*_loo, r2_outside_cf_strict, r1_*_cf_*       self-train 중간 라운드
  challenge_camfacing_ft/scratch, challenge_ft_mp40, ckpt 0개 실패작
  r1_outside_cf_loo, r1_outside_cf_loo_fast
  f5_ep100/ep125/reproduce, f4_*, f3_*, r1_outside_ransac, ...  f-series 실험 중복
  ep65_pl_realonly, selftrain_r1, combined_v1, legacy_filter_100, pallet_category_test

[data — 재생성 가능 / 미참조]
  data/isaac/isaac_assets                           Isaac 합성 안 함 → 불필요
  data/pallet/eval_results, results                 코드 재실행으로 복원
  data/pallet/training_data/{train,val,mixed_v9_train,mixed_v10_train,
    blender_*,test_blender_*,mixed_v8_st_noapril,*_batch,*_backup,pl_*,pseudo_*}
  data/pallet/test_data_results                     입력 소스 없는 고아 산출물
  data/pallet/raw_data/{vdoframes,capture02,capture03,real_pool_all,internet_pallet_data}
  *.log
```

## 5. rsync 복사 리스트 (예시)

전송 방식 확정 후 사용. 같은 PC 듀얼부트면 마운트 경로로, 다른 머신이면 `rsync -avP -e ssh`.

```bash
# 예: 윈도우 파티션이 /mnt/win 에 마운트된 경우
SRC=/mnt/win/Users/minjae/Documents/github/FoundationPose
DST=~/FoundationPose

# weights (challenge 라인)
rsync -avP \
  "$SRC/weights/challenge_track/challengenight" \
  "$SRC/weights/challenge_track/challenge0123" \
  "$SRC/weights/challenge_track/challenge0123_ft_manual" \
  "$SRC/weights/selftrain/r1_outside_loo" \
  "$SRC/weights/misc/f5_noapril_ransac_loo_realonly" \
  "$DST/weights/"
rsync -avP "$SRC/challenge/weights/baseline_v8_A.pth" "$DST/challenge/weights/"

# data (KEEP)
rsync -avP "$SRC/data/night" "$SRC/data/outside" "$DST/data/"
rsync -avP \
  "$SRC/data/pallet/raw_data/capture0403middle" \
  "$SRC/data/pallet/raw_data/capture0403noapril" \
  "$SRC/data/pallet/raw_data/real_data" \
  "$SRC/data/pallet/raw_data/models_usd" \
  "$DST/data/pallet/raw_data/"
rsync -avP "$SRC/data/pallet/scan_cleanup" "$DST/data/pallet/"
rsync -avP "$SRC/data/_eval_sets" "$DST/data/"
rsync -avP "$SRC/data/pallet/training_data/mixed_v8_train" "$DST/data/pallet/training_data/"

# untracked 코드 (§3)
rsync -avP "$SRC/challenge/yolo_pose" "$SRC/challenge/_docs" "$DST/challenge/"
# scripts 는 clone된 위에 덮어쓰기 (untracked 파일만 추가됨)
rsync -avP "$SRC/scripts/data_prep/eval/" "$DST/scripts/data_prep/eval/"
rsync -avP "$SRC/scripts/data_prep/inference/extract_pl_v5.py" "$DST/scripts/data_prep/inference/"
```

## 6. 우분투 환경 재구성 체크리스트

```
[ ] conda env 재생성: conda create -n pallet-pose python=... ; pip install -r requirements.txt
    - pyrealsense2 는 depth_cam 제거로 불필요 (RealSense 미사용)
[ ] CUDA/PyTorch 리눅스용 재설치 (cu126 호환 확인)
[ ] Deep_Object_Pose/ 가중치(VGG-19 등) 필요시 재다운로드
[ ] *.bat → *.sh 변환 (challenge/yolo_pose/scripts/*.bat 4개)
[ ] clone 후 depth_cam/ 폴더 없음 확인 (이미 커밋으로 제거됨)
[ ] config/*.yaml 의 weight 경로가 가져온 challenge 라인을 가리키는지 점검
```

## 참고 — 분류 근거
- weights/data 전수 dependency chain 추적 결과 (2026-06-01 분석)
- 활성 라인 = challenge 계열, 학습 base = mixed_v8_train, 평가 = evaluate_real.py / eval_3_domains.sh
