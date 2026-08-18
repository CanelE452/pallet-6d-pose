# yolo_pose_one_model — 과제용 YOLO26n-pose 단일 모델

리프터가 정지한 뒤 RealSense RGB 한 장에서 팔레트 bbox + 9 keypoint 를 예측하는
모델을 처음부터 준비한다. 예측은 기존 배포 후처리
(`challenge/25y_automatic_lifter-master/.../depth_cam/calib/pose6d_adapter.py`)의
PnP + depth 로 yaw / lateral offset / forward distance 가 된다.

## 이번 라운드 범위 — 합성만

사용자 지시(2026-08-14): **G + T 합성 데이터만 학습**한다. real 은 이후 별도
finetune 단계로 미룬다. 따라서 이 폴더에는 Stage B, real split, real 평가가 없다.

★ real validation 이 없으므로 checkpoint 선택 근거는 **합성 val 뿐**이다.
   real task metric(detection recall, PnP 성공률, yaw 오차)은 측정하지 않았고
   주장하지도 않는다.

## 데이터

```
G  generic_synth   40,000   data/pallet/training_data/paper_release/v2_prod40k_clean_merged
                            여러 팔레트 자산, 해상도 4종, 프레임마다 독립 렌더
T  target_synth    19,968   challenge/data/02_synthetic/training/{v1,v2}  (palletobj)
                            과제 팔레트, 640x480, frame_meta.scenario 로 장면 그룹
R  real              (미사용)  01_real/manual_gt + eval_canonical — 다음 단계
```

## 실행 순서

```bash
conda activate pallet-yolo26          # ★ pallet-pose 는 ultralytics 8.0.120 이라 YOLO26 불가

R=challenge/yolo_pose_one_model
python $R/scripts/discover_and_audit.py          # registry + reports/01
python $R/scripts/verify_kp_contract.py --n 400  # 계약 실증 (reports/02 §1)
python $R/scripts/verify_v4_conversion.py --n 600
python $R/scripts/test_padding_contract.py       # 실패하면 여기서 중단
python $R/scripts/build_splits.py                # manifests + reports/03
bash   $R/scripts/prepare_stage_a.sh 10          # 60k 프레임 padding 변환 (~48 GB)
python $R/scripts/build_stage_dataset.py --stage stage_a --target-repeat 2
python $R/scripts/audit_yolo_pose_labels.py --dataset datasets/stage_a --report reports/04_label_audit.md
python $R/scripts/render_overlays.py --dataset datasets/stage_a --split train --prefix T__ --n 100
bash   $R/scripts/run_smoke.sh                   # 2 epoch, batch 32
bash   $R/scripts/train_stage_a.sh               # 60 epoch
python $R/scripts/select_final_checkpoint.py --run runs/stage_a_synth_640_b32_seed42
# Jetson 에서:
bash   $R/scripts/export_jetson.sh
```

## 고정 계약

`contract/pallet_pose_contract.yaml` 이 단일 출처다. 요점:

```
keypoint   0-3 near face (0 nTL, 1 nTR, 2 nBR, 3 nBL) / 4-7 far / 8 centroid
           {0,1,4,5} top, {2,3,6,7} bottom.  left/right 는 이미지 기준
padding    100 px BORDER_REFLECT_101, 전 도메인. 640x480 -> 840x680
           (G 는 해상도가 4종이라 padded 크기가 프레임마다 다르다)
batch      32 고정, nbs 64. OOM 이면 GPU 점유를 먼저 보고 batch 를 낮추지 않는다
fliplr     0.0 (flip_idx 는 계약 기록용으로만 보존)
```

## 미해결 (real finetune 전에 결정 필요)

```
팔레트 height     real GT 0.11 vs 배포/합성 0.12   → 실측 확인
width/depth       뷰에 따라 1.10 / 1.30 이 바뀜. 배포는 1.10 고정이라
                  긴 변 정면 프레임(real 52%)에서 reproj 7.6 px (자기 dims 1.5 px)
PnP 방법          프롬프트 SQPnP vs 배포 EPnP+LM(>=6점)
yaw convention    배포 psi 정의를 평가와 일치시켜야 함
real 학습 가용    133장(13세션)뿐 — 161장은 봉인된 정본 평가셋
```

## 주의

- `convert_to_camera_facing_v4.py` 를 G/T 에 **다시 돌리지 않는다**. 2D area 휴리스틱이
  저앙각에서 오판한다 (reports/02 §7).
- `project=` 는 절대경로로 준다. 상대면 Ultralytics 가 `runs/pose/` 아래로 묻는다.
- G 에는 라벨되지 않은 팔레트가 함께 찍힌 프레임이 있다 (reports/05). real recall 이
  낮게 나오면 가장 먼저 의심할 것.
