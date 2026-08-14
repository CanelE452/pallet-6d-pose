# Pallet 6D Pose — Geometry-aware Self-Training

팔레트 6D 포즈 추정을 위한 기하학적 제약 기반 준지도 DA 프레임워크.
Python + PyTorch + Isaac Sim + DOPE.

## 🛑 평가셋 (새 세션 필독 — 다른 셋 쓰지 말 것)

> **평가는 `objects[0].split == "eval"` 인 manual GT 161장만 사용한다.**
> 경로는 `challenge/data_paths.py` 의 `EVAL_CANONICAL` 이 유일한 출처다 —
> 문자열로 다시 쓰지 말고 import 한다.
> 상세·이력·금지사항: `_docs/EVAL_SET_CANONICAL.md` (테스트로 강제:
> `challenge/tests/test_eval_set_canonical.py`)

```
01_real/eval_canonical/_outside_eval_manual_gt         22   filter-val 세션
01_real/eval_canonical/capture0403noapril_manual_gt    12
01_real/eval_canonical/capturepalletcad_manual_gt      22
01_real/manual_gt/capturepallet07_manual_gt            27   ★final-test
01_real/manual_gt/capturepallet09_manual_gt            36   ★final-test
01_real/manual_gt/capturenight08_manual_gt             17   ★final-test
01_real/manual_gt/capturenight09_manual_gt             25   ★final-test
                                        (challenge/data/ 아래)  합 161
```

- ⚠️ **"56장" 은 폐기된 수치다.** 2026-08-07/08-08 에 `metric_split_lock.md` §1.6 의
  final-test 세션 4개를 봉인 해제해 정본에 편입했다(63+42=105장 추가). 옛 문서·메모리에
  56장이 보이면 그건 07-3x 기준이다. 실측·테스트 선언값 모두 161.
- ★final-test 4개는 정본이지만 **threshold 튜닝·모델 선택 금지**(봉인 소진, 재봉인 불가).
  보고 시 outside/night 를 한 덩어리로 합치지 말 것 — `data_paths.FINAL_TEST` 참조.
- `split` 은 **최상위가 아니라 `objects[0].split`**. 최상위만 읽으면 전부 "없음" 으로 보인다.
- `data/_eval_sets/outside_combined` / `night_combined` 는 **구본**(05-27, split 없음) — 평가 금지.
- split 없는 JSON 을 eval 로 간주하지 않는다(구 규칙 폐기). `train` 표시 프레임 포함 금지.
- ⚠️ PAPER_S2 의 N87(87장)은 이 규칙 이전에 만들어져 **eval 10장 누락 + train 13장 오염**.
  07-29~08-04 판정의 적용범위는 그 셋 기준이다.

## ⚠️ 핵심 방향 (새 세션 필독 — v8 회귀 금지)

> 새 세션에서 옛 문서(`_docs/filter/2026-04-11_selection.md`, "RANSAC c≥6", "Y=UP object-frame")를 보고 **v8 로 돌아가지 말 것.** 자세한 내용은 memory 의 [v8 폐기], [camera-facing convention], [두 트랙] 3개 메모리.

1. **v8 = 완전 폐기 실패작.** `v8_ablation_A_coord`, object-frame `mixed_v8`, `pl_*_r0_*`(소문자) 절대 사용 금지. object-frame 점을 0123 으로 잘못 구성한 데이터로 학습한 모델.
2. **convention = camera-facing 0123** (v4, 2026-05-22 결정). 0~3=앞면, **{0,1,4,5}=위 / {2,3,6,7}=아래**, 8=centroid. (아래 Architecture 의 "Y=UP" 표기는 폐기됨.)
3. **두 트랙 병행**:
   - **논문용**: v1/v2(내 파렛트) 제외, 인터넷 무료 모델로 학습 → 처음 본 파렛트 일반화. 비율 강건성(squash + JSON 꼭짓점 동기화) + truncation padding + 기하 필터 self-training.
   - **과제용 (challenge)**: v1/v2 과적합, forklift 실배포.
4. **PnP 용도 분리**: self-training PL 필터 = 2D 기하로 PnP 불필요(처음 본 파렛트 가능) / 평가·거리추정 = 치수 known 데이터에서만.
5. **필터 동기**: PL 수보다 **품질(신뢰도)** 이 핵심 (발표 교훈 — indoor 소량 PL 로 R1 ↑, outdoor/night 다량 PL 인데 R2 ↓).

## 연구 가이드

최신 연구 설계: `_docs/` (README.md에서 목차 확인)
도메인 서베이: `_docs/survey/survey-6d-pose-estimation.md` (방법론/학습전략/메트릭 비교)

## Commands

### Step 1: 합성 데이터 생성 + DOPE Pretrain
- 합성 데이터 생성 (단일): Isaac Sim에서 `scripts/data_prep/isaac_sim/gen_replicator_data.py` 실행
- 합성 데이터 생성 (배치): `bash scripts/data_prep/isaac_sim/generate_all.sh` (64프레임/배치, 자동 재시작)
- DOPE pretrain: `bash scripts/train_dope.sh` (config/default.yaml 기반)
- DOPE fine-tune: `bash scripts/train_dope.sh --finetune`

### Step 2-3: Self-Training + 평가
- Self-training: `python scripts/self_training/self_train.py` (설정: `config/stage3_selftrain.yaml`)
- Synthetic val 평가: `python scripts/data_prep/eval/evaluate_on_val.py --weights <path> --val_dir <path>` (PCK + PnP Reproj + Volume Ratio)
- Real test 평가: `python scripts/data_prep/eval/evaluate_real.py --weights <path> --test_dir <path>` (ADD + 5cm5° + Reproj)
- 실험 비교: `python scripts/compare_experiments.py`

### Real Data + AprilTag GT
- AprilTag GT 생성: `python scripts/data_prep/apriltag/apriltag_gt.py --image <tag_image> --visualize`
- Real data split: `data/pallet/real_data/{real_test_seen,real_test_unseen,real_unlabeled,real_dev}/`
- 촬영 프로토콜: `data/pallet/real_data/README.md`

### 유틸리티
- Annotation 시각화: `python scripts/data_prep/visualize/visualize_annotations.py`
- 추론 시각화: `python scripts/data_prep/visualize_inference.py --weights <path> --num_syn 10 --num_real 10`
- 데이터 검증: `python scripts/data_prep/validate/merge_and_validate.py`, `python scripts/data_prep/validate/verify_keypoints.py`
- 실시간 추론 (native): `python scripts/dope/run_dope_live.py --realsense --weights <path>` (RealSense SDK + `pip install pyrealsense2` 필요)

## Architecture

- **Pose 표현**: Keypoint-based (DOPE) — 팔레트는 비대칭 직육면체로 keypoint 방식에 적합
- 3단계 파이프라인: Step 1 (합성데이터+DOPE학습) → Step 2 (Geo Filter+Pseudo-label) → Step 3 (Finetuning) → 반복
- DOPE 모델: `Deep_Object_Pose/` (VGG-19 backbone, 9 belief maps + 16 affinity fields)
- **PnP**: EPnP + RANSAC (`scripts/self_training/pnp_solver.py`) — keypoint → 6D 포즈 복원
- 합성 데이터: Isaac Sim 4.5.0 + Omniverse Replicator, NDDS 포맷 JSON annotation
- USD 모델: `data/pallet/models_usd/scene*.usd` (4종 팔레트)
- Geometric Filter: **재설계 중 (camera-facing 0123 기반 2D 기하 필터).** 옛 "RANSAC c≥6" 선정(`_docs/filter/2026-04-11_selection.md`)은 v8/object-frame 기준이라 폐기. 새 방향: 공간 대각선 교점≈centroid, {0,1,4,5}위/{2,3,6,7}아래, 변 비율 등 2D projective 기하 (PnP 불필요). 정확한 설계는 `3d-expert` 위임. 필터 전용 문서: `_docs/filter/`.
- Keypoint convention: **camera-facing 0123** (0~3 앞면, {0,1,4,5}=위/{2,3,6,7}=아래, 8=centroid). memory `camera-facing-0123-convention` 참조. (위 "핵심 방향" 참고 — 옛 Y=UP object-frame 은 폐기)
- **평가 (Synthetic val)**: PCK@3/5/10px + PnP Reproj + Volume Ratio — `scripts/data_prep/eval/evaluate_on_val.py`
- **평가 (Real test)**: ADD + 5cm5° + Reproj — `scripts/data_prep/eval/evaluate_real.py` (AprilTag GT 기반)
- **Real data**: seen/unseen/unlabeled/dev split — `data/pallet/real_data/`
- 실시간 추론: `scripts/dope/run_dope_live.py` + RealSense D435i (native, pyrealsense2)
- 팔레트 규격: **논문용은 고정 비율 없음** (처음 본 파렛트 일반화 목표 → squash 로 여러 비율 학습). PnP 가 필요한 평가/거리추정은 치수 known 데이터(내 파렛트 GT, 과제용)에서만. config 의 KS T-11형 1100×1100×150mm 은 v8 잔재 — 실측 치수 별도 확인 필요.

## Code Style

- conda env: `pallet-pose`
- 설정은 `config/default.yaml`에서 중앙 관리 (`train_dope.sh`가 yaml 읽음)
- Isaac Sim 스크립트는 standalone 실행 (Isaac Sim 내장 Python), 모듈화: `scripts/data_prep/isaac_sim/sdg_*.py`
- DOPE 데이터 로더: CleanVisiiDopeLoader (`{i:06d}.png` + `{i:06d}.json` 쌍)

## Compaction Rules

- **compact 시 원문 디테일 보존** (요약 금지):
  - 정본 평가셋 규칙 — `objects[0].split=="eval"` 161장(7폴더) + wood45.
    평가뿐 아니라 **development·진단·probe subset 도 정본에서** 뽑는다
    (`data/_eval_sets/*combined` 사용 금지 — 여기서 뽑아 판정 4건이 뒤집힌 이력)
  - 현재 architecture 판정과 그 근거 한 줄
  - 진행 중 실험의 checkpoint 경로·SHA·고정 상수(sigma/tau/threshold)
  - 미해결 에러와 시도한 해결책
  - 사용자가 명시한 제약(SEALED holdout, final-test 미접근 등)
- **요약 허용**: 파일 탐색·Glob/Grep 과정, 이미 종결된 실험의 개별 수치
  (`_docs/audits/`·`_docs/history/`에 있음 — 복붙 말고 가리킬 것), 중간 시행착오 턴
- 탐색·조사는 subagent(Explore)에 위임해 main context 오염 방지
- `/context` 60% 도달 시 작업 단위 마무리 후 수동 compact (auto trigger 회피)
- 긴 실험 착수 전 `_docs/handoff/{날짜}_{실험명}.md`에 확정 사실을 남겨
  compact 후에도 미지수 0에서 재개할 수 있게 한다

## Gotchas

- Isaac Sim DLL 충돌: `CUDA_MODULE_LOADING=LAZY` + `PYTHONUNBUFFERED=1` 설정 필요
- Isaac Sim ~2분/프레임, 64프레임마다 재시작 (메모리 누수)
- Replicator `rep.distribution.choice()`는 머티리얼 생성 시 1회만 평가됨 → USD API로 직접 변경
- 팔레트 기울기 없이 바닥 수평 고정 (tilt=0)
- 어려운 케이스(낮은 대비, 유사 색상)는 유지 — 모델 로버스트니스에 필수
- Belief map sigma=4.0 유지 (sigma<1은 gradient vanishing 발생, `_docs/method/step1_synthetic_data.md` 3.6절 참조)
- **3D/2D 작업은 항상 `3d-expert` agent 에 먼저 위임** (2026-05-22 사용자 규칙). 좌표계 변환 / cuboid 라벨링 / camera convention (OpenCV vs USD vs ROS) / projection / rendering / annotation 시각화. 직접 trial-and-error 금지. v1/v2/v3 keypoint 변환 4 회 시행착오 후 3d-expert 가 한 번에 v4 해결한 경험. 자세한 근거: `feedback_3d_expert_first.md` 메모리.
- 장시간 작업(데이터 생성, 학습 등) 실행 중에는 주기적으로 로그/프로세스를 확인하여 정상 진행 여부를 모니터링한다

## Self-Verification

- [ ] 연구 설계 변경 시 `_docs/`의 해당 문서도 함께 업데이트했는가?
- [ ] Isaac Sim 스크립트 수정 시 ORIENTATION_OVERRIDES 건드리지 않았는가?
- [ ] 새 스크립트 추가 시 재현성을 위한 config/argparse 지원이 있는가?
- [ ] Geometric Filter 임계값 변경 시 `config/stage3_selftrain.yaml`에 반영했는가?
- [ ] 학습 설정(sigma, batch, lr) 변경 시 `scripts/train_dope.sh`와 `_docs/method/step1_synthetic_data.md` 3.6절 동기화했는가?
- [ ] 생성된 이미지/overlay 확인 시 각 프레임을 개별 로드하여 변경 사항(적재물, 카메라 높이, 배경 등)이 실제 반영되었는지 눈으로 검증했는가? 로그만 보고 판단하지 않는다.
