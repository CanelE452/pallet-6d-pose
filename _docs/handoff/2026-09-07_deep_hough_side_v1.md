# deep_hough_side_v1 실행·결과 인계 — 2026-09-07

최종 상태: 2026-09-07. **6개 본학습·평가·분석·시각화와 QA를 모두 완료했다.** 각 실행 6,000 step, 평가 CSV 131,508행, 시각화 12장·108개 PNG다. `COMPLETION.json`, `VISUAL_QA.json`, 통합 `TASK_COMPLETION.json`에서 PASS를 확인했다. 사전 설계는 [프로토콜](../experiments/deep_hough_side_v1/README.md), 결과는 [REPORT](../../data/pallet/results/deep_hough_side_v1/REPORT.md)와 [갤러리](../../data/pallet/results/deep_hough_side_v1/deep_hough_gallery.html)를 참조한다.

## 확인된 결과

| 지표 | Direct | DHT |
|---|---:|---:|
| synth_test256 위치 중앙값 px | 17.503 | 4.779 |
| cross_v4 128 위치 중앙값 px | 25.571 | 6.536 |
| real DEV52 위치 중앙값 px | 25.654 | 7.569 |
| real DEV52 방향 중앙값 ° | 7.992 | 4.518 |
| real DEV52 위치 p90 px | 87.844 | 67.940 |

표는 seed별 선 오차 통계의 3회 평균이다. pooled 중앙값이 아니다. 역할 평균→고정 seed 평균을 계산한 실사 프레임별 평균 위치 차이(DHT−Direct)는 **-10.480px**, 프레임 대응 bootstrap95% 구간 **[-17.973,-2.918]px**, 52장 중39장에서 DHT가 개선됐다. 이는 기존 프레임/seed에 조건부인 기술적 비교이며 촬영 세션 상관과 새로운 학습 seed의 불확실성을 반영하지 않는다.

개선은 모든 부분집합에 균일하지 않다. outdoor22의 프레임 평균 위치 차이는 **+8.365px**(구간[-1.546,+18.592]), camera-facing25장의 해당 측면 선은 **+1.714px**(구간[-13.155,+18.429])로 관측 평균이 나빠졌다. 두 부분집합의 구간은0을 포함한다. 주 구조적8선 전체 개선과 실제 visible-edge 성능 개선을 동일시하지 않는다. 저장된 주 판정은 `REAL_DEV_DISTANCE_IMPROVEMENT`이며, 위 야외/면 방향 실패와 함께 보고한다.

## 요청·범위·정본 규칙

사용자는 합성 학습→실사 전이 실패 진단 이후 Deep Hough Transform으로 팔레트 옆 edge를 학습하는 실험을 요청했다. 추가 선택은 긴 위·아래 선과 짧은 높이 선을 모두 포함한 **측면 외곽 전체**다. 좌우 측면의 8개 구조적 cuboid 지지 직선을 학습한다. 내부 판재 선이나 실물 visible-edge 수동 정답을 새로 만들지 않았다.

새 head의 순서와 기존 role: `0→1:(1,2)`, `1→3:(3,0)`, `2→5:(5,6)`, `3→7:(7,4)`, `4→8:(0,4)`, `5→9:(1,5)`, `6→10:(2,6)`, `7→11:(3,7)`. convention은 `camera_dynamic_0123_v4`, 앞면 0–3/뒷면 4–7이다.

기존 정본 DEV52만 사용한다: `challenge.data_paths.EVAL_CANONICAL`의 `eval_outside` 22장, `eval_noapril` 12장, `eval_cad` 18장. 52개 JSON의 `objects[0].split == "eval"`을 직접 재확인했다. 정본 전체 140장과 이번 DEV52의 분모를 혼동하지 않는다. 과거 56/161장 및 `data/_eval_sets/*combined`를 사용하지 않는다. 다른 canonical 세션이나 새로운 최종 테스트로 범위를 넓히지 않는다. 이번 실제 이미지는 기존 개발/진단 데이터이므로 독립 final test라고 부를 수 없다. 실사 학습·threshold 튜닝·checkpoint 선택은 없다.

주 평가는 유효 GT 끝점, 원영상 길이≥2px, 원영상과 선분 교차 조건의 구조적 8선 전체다. 실제 DEV에는 414개 선이 지원된다. 기하적으로 카메라를 향한 측면의 경계는 보조 진단 25장·100선이다. 합성 5,374개 면에서 투영 부호 면적과 3D camera-facing 판정이 모두 일치했다. **기하적 camera-facing과 물리적 가시성은 다르다.** 적재물·구멍·가림의 per-edge GT가 없으며, 숨은 cuboid 선도 포함된다.

## 경로·환경·고정 설정

- 저장소: `/home/minjae/Documents/github/pallet-pose`.
- 코드: `scripts/research/deep_hough_side_v1/{runner,network,dht,targets,visualize,test_dht}.py`.
- 결과: `data/pallet/results/deep_hough_side_v1/`; `PURPOSE.md`와 `CONFIG.json` 존재.
- 원본 캐시: `data/pallet/results/hough_attention_transfer_v1/`의 `manifest.json`, `features.npy`, `targets.npz`, `CONFIG.json`, `CACHE.json`.
- Python: `/home/minjae/anaconda3/envs/pallet-pose/bin/python`; PyTorch `2.1.1+cu118`, NVIDIA GeForce RTX 3080. 셸 호출은 `login:false`.
- 본 실행: `runner.py --phase all`; root exec session **67714**에서 학습·평가를 진행했고 완료됐다. 기록 로그 `data/pallet/results/deep_hough_side_v1/run.log`. 후속 분석·렌더링은 root가 별도로 실행했다. 이후 생성된 `experiment.py` 통합 driver로 이번 결과를 실행했다고 주장하지 않는다.

최초 실행의 cwd는 저장소 루트였으며 source/run-dir는 절대 경로였다. 결과 폴더에 PURPOSE는 이미 있었다. 재개 명령은 프로토콜의 재현 절처럼 결과 디렉터리에서 실행한다.

두 arm은 `Direct`와 `DHT`. 동일한 합성 사전학습 고정 VGG `128×50×50` 캐시를 사용한다. 원영상은 반사 pad100→400×400, float16 캐시를 학습 시 float32로 읽는다. 백본은 갱신하지 않는다. DHT는 학습된 16채널 공간 특징의 선형 보간 sparse 투표를 가중치 합으로 정규화하고 seam-aware Hough convolution을 적용한다. 기존 Direct는 8개 역할 attention descriptor를 사용한다. 이는 구조 비교이며 단일 요인 ablation이나 원 논문 전체 ResNet/FPN 재현이 아니다.

학습 모집단: synth_train 2,048 / synth_val 256 / synth_test 256 / cross_v4 128 / real_dev 52. seed 1·2·3, 각 arm 6,000 step, batch12, AdamW lr0.001/weight_decay0.0001, seed별 동일한 배치 순서. 최종 고정 step checkpoint를 사용한다. sanity는 합성 학습 앞32장, DHT seed101, 1,500 step의 별도 실행이며 본학습 초기값으로 쓰지 않는다.

본학습 중 기록하는 곡선은 train loss다. synth_val은 최종 checkpoint에서만 평가하며 중간 검증·선택에 사용하지 않는다.

격자: 법선 theta `0..179°` 1° 간격, rho `-35..35` 0.5 특징셀 간격, 180×141 후보, 중심 `(24.5,24.5)`. 투표 없는 후보는 loss/decode에서 제외. Gaussian CE target sigma(theta)=1°, sigma(rho)=0.5셀. theta seam에서 rho 부호가 바뀌는 무방향 직선 동치 관계를 유지한다. 최고 후보를 원영상의 무한직선으로 해석하며 선분 끝점 예측이라고 주장하지 않는다.

## 본학습 전 좌표 수정과 검증

원영상→특징 픽셀 좌표는 `(original+pad+0.5)*50/padded_size-0.5`, 역변환은 `(feature+0.5)*padded_size/50-0.5-pad`다. 실제 VGG 세 maxpool의 stride8/첫 receptive-field 중심3.5와 OpenCV resize half-pixel 규칙을 확인했다. 이전 캐시의 연속 grid를 `targets.make_targets(...,pad=100)` 안에서 보정한다. 640×480+pad100 기준 보정은 x=-0.470238095, y=-0.463235294셀이다. 입력 캐시 grid를 호출부에서 다시 보정하면 안 된다.

수정 전 예비 sanity/check/config는 `preliminary_continuous_grid_sanity/`로 옮겼다. 최종 결과와 섞지 않는다. 수정된 `target_audit/TARGET_AUDIT.json`: 2,740장 왕복 최대 0.000035643px/0.000009469°, 특징 중심→원영상 중심 오차0px. `CHECKS.json` PASS이며 target/lattice/gradient 검증을 통과했다.

수정 후 `SANITY.json` PASS: 32장·256선, 방향 중앙값 **0.35043°**/p90 **0.84489°**, 위치 중앙값 **2.53994px**/p90 **5.74178px**, joint≤5°·8px **98.4375%**, loss **10.03313→3.11363**. 이것은 학습32장 fitting 결과이며 전이 성능이 아니다.

## SHA-256와 원본 재검증

| 대상 | SHA-256 |
|---|---|
| 현재 `CONFIG.json` | `8ee34f3ce54c99f225244cd01d1ae79cc2bd5735ff1c810e6a8eecf1747c25f9` |
| 원본 `manifest.json` | `5cbe10940c6b6c0cd22ebd407e262284bd311ec1fe3db1e1dbe32c61c916d31a` |
| 원본 캐시 `CONFIG.json` | `72e90401bc49cfded77f0689b1a6ca890bded9b2b9498fefbb612a86a02edb16` |
| 고정 백본 `weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth` | `0de80490cb3b4f9b11565db7a4aea6338f64edb8f9614910bfb52bf03ce0dc3f` |
| `runner.py` | `0610a88850975eed1ae64dded0c6c57184e97730d20feed541f14697c84d4c4f` |
| `targets.py` | `fcef6d0036dae8e8f7c90044ee65fef864c5583c71d8f68460bf2deb6d6b0761` |
| `network.py` | `d0d713e698f70996e7b82da5bc0382408ba2953ccaa5a08a12a82955b8551d81` |
| `dht.py` | `8ef414664bfc48ed6e6b6e4e8c68ec298d44bc667055bfa3c3dfbed4c84f580a` |
| `SANITY.json` | `b3d3fcb250defc65a683f945a356c1ce68e0c83d8ff9d7a61f7a94e09a3ead4a` |
| `SOURCE_REVALIDATION.json` | `563cef1811a79e7eeb5f6554254789f8c33bcc8b47963800bde366be4b5264e6` |
| 최종 `COMPLETION.json` | `5431acc1594ae7fcf84cab11f0ec727f7a0b02a61a94761b6b887781700841fa` |
| 최종 `ANALYSIS.json` | `8784746bb0b328f6d9060218036811c7208d4651b11bd47bd67f6cd060a75d91` |
| 최종 `RESULTS.json` | `486b52e4430fb201ea907be2c841568910084f64384d37a9640177a8511e1b65` |

`SOURCE_REVALIDATION.json`은 본학습 중 CPU 읽기 전용 감사이며 PASS다. 2,740개 이미지와 2,740개 annotation, 총5,480개 파일 SHA가 원본 manifest와 전부 일치했다. 모집단 10쌍의 이미지 SHA 교집합은 모두0. train 인덱스0–2047과 real2688–2739는 겹치지 않고, real source_kind가 train에 없다. runner가 synth_train만 샘플링함을 확인했다. 현재 원본 검증은 삭제된 백본 사전학습 manifest의 중복까지 보장하지 않는다. 평가 합성과 백본 사전학습의 전체 중복은 미검증이다.

## 완료 checkpoint와 전달 사항

최종6개 checkpoint는 결과 폴더 `checkpoints/`에 있다. `COMPLETION.json`이 각 checkpoint의 finite weights, 설정 SHA, frozen core source SHA, 6,000 step 메타데이터와 학습 history 최종 step, CSV/RESULTS 수치 일치를 검증했다. 같은 파일의 `expected_runs=6`, `expected_steps=6000`, `evaluation_csv_rows=131508`, `PASS=true`를 확인했다.

| checkpoint | SHA-256 |
|---|---|
| `DHT_sanity32_seed101_final.pth` | `50eefb52d26c97b9e90a57ff4814537bde2f4e3e57a849bd139567fd54d11041` |
| `Direct_seed1_final.pth` | `29f3f7180a326533834f8ff6c1433710a73df4d1915dfdbdb912f515a0ff7a5d` |
| `Direct_seed2_final.pth` | `4af9af0c303e363fef8127e6ade06d662ee6626ae7cf43f646160d9c6ba230f1` |
| `Direct_seed3_final.pth` | `eccac8746881ccbf4fb1dd82d8bdf51fd0f8189c28cb75f1ad0abfa6d3bc6608` |
| `DHT_seed1_final.pth` | `10cb85860e5e10929145ab1babeda5ca4f253a8af9e0c4d806393b61c95eb5c3` |
| `DHT_seed2_final.pth` | `a3e9c1c898c7f00c4a91eae03af2688bb874b3bf15b96da42d1c0d7278f45151` |
| `DHT_seed3_final.pth` | `917a69494dbece71d7afb89473f5483520d0f6500bad4e453b703eb131b439b1` |

1. 브라우저·전체 이미지 QA를 완료했다. 모든 12개 overview를 개별 확인했고 실제 높이·깊이 역할 패널을 점검했다. 오프라인 Chrome의 기본 실사 선택, 이미지/역할 전환, 이미지 로딩이 정상이며 JS 오류는 0건이다. `VISUAL_QA.json`에 최종 visualizer SHA와 수치 일관성 검사를 저장했다.
2. 사용자에게 결과 REPORT와 gallery를 전달한다. 전체 위치 개선뿐 아니라 outdoor/camera-facing 집합의 관측 평균 악화를 함께 설명한다. 50×50 해상도, amodal GT, DEV 사용, 사전학습 중복 미검증이라는 해석 범위를 유지한다.
3. DHT spatial heatmap은 최고 예측 bin의 pre-softmax logit에 대한 `abs(gradient×input feature)` 채널합이다. attention이나 인과적 픽셀 중요도로 설명하지 않는다. Hough 확률·예측선·GT를 함께 보고 반사 패딩 경계를 표시한다. gallery selection은 고정 SHA 기준·seed1이며 예측을 본 뒤 좋은 예만 선택하지 않았다.

기존 history·이전 실험 원본은 수정하지 않았다. 수정 전/후 sanity가 서로 다른 폴더에 있으므로 재개 시 혼합하지 않는다.
