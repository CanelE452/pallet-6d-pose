# Clean → 자연 가림 자세 전이 재현

기존 결과를 덮어쓰지 않는 후속 실험이다. `PRIMARY_PROTOCOL.json`을 먼저 읽는다. 원 RGB, 직접 클릭 좌표, 전체 예측 좌표 및 checkpoint는 공개 Git에 넣지 않으므로 로컬 원자료와 잠금 SHA가 필요하다. Git만 내려받아 원자료까지 포함됐다고 가정하지 않는다.

## 환경과 입력

Ubuntu / Python 3.10 / PyTorch 2.1.1+cu118 / Ultralytics 8.4.60 / RTX3080. 이 실행에서 사용한 Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`이다. 격리 환경의 CUDA 접근 실패와 호스트 드라이버 장애를 구분한다. 다른 프로세스 종료·환경 재설치·재부팅은 하지 않는다.

실행 전 `MPLCONFIGDIR=/tmp/pallet_clean_pose_mpl`, `OMP_NUM_THREADS=4`를 설정한다. 모든 명령은 저장소 루트에서 실행한다. 데이터/코드 SHA가 달라지면 기존 잠금을 덮어쓰지 말고 차이부터 확인한다.

## 감사와 학습 전 검사

```bash
python -m scripts.research.pallet_clean_to_pose_transfer_v1.prepare
python -m scripts.research.pallet_clean_to_pose_transfer_v1.preflight
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train lock
python -m pytest -q scripts/research/pallet_clean_to_pose_transfer_v1
```

RGB-only review로 고정한78장/4recording을 재사용한다. 마커판 영상은 사용자의 결정에 따라 제외했다. 같은 이미지의 RAW/REF 좌표는 기존 저장 label을 그대로 사용한다. 새 affine 뒤에는 공통 v2 교집합만 감독하고 나머지는 v1 true-ignore로 처리한다. 이는 과거 실험과의 명시적인 계약 차이다. 새 가림은 입력 RGB에만 적용한다.

## 고정 2×2

```bash
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train train --arm CLEAN_RAW_CLEAR --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train train --arm CLEAN_REF_CLEAR --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train train --arm CLEAN_RAW_OCC --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train train --arm CLEAN_REF_OCC --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.parity --seed 42
```

각각 같은 R0에서320update. `last.pt`만 선택하며 validation best를 선택하지 않는다. 프레임워크의 종료 시 합성32장 검증은 checkpoint 선택 근거가 아니다. 완료된 fit은 해시 검증 후 재사용하고, 미완료 실행 디렉터리를 조용히 덮어쓰지 않는다. 각 epoch 상태·학습 로그·전체320batch trace는 private 결과 폴더에 보존한다.

## 예측 잠금 → 별도 scoring

```bash
python -m scripts.research.pallet_clean_to_pose_transfer_v1.eval_student infer --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.eval_student score --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.eval_student oracle-freeze --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.eval_student oracle-score --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train_target_following infer --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.train_target_following score --seed 42
python -m scripts.research.pallet_clean_to_pose_transfer_v1.analysis --seed 42
```

평가128장의 모든 학생 예측과 D9를 먼저 잠근다. 자연가림99장의 T/R은 해당99장의 오차를 직접 모아 계산한다. 난도별 중앙값의 평균이 아니다. 교사 좌표에 가장 가까운 검출로 바꾸거나 oracle를 최종 성능으로 보고하지 않는다. TRAIN78 진단은 native 입력의 의사 타깃 추종이며 실제 정답 정확도나 증강 입력 전체의 학습 적합도가 아니다.

## 해석과 공개

기존217/현재78, 과거Clean19, legacy/검수 좌표, 반복DEV/독립TEST를 구분한다. conditional selector·bridge·추가 seed는 결과와 TRAIN 근거를 확인해 별도 사전 잠금이 있는 경우에만 실행한다. 예산이 남았다는 이유로 실행하지 않는다. 총학생10fit/selector1fit/GPU학습6시간 상한은 `RESOURCE_LEDGER.json`에서 확인한다.

최종 공개물은 한국어 보고서·집계 JSON·코드·숫자 그림이다. contact sheet와 원 RGB는 private에 남긴다. 관련 변경만 stage하고 실제 diff와 원격 SHA를 확인한다.
