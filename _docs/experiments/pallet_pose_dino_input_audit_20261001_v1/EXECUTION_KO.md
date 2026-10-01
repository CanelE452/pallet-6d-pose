# 실제 실행과 범위

직전 goal turn은 비대칭 Huber 결과를 GitHub main `232e2bb3`에 push하고 원격 파일 60개를 검증한 진전이었다. 안정적인 T/R 공동 개선 목표는 달성되지 않아 유지했다.

이번 실행은 새 입력이 실제 후보 위치와 일치하는지 확인하는 단계다. 새 학습, optimizer 갱신, candidate argmin, source VAL 품질 평가, 실사 평가는 모두 0회다. 기존 목표·합격 조건은 줄이지 않았다.

## 실행 환경과 명령

Python은 `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B`를 사용했고, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/pallet-mpl MPLBACKEND=Agg`로 실행했다.

```bash
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.cache_contract
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.visual_features
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.freeze_train seal
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.freeze_train freeze
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.verify_train_inputs --selfcheck
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.verify_train_inputs --write
python -B -m scripts.research.pallet_pose_dino_input_audit_20261001_v1.padding_audit
```

입력 protocol 봉인은 session 1742 exit0, 실제 GPU 추출은 session 21971 exit0이었다. RTX 3080에서 이미지 2,597회 forward를 실행했고 전체 2,598행 및 원래 전부 무효 1행을 보존했다. 저장·해시를 포함한 생산 영수증의 경과 시간은 213.85초다. FP16 token 파일은 4,692,861,056 bytes, descriptor NPZ는 29,774,931 bytes이며 로컬에 보관한다. 코드·설정·해시는 공개하지만 이 대용량 캐시와 원본 데이터는 GitHub에 포함하지 않는다.

샌드박스 안의 `nvidia-smi`는 driver에 접근하지 못했으나 허용된 외부 실행에서 RTX 3080을 확인했다. 실제 GPU 작업은 그 경로로 한 번 실행했다. xFormers 미설치·TypedStorage 경고·CuDNN workaround 메시지가 있었으며 종료 코드는 0이었다. backbone 다운로드·업데이트는 없었다.

독립 검산 session 5629는 exit0/PASS였다. 저장 token 이후의 투영·bilinear sampling·지원점·평균·정규화와 모든 유효 후보를 별도 scalar/Torch float64 구현으로 확인했다. 최대 절대차는 약 1.71e-6, 사전 고정 허용오차 대비 최대 비율은 약 0.187이다. 이 검산에서 backbone forward 자체를 다시 실행하지는 않았다. 기존·신규 crop/FP32 전처리 형식의 별도 검산은 [순수 전처리 영수증](PREPROCESSING_PURE_CHECK.json)에 있다.

## 감사 중 수정한 부분

캐시 재사용 감사의 최초 실행은 과거 localization signature의 행 순서를 잘못 가정하여 종료했다. 당시 출력·forward·학습은 0회였고, 과거 고정 생산 코드의 sorted unique 순서에 맞춰 감사 코드만 고쳤다. 최종 session 79080 exit0/PASS이며 자세한 기록은 [캐시 감사](CACHE_CONTRACT.json)에 있다. 과거 데이터·특징은 바꾸지 않았다.

실제 새 추론 전에는 기존 DINO 경로와 같은 TF32 비활성 설정, RAW 밖 쓰기 차단, 입력 해시 연결을 검토했다. 꼭짓점 합산 순서를 정렬하여 같은 샘플 집합의 순열이 FP32 합산 오차를 바꾸지 않게 한 뒤 protocol을 봉인했다.

독립 검산의 실제 실행 전에는 D1/D2/D3의 anchor 차분 진단이 각자의 동명 가설 대신 실제 R0 anchor descriptor를 빼도록 고쳤다. 입력 생산값·추론·후보는 바꾸지 않았다.

padding 진단 session 1345는 exit0이었다. 지원점 일부가 준비영상의 반사 영역에 있는 것을 확인했으므로, 후속 학습 전에 원본 영상 영역을 구분하는 별도 입력 수정을 진행한다. 이번 봉인된 descriptor나 평가 기준을 사후 변경하지 않는다. 결과·이미지·치수를 공개하는 일도 성능 목표 달성을 뜻하지 않는다.
