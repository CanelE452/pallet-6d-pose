# 공개 계산 근거

기존 파일을 바꾸지 않고 gzip으로 압축한 사본입니다. [MANIFEST.json](MANIFEST.json)의 SHA-256은 압축 전·후 바이트를 모두 기록합니다. 압축을 풀면 원래 결과와 같습니다. GitHub에서는 다운로드하여 JSON/NumPy를 확인할 수 있습니다.

전체 원영상·원래 예측 입력·체크포인트는 이번 공개에 포함하지 않았습니다. 학습 및 전체 추론 재현에는 원래 입력이 필요합니다. 기존 receipt의 `/tmp` 등 절대 경로는 당시 출처이며 아래 매핑으로 공개 사본을 찾습니다.

| 원래 결과 경로 | 공개 압축 사본 | 원본 bytes |
| --- | --- | --- |
| `data/pallet/results/pallet_n3_static_closeout_v1/BOOTSTRAP_SESSION_COUNTS.npy` | [pallet_n3_static_closeout_v1__BOOTSTRAP_SESSION_COUNTS.npy.gz](pallet_n3_static_closeout_v1__BOOTSTRAP_SESSION_COUNTS.npy.gz) | 1040128 |
| `data/pallet/results/pallet_n3_static_closeout_v1/PAIRED_POSE_FRAME_DELTAS.json` | [pallet_n3_static_closeout_v1__PAIRED_POSE_FRAME_DELTAS.json.gz](pallet_n3_static_closeout_v1__PAIRED_POSE_FRAME_DELTAS.json.gz) | 935579 |
| `data/pallet/results/pallet_n3_static_closeout_v1/PNP_TRACES.json` | [pallet_n3_static_closeout_v1__PNP_TRACES.json.gz](pallet_n3_static_closeout_v1__PNP_TRACES.json.gz) | 13375167 |
| `data/pallet/results/pallet_n3_static_closeout_v1/POSE_HYPOTHESIS_CHANGES.json` | [pallet_n3_static_closeout_v1__POSE_HYPOTHESIS_CHANGES.json.gz](pallet_n3_static_closeout_v1__POSE_HYPOTHESIS_CHANGES.json.gz) | 1276747 |
| `data/pallet/results/pallet_n3_static_closeout_v1/RUNTIME_YOLO_RAW.json` | [pallet_n3_static_closeout_v1__RUNTIME_YOLO_RAW.json.gz](pallet_n3_static_closeout_v1__RUNTIME_YOLO_RAW.json.gz) | 311659 |
| `data/pallet/results/pallet_n3_static_closeout_v1/YOLO_SCORES.json` | [pallet_n3_static_closeout_v1__YOLO_SCORES.json.gz](pallet_n3_static_closeout_v1__YOLO_SCORES.json.gz) | 9421366 |
| `data/pallet/results/pallet_n3_completion_v3/evaluation/dope.json` | [pallet_n3_completion_v3__evaluation__dope.json.gz](pallet_n3_completion_v3__evaluation__dope.json.gz) | 2508975 |
| `data/pallet/results/pallet_n3_completion_v3/evaluation/resnet18.json` | [pallet_n3_completion_v3__evaluation__resnet18.json.gz](pallet_n3_completion_v3__evaluation__resnet18.json.gz) | 2755537 |
| `data/pallet/results/pallet_n3_completion_v3/reuse/PER_FRAME_SCORES.json` | [pallet_n3_completion_v3__reuse__PER_FRAME_SCORES.json.gz](pallet_n3_completion_v3__reuse__PER_FRAME_SCORES.json.gz) | 29882735 |
