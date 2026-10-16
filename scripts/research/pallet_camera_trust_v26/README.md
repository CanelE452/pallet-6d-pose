# V26 실제 카메라와 점/선 신뢰도 진단

실제 CAD 렌더180개, clean/좌우 잘림/물리 가림360관측, 고정 모델3시드. 새 학습은 없다. 기존 실사297장 재사용 DEV와 V23 신규 합성128장의 선 정확도·역할·보정 방향을 별도 진단한다. GT oracle는 실현된 모델 성능이 아니다.

- [설계·결과 노트](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_camera_trust_v26.md)
- [HTML](../../../data/pallet/results/pallet_camera_trust_v26/index.html)
- [등록 계획](../../../data/pallet/results/pallet_camera_trust_v26/PROTOCOL.json)

저장소 루트에서 실행한다. GPU 렌더는 NVIDIA EGL vendor를 이 프로세스에만 지정한다. 서로 다른 Python 환경을 사용하며 운영 모델이나 이전 완료 실험을 수정하지 않는다.

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json /home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/pallet_camera_trust_v26/render26.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/infer26.py --phase extract
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/infer26.py --phase parents
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/infer26.py --phase targets
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/trust26.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/score26.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/audit26.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/report26.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/delivery26.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_camera_trust_v26/complete26.py
```

완료된 실행의 원본/해시를 덮어쓰지 말고 새 비교는 별도 RUN에 수행한다. GT는 추론 후 평가에서만 읽으며 검출은 GT matching 없이 최고 confidence를 사용한다. 검출된 조건부 px와 전체 분모의 실패 포함 정규화 지표를 함께 기록한다. 실제 깊이 가시 마스크는 물체 표면 관측이며 cuboid 가상 선의 가시 GT가 아니다.
