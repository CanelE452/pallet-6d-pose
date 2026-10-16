# GPU 드라이버 불일치 복구 및 실험 재개

> **최신 상태: 사용자가 재부팅을 금지했고, 재부팅 없이 실험 프로세스의 CUDA 복구를 검증했다.** 아래의 재부팅 제안은 이전 진단 기록이며 실행하지 않는다. `data/pallet/results/gpu_recovery_20260911/no_reboot_recovery.json`의 PASS가 근거다.

## 재부팅 없는 복구 완료

이전 V44에서 이미 공식 NVIDIA 580.173.02 라이브러리를 프로세스에만 적용한 기록과 파일이 있었다. 이를 먼저 확인하지 못한 것이 앞선 진단의 누락이었다. 기존 공식 아카이브 해시를 V44 기록과 대조하고 CUDA 관련5개 라이브러리를 Git 제외 경로 `.cache/nvidia-runtime/580.173.02/`에 보관했다.

`scripts/diagnostics/with_gpu.py`는 정상 시스템에서는 원래 환경을 사용하고, 불일치가 있을 때에만 현재 로드 모듈과 정확히 같은 버전의 로컬 라이브러리를 자식 프로세스에 적용한다. 매 실행 시 파일 해시와 NVML 인식을 확인한다. 전역 환경·시스템 드라이버·바탕화면은 변경하지 않는다.

```bash
python scripts/diagnostics/with_gpu.py nvidia-smi
python scripts/diagnostics/with_gpu.py /home/minjae/anaconda3/envs/pallet-yolo26/bin/python YOUR_SCRIPT.py
```

실제 검증: RTX3080 인식, PyTorch CUDA 사용 가능, convolution 순전파/역전파와 유한한 비영 gradient PASS, CPU 대비 최대 오차 `6.67572021484375e-06`. 시스템의 설치/로드 버전 불일치는 남아 있으며, **위 실행기를 사용하는 프로세스의 GPU 사용이 복구된 것**이다. 일반 `nvidia-smi`나 이 실행기를 통하지 않은 새 프로세스까지 고쳤다고 주장하지 않는다. 재부팅·로그아웃·프로세스 강제 종료는 하지 않았다.

2026-09-11. 사용자가 네 군 치수·대칭 실험 진행 중 GPU 문제 해결을 요청했다.

## 확인된 원인

- 실행 중 커널: `6.8.0-136-generic`.
- `/proc/driver/nvidia/version`: 메모리에 로드된 모듈 `580.173.02`.
- `modinfo nvidia` 및 설치된 NVIDIA 라이브러리: `580.178.04`.
- 실제 호스트의 `nvidia-smi`: NVML driver/library version mismatch.
- 실제 호스트의 PyTorch `2.1.1+cu118`: CUDA 사용 불가, 초기화 오류804.
- DKMS: `580.178.04`가 커널136 및138 양쪽에 설치됨.
- 새 커널138의 initrd와 vmlinuz 존재, 모듈 vermagic 일치. Secure Boot 비활성.
- `dpkg --audit` 출력 없음. OS의 reboot-required 표시 있음.
- Xorg, GNOME, VS Code, Chrome 등 현재 데스크톱이 GPU를 사용 중.

이는 GPU 하드웨어 고장으로 판정한 것이 아니다. 새 라이브러리와 구형 로드 모듈의 불일치가 확인됐다. [Ubuntu NVIDIA 설치 문서](https://ubuntu.com/server/docs/nvidia-drivers-installation/)와 [NVIDIA 업그레이드 문서](https://docs.nvidia.com/dgx/dgx-el9-user-guide/upgrading.html)의 드라이버 업데이트 후 재시작 설명에 해당한다.

## 복구 상태

**재부팅 금지라는 사용자 제약을 유지한다.** 재부팅·재설치·다운그레이드·전역 라이브러리 변경·프로세스 강제 종료를 하지 않았다. 위의 프로세스 전용 실행기로 CUDA를 복구했다. 검사 결과에서 `system_installation_consistent=false`와 `active_runtime_matches_loaded=true`를 구분한다. `PASS=true`는 실제 활성 라이브러리와 커널이 맞고 CUDA 순전파/역전파가 통과했다는 뜻이다.

## 보존한 원래 실험

사용자가 승인한 목표는 기존 YOLO 기준 **A 기존 / B 대칭 로스만 / C 치수만 / D 둘 다** 비교다. [공유 대화 대조 기록](../notes/pose_geometry/shared-dimension-symmetry-review.md)을 기준으로 한다.

- `scripts/research/pallet_dimension_symmetry_v1/`의 네 군 구현·검사·학습·평가를 완료했다. 복구된 GPU에서 실제 합성2,048장 학습 및512장 평가, 12개 모델 총12,000업데이트를 실행했다.
- 프로토콜 동결, 기존 출력 및 로스/gradient 일치 검사, GT 재투영, observation-only 체크포인트 재생과 독립 평가 재계산을 통과했다.
- 기존 공간 점 특징을 고정하고 마지막 점/sigma 출력층과 치수 FiLM을 학습하는 제한된 비교다. 전체 YOLO 학습을 완료했다고 확대하지 않는다.
- 치수 입력은 같은 추가 학습 A 대비 세 시드 모두0.29~0.58% 개선했지만, 모든 시드에서 기존 YOLO를 넘는 기준에는 미달했다. 대칭은 초기 C2 객체1,364개 중2개만 다른 정답을 선택했다. [실제 결과](../notes/pose_geometry/dimension-symmetry-2x2-results.md) 참조.
- GT의 물체 기준 치수와 camera-dynamic 점 번호를 구분하고, 실제 물체 C1/C2를 사용한다. C4는 감사된 데이터에 없다.
- 기존 출력 정확도의 초기 보존, 동일 seed·배치·예산, 동일 유효 대칭 평가가 필요하다. 실패한 새 head 학습을 반복한 결과를 치수/대칭 자체의 실패로 단정하지 않는다.
- GPU 복구 완료 범위는 전용 실행기를 쓰는 프로세스다. 시스템 전역 드라이버가 바뀌었다고 보고하지 않는다.

저장소의 기존 변경 파일은 보존했다. 기존 데이터·GT·체크포인트는 수정하지 않았다.
