# 실행 기록과 재현

[확인] 진입 지시의 E1만 실행했고 E2는 clean pool gate에서 중단했다. 다음 명령은 기존 의존성/로컬 모델/원본 이미지가 있는 저장소 루트 기준이다. GPU 단계는 host CUDA 접근이 필요하며 GPU 사용 전 다른 작업과 온도를 확인한다. 패키지 설치/시스템 변경은 없다.

```bash
PYTHON=/home/minjae/anaconda3/envs/pallet-yolo26/bin/python
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run preflight
$PYTHON -m pytest -q scripts/research/pallet_occlusion_refiner_transfer_v1/test_contracts.py scripts/research/pallet_posefix_large_error_v1/test_core.py
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run cache
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run geometry
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run infer
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run freeze
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run score
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.run gate
$PYTHON -m scripts.research.pallet_occlusion_refiner_transfer_v1.report
```

[확인] 완료 artifact는 다르면 overwrite를 거부한다. 다음 실험에서 조건/코드를 바꾸려면 새 namespace를 사용해야 한다. 현재 driver는 E2 학습을 구현하거나 허용하지 않는다. 새로운 eligible pool을 확보해도 target 생성/parity/2×2 protocol을 먼저 구현·고정해야 한다.

[확인] 최종 검사: 계약 단위검사11개 통과, 예측 보존 검사1,624개, 과거 R0 프레임 지표344개 재현, 주요 입력/모델/소스 binding39개 재검증. E2 전용 target parity/OCC R0 재추론/노출량 검사는 학습 미실행으로 NOT_RUN이다. 통과로 위장하지 않는다.

[확인] 로컬 HTML9개에서 이미지/내부링크1,214개 경로 존재 검사 통과. 기존 Chrome debugging endpoint는 종료돼 실제 브라우저 렌더 검사는 완료하지 못했다. `HTML_LINK_AUDIT.json`은 파일 경로 검사이지 스크린샷 검증이 아니다.

[확인] 시작/종료 HEAD `73bfe38a259b3c846e49a98fb82c44578f6e2248`, tracked diff 없음. 새 실험 문서/코드/출력은 untracked이며 commit/push하지 않았다. 다른 기존 untracked 연구 자료도 그대로 보존했다.
