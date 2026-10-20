# 현재 D1+H1 세 backbone 원실행 소스 — 정확한5파일 snapshot

사용자가 별도로 승인한5개 소스만 실행 당시 bytes 그대로 공개했다. 원파일은 변경하지 않았다. 기존106파일 감사 패키지와 별도 커밋이며 [기존 provenance](../paper_tables/THREE_BACKBONE_PROVENANCE.json)의 uncommitted overlay를 추적하는 보완 자료다. 이 파일들은 기준commit93c24d14에 들어 있지 않았고 SHA 메타데이터만 공개돼 있었다.

| 파일 | 역할 |
|---|---|
|upstream.py|기존 frozen 2D cache → native SubPix/current F → D1 입력|
|execution.py|새 backbone 입력에서 원 D1 관측·P1/WD0/G10/G13·특징 실행 및 재개|
|prediction.py|원3개 frozen H1와 각 원 normalizer 적용; fit 없음|
|evaluation.py|출력봉인 후 기존 physical C2 참조 채점|
|run_pipeline.py|고정계획·기술pilot·ROOT_GO 확인, 단계 실행·재개·영수증 기록|

[MANIFEST.json](MANIFEST.json)은 실행 시점SHA·원 상대경로·파일크기를 기록하고 [SCAN.json](SCAN.json)은 개인정보/비밀/비공개 절대경로 패턴 검사와 의존성을 기록한다. 현재 원본→snapshot SHA는5/5 정확히 일치한다. 검사 과정에서 자격증명 파일은 읽지 않았다. source에는 사용자home/tmp 절대경로나 토큰·개인키·email literal이 발견되지 않았다. `/seed1/last.pt`, `/candidates/` 같은 문자열은 동적경로 처리의 suffix이며 개인별 경로가 아니다.

## 역사적 실행과 재현 범위

실제 core는2026-10-20 03:01:26–04:56:41UTC, 전체 작업은05:18:01UTC 완료됐다. 기존 N3-only 비교가 아니라 frozen **현재 D1+H1 전체 경로**다. 동일 original YOLO-derived CAL/state/mask/prior/normalizer와3개 H1를 DOPE/ResNet18 입력에 이전했고 own-backbone H1/CAL 재학습은 없었다. 전체192 성공분모와 DOPE46미산출, 연속오차 available-only 한계를 유지한다.

이5파일의 공개는 **원실행 소스 버전의 누락을 보완**한다. 공개자료만으로 raw부터 일괄 재실행 가능하다는 뜻은 아니다. 원RGB/2D caches/weights/등록geometry/고정reference, 원D1 bank, 실행계획·기술pilot·ROOT_GO 영수증이 별도로 필요하다. 이 자료와 비승인 소스/원본 실행영수증은 이번에 복사·공개하지 않았다. source guard를 제거하거나 승인영수증을 가짜로 만드는 재현 방법도 제공하지 않는다.

허용된 기존 자료가 있는 **새로운 재현 checkout**에서만 다음처럼 모듈 위치를 구성할 수 있다. 원사용자 checkout이나 완료된 RUN을 덮어쓰지 않는다. 아래는 안내이며 이번 공개 과정에서 실행하지 않았다.

```bash
# NEW_REPRO_REPO: base93c24d143fa3901de1fa127b3156e8250d81dba8의 새 checkout
# SNAPSHOT: 이 폴더의 절대경로
mkdir -p "$NEW_REPRO_REPO/scripts/research/pallet_d1_three_backbones_20261020_v1"
cp "$SNAPSHOT/"*.py "$NEW_REPRO_REPO/scripts/research/pallet_d1_three_backbones_20261020_v1/"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

원실행 command의 변수화된 형태는 다음과 같다. **새로운 소유 RUN에 대한 실제 봉인 입력·계획·기술pilot·ROOT_GO가 먼저 준비·검증돼야 한다.** 기존 완료RUN 경로를 넣어 곧바로 재실행하지 않는다.

```bash
python -B -m scripts.research.pallet_d1_three_backbones_20261020_v1.run_pipeline \
  --repo "$NEW_REPRO_REPO" --private "$NEW_OWNED_RUN" \
  --original-d1 "$ORIGINAL_D1_RUN" --original-score "$FIXED_SCORE_ROOT" \
  --root-go "$NEW_OWNED_RUN/D1_THREE_BACKBONE_ROOT_GO_V1.json"
```

snapshot은 원코드의 guard와private 입력 요구를 그대로 보존한다. import되는 기존 tracked dependency는base93c24d14와 각 원seal에 연결된다. 신규fresh 환경의 완전 재현에 필요한 비공개 입력·계획 준비는 별도 작업이며 이 공개 과정에서 새 실험을 실행하지 않았다.
