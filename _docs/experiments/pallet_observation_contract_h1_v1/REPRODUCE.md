# 재현 안내

이 디렉터리는 공개 가능한 작은 파생 결과와 원소스 SHA를 담는다. private RGB·전체 후보 bank·GT·weight 파일은 복사하지 않는다. [입력 manifest](INPUT_MANIFEST.json)의 SHA가 맞는 승인된 원 캐시가 있어야 end-to-end 재현할 수 있다. 공개 자료만으로 원 학습/실사 생산을 모두 실행할 수 있다고 주장하지 않는다.

아래 명령은 저장소 root에서 실행한다. 경로는 실행자가 실제 private 입력 위치로 설정한다. 실제 전체 CLI와 종료코드는 [원 실행 명령의 경로 치환본](RECORDED_COMMANDS.json)에 있다. 변수 PYTHON/RUN_DIR/REPO_ROOT/PUBLIC_ROOT/RISK_RUN/D1_RUN과 manifest로 대응한 나머지 private 입력을 제공해야 한다. 재현은 기존 봉인 출력을 변경하지 않는 별도 run 위치에서 수행한다. AdamW를 다시 실행하면 실제 새 fit이므로 단순 검산을 위해 학습 명령을 반복하지 않는다. 실패한 구버전 source와 raw trace는 private에 보존했으므로 현재 source를 실행해 같은 logger 오류를 일부러 재현하지 않는다.

```bash
python -B -m unittest discover -s scripts/research/pallet_observation_contract_h1_v1 -t . -p 'test_*.py'
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_evaluation --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_no_b --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_policy --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.independent_b_query --help
python -B -m scripts.research.pallet_observation_contract_h1_v1.public_pack --help
```

source 역할은 `preflight.py` 입력 결합, `no_b.py/no_b_train.py` 특징 차단·고정 학습, `b_factorial_batch.py/b_mask_bounds.py/typed_observation.py` 관측 계약 진단, `policy.py/coverage.py` 고정 정책과 합성 분할, `predict.py/evaluate.py` 봉인·평가, `independent_*.py` 별도 계산, `figures.py/gallery.py/gallery_extremes.py` 사후 표시, `report_artifacts.py/public_pack.py` 공개 패키지로 나뉜다. 실제 source 목록·해시는 공개 pack manifest를 기준으로 한다. 구체적인 CLI 인자는 각 모듈 `--help`와 sanitization된 실행 receipt를 참고한다.

`verify_remote_publication.py`는 실제 fetch/ls-remote, 결과 커밋 포함, 두 새 namespace의 전체 파일 집합 및 모든 blob bytes를 manifest SHA·검산된 현재 파일과 대조한다. commit/push를 수행하지 않는 별도 도구이며, offline pack 통과를 원격 검증으로 바꾸지 않는다. 공개 코드만으로 이 조회 로직을 확인할 수 있다.

원 수치를 생산한 `predict.py` bytes는 새 namespace의 `predict_at_prediction_seal.py`에 그대로 보존했다. 현재 `predict.py`에는 봉인 후 누락 점수 fallback 기술 수리가 있으며 actual finite1,428개 출력은 동일하다. 원 실행 source hash까지 재현할 때는 별도 재현 checkout에서 보관본을 `predict.py` 위치로 배치하고, 기존 봉인/원 사용자 checkout은 수정하지 않는다. [source 연결](ARCHIVED_H1_FALLBACK_PUBLIC_SOURCE_BINDING.json)과 [전체 출력 동등성](ARCHIVED_H1_FALLBACK_TECHNICAL_AMENDMENT.json)을 함께 확인한다.

실행 순서는 P0 byte/shape/source binding → 학습 전 계획 고정 → 3seed no-B(기술 실패 포함5시도) → B2×2/mask/shape/typed 진단 → label-free split → 합성 정책 선택·한 번 확인 → 조건부 증강 게이트 → 26경로 예측 봉인 → 실사 참조 open/평가 → 독립 수치 검산 → 그림/갤러리 생성·실제 픽셀 검토 → 공개 transform/링크/누출 검사 → ordinary main 게시다.

원 파일을 그대로 복사한 CSV/선택 자세와 JSONsafe 파생의 original/public SHA·변환 검사는 PUBLIC_PACK_MANIFEST.json에 있다. 게시 영수증 PUBLICATION_RECEIPT.json은 실제 원격 결과 확인 이후에만 추가한다. 영수증의 A는 내용 커밋이며 후속 metadata B 자체를 미리 검증했다고 주장하지 않는다.
