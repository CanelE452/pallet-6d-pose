# 실행과 재현

환경: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`.
학습/원래 YOLO 추론은 GPU가 보이는 실행 환경이 필요하다.

저장소 루트 기준:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m pytest -q scripts/research/pallet_point_line_decisive_v6/tests
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_decisive_v6/preflight.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_decisive_v6/oracle_preflight.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_decisive_v6/cli_driver.py
```

관측 특징은 기존 `pallet_point_line_v5_local/export/`의 동결 캐시를 재사용한다.
각 seed의 실제 학습 update 수는 `cache_heads/*/COMPLETION.json`에 있다.
드라이버는 완료 cell을 재사용하고, 부분적으로 만들어진 cell은 덮어쓰지 않는다.
이미 있는 preflight/diagnostic 파일은 재실행 시 덮어쓰기를 거부한다.
처음부터 다시 비교하려면 새 실행 폴더를 명시해 경로를 변경하고 새 프로토콜을 고정해야 한다.

추가 진단 실행 명령:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_decisive_v6/stock_geometry_baseline.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_decisive_v6/quantization_diagnostic.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_line_decisive_v6/point_fit_diagnostic.py
```

원래 모델 비교·학습 적합도·격자 진단은 학습 가중치/threshold를 변경하지 않았다.
ORACLE_PREFLIGHT는 처음에는 동일한 식의 임시 CLI로 실행했고, 재현 가능한 코드를
`oracle_preflight.py`에 별도로 저장했다. 원래 동결 결과 파일은 다시 쓰지 않았다.
학습 적합도의 처음 role-aware 진단은 `POINT_FIT_DIAGNOSTIC.json`, 번호를 제외한 오차까지
확인한 재현 가능한 후속 진단은 `POINT_GEOMETRY_FIT_DIAGNOSTIC.json`이다.

평가에서 실패 frame은 분모에 포함한다. GT선/GT근처 초기값/최적 코너 대응은 모두
명시적으로 구분된 진단이며 추론에 몰래 넣거나 실제 성능으로 세지 않는다.
