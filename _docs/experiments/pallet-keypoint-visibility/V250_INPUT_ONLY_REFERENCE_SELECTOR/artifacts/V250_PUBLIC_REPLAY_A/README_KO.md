# V250 공개 저장소 SOURCE replay

NumPy가 설치된 Python으로 실행합니다. 동일 artifacts 아래의 `V250_PUBLIC_REPLAY_A`와 `V250_INPUT_REFERENCE_SELECTOR_B` 디렉터리 배치를 보존하세요.

```bash
python -B replay_public_selector.py --public-repo /absolute/public/repository --output-dir /absolute/fresh/replay
```

`public_dependencies.json`의 공개 저장소 상대 경로 10개만 SHA256/size로 검증합니다. 원 로컬 절대 경로는 provenance 문자열로만 비교하며 열거나 대체 경로로 사용하지 않습니다. 원 protocol af9 바이트를 복사하고 inputs8/metadata_provenance2의 path만 바꾼 protocol·entry·scope를 새 output에 보존합니다. 그 다음 exact-pin sibling runner를 한 번 호출합니다. 기존 output은 거절하며 실패를 보존하고 자동 재시도하지 않습니다.

수학은 sibling runner만 수행합니다. 다른 q8 입력 노드로 reference bank를 선택하고, target/role/object ID/oracle CSV 해석은 선택·점수의 봉인 뒤에만 허용됩니다. bank는 기존 SOURCE FIT-D 감독 자산이며 DEV는 이미 노출된 반복 진단입니다. query GTID를 selector 입력으로 넣지 않습니다. 새 optimizer·encoder·head forward, RGB/GT/weight/current/held 입력은 없습니다. scientific CSV/JSON/보고서는 절대 경로에 독립적이며 wrapper는 이 파일들을 고치지 않습니다. entry/COMPLETE/freeze/provenance 경로와 hash는 이동에 따라 달라질 수 있습니다.

이 준비에서는 CODE와 protocol/map metadata만 읽었습니다. 실제 SOURCE payload stat/hash/decode, runner 실행 및 결과 비교는 0회입니다. Root의 별도 실제 replay 지시 후 최종 portable package에서 한 번 실행하고 primary와 scientific output을 비교합니다. 실행 전 10개 bytes의 무결성 검산은 opaque이며 숫자/teacher 의미 해석을 하지 않습니다.
