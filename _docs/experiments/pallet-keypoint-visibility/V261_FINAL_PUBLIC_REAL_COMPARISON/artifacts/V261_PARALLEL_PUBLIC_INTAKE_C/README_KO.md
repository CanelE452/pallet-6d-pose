V4는 공개 파일 수집의 I/O만 병렬화합니다. 메타데이터에서 고정한 284묶음·1142개 파일, 원본 MD5·크기, coarse-directory TRAIN/DEV 구분을 바꾸지 않습니다. 디코딩·라벨 생성·모델 실행·학습 권한은 없습니다.

V3는 Root가 속도 개선을 위해 의도적으로 중단했습니다. 완료 기록 210개(이전 복사 53개와 새 GET 157개)를 모두 새 출력 폴더에 직렬 복사하며 SHA256·크기·공식 MD5를 다시 확인합니다. `.part`는 복사하지 않습니다. 나머지 932개만 `ThreadPoolExecutor(max_workers=8)`로 각 파일당 한 번 요청합니다. 복사 실패 시 GET으로 대체하지 않습니다.

실행 전 Root는 `DOWNLOAD_ENTRY_FALSE_TEMPLATE_V4.json`의 세 flags `approved_actual_public_allowlist_download`, `explicit_user_authorized_resumption`, `approved_bounded_parallel_intake`를 true로 바꾼 새 entry를 발행합니다. 기존 `Rootapproved_same_dataset_official_redirect=true`, `opaque_only=true`, `training_admission=false`, `automatic_retry=false`, `max_workers=8` 및 정확한 code/utility/이전 entry/실패/210개 reuse bindings는 필요합니다. C는 실제 파일 본문이나 네트워크에 접근하지 않았습니다.

```sh
python -B /ABS/R/V261_PARALLEL_PUBLIC_INTAKE_C/public_batch_v4.py --download-entry /ABS/ROOT_TRUE_ENTRY.json
```

동일 이름의 출력 폴더가 있으면 거부합니다. 공식 RDR 파일 ID URL에서 같은 DOI의 승인된 HTTPS storage origin으로 한 번 이동하는 V3 handler를 그대로 사용합니다. 서명 query 값·전체 storage URL·원문 예외는 기록하지 않습니다.

실패 또는 중단 시 대기 작업을 취소하고 이미 실행 중인 요청을 마친 뒤 성공 파일 기록과 `.part`를 보존합니다. 성공·실패의 완료 목록은 `publisher_relative_path`순이며, 재시도·자동 재개는 없습니다. 개별 파일의 정확한 내용 SHA는 네트워크 완료 순서와 관계없지만, 완료 기록의 시간·HTTP 진단은 실행마다 같다고 주장하지 않습니다.

준비 검사는 AST/단일 EOF, false-entry 차단, 가짜 19작업의 동시성 상한 8·중복 요청 없음·정렬된 기록, 주입 오류의 비재시도/완료 수집/비밀 예외문 억제, 기존 redirect 범위, 실제 V3 완료 **메타데이터만**의 210개 binding 일치성입니다. 원본 210개 파일은 검사 과정에서 stat/hash/read하지 않았습니다. V1–V3는 수정하지 않았습니다.
