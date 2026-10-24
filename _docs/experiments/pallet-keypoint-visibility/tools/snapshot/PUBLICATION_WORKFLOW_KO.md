# 실험 원문과 검토 자료 게시 절차

원본은 수정하지 않는다. `archive/task-2/<원래 루트 이름>/<상대경로>`에 원바이트를 보관한다. 1 MiB 이상 텍스트는 원문 SHA256을 유지하는 32 MiB 단위 gzip으로 보관할 수 있다. 논리 경로의 `.parts.json`은 실제 파일이며, 압축 해제 후 순서대로 이어 붙인 바이트의 SHA256을 검증한다. Portable Markdown은 별도 파생물이다.

현재 후보 목록은 `v5_final_candidates/PUBLICATION_FILE_MANIFEST_CANDIDATE.jsonl`이다. 모든 문서·코드·표·검토 이미지와 개별 HTML/MD 직접 이미지 참조를 보존한다. 대형 원 RGB/GT 마스크 은행, 특징·P3·입력 캐시, 외부 backbone 가중치, 환경, 원 센서 기록은 제외 사유를 남긴다. 소형 자체 학습 체크포인트 201개와 저장 수치 출력 344개는 별도 명시 목록이다. NPZ는 헤더만 검사했으며 배열·pickle 가중치는 역직렬화하지 않았다.

`package_publication_snapshot.py --prepare`는 발명 배열과 임시 디렉터리만 검사한다. 실제 복사는 root가 `ROOT_COPY_PLAN_TEMPLATE_APPROVED_FALSE.json`을 별도 승인 계획으로 작성하고 코드·후보 목록 SHA를 확인한 뒤 `--approved-plan <계획.json>`으로 실행한다. 기본 원문 `INCLUDE` 외에는 `source_root_id:source_relative_path`별 whitelist가 필요하다. 저장소 명령·commit·push는 이 도구가 실행하지 않는다.

게시 후 독립 검증기는 원문 복사 SHA 또는 gzip 조각 SHA와 압축 해제 원문 SHA를 확인한다. HTML/MD 링크 변환은 원본을 건드리지 않고 portable 파생본에만 적용한다. 논리 JSONL/CSV 링크가 압축되면 실제 `.parts.json` descriptor로 안내하며 무손실 압축임을 표시한다. 코드·표·실패 기록을 크기만으로 누락하지 않는다. 같은 SHA의 여러 원경로는 모두 유지되며 Git의 동일 blob으로 중복 저장이 줄어든다.

후속 실험은 완료 범위와 최종 영수증을 먼저 고정한다. 새 phase만 담은 동일 스키마 delta 후보 manifest와 별도 승인 계획을 만들고, `output_manifest_relative`/`output_complete_relative`를 날짜·phase별 새 경로로 지정한다. `allow_existing_identical=true`는 이전 원문 또는 gzip이 SHA까지 동일할 때만 재사용한다. 기존 경로 내용이 달라지면 덮어쓰지 않고 중단하므로 이전 snapshot을 보존한 새 논리 경로가 필요하다. Root는 원문·파생본·검산을 완료한 뒤 main에 512 MiB 이하 논리 파일 묶음으로 commit/push한다. 미래 실험 결과도 동일 절차로 반영한다.

V112 이름은 metadata stat보다 먼저 거부한다. Symlink는 대상 조회·resolve·따라가기를 하지 않는다. 자격증명 이름과 환경은 먼저 제외하고, 게시 대상 텍스트의 credential 검사는 독립 검증 단계에서 별도로 수행한다. 제외된 트리의 미측정 크기는 0이 아니라 UNKNOWN이다.
