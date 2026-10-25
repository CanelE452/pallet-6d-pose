# V256 전체 바이트 공개 패키지 도구

이 폴더는 CODE-only 준비입니다. 실제 SOURCE 배열·RGB·체크포인트·결과 본문은 아직 읽지 않았습니다. Root가 완료 후 exact allowlist와 scope를 넣어 한 번 실행합니다. 학습·성능 비교·결과 선택을 하지 않습니다.

`build_portable_package.py --entry ROOT_TRUE_ENTRY.json`은 4개 raw view NPY, 6개 canonical feature, 전체 labels/samples, 실패 포함 V1/V2 코드·사전 규칙·QA, 두 arm의 CP/log/draw/results, 실제 보고서·그림을 그룹별 `coverage_contract`와 대조합니다. 경로 추측이나 디렉토리 treewalk는 없습니다. Root는 actual COMPLETE의 모든 선언 leaf를 contract에 넣어야 합니다. 지금은 actual 목록의 완전성을 주장하지 않습니다.

Root entry의 `files` 각 행은 `relative_path`, `source={path,sha256,size_bytes}`, `group`, `encoding=raw|gzip`입니다. `coverage_contract`는 각 group의 exact source binding 목록입니다. false entry의 빈 목록/미정 값으로는 실행할 수 없습니다. 새 SOURCE 파일을 임의로 더 찾거나 원 메타데이터의 절대 경로를 따라가지 않습니다.

큰 NPY는 NumPy로 읽지 않고 원래 파일의 **헤더를 포함한 전체 바이트**를 1MiB씩 gzip으로 보존합니다. filename 헤더는 비우고 mtime=0, level=6입니다. 압축 후 전체 원복 SHA/size를 검사합니다. 압축본이 Root가 지정한 단일 파일 한도를 넘는 경우에만 64MiB 조각으로 나누며, 조각을 순서대로 이은 압축 바이트의 SHA도 검사합니다. 원본 NPY/CP/RGB는 변경하지 않습니다.

패키지 루트에서 물리 파일 검증:

```bash
python -B tools/restore_artifacts.py --package /ABS/PACKAGE --verify-only
```

전체 원본 바이트를 새 디렉토리에 복원:

```bash
python -B tools/restore_artifacts.py --package /ABS/PACKAGE --output-dir /ABS/FRESH_RESTORED
```

raw 보고서·그림·작은 파일은 원래 package-relative 경로에 바로 있고, 압축된 큰 파일은 `transport/` 아래에 있습니다. 복원 뒤에는 모든 logical 경로에 원래 파일 바이트가 있습니다. restore는 공개 package-relative 목록만 사용하며 로컬 원본이 없어도 됩니다. `RESTORE_QA.json`은 각 복원 SHA/size를 기록합니다. `--verify-only`는 압축 파일을 포함한 물리 파일 검증이고, 실제 원복 검증은 두 번째 명령에서 수행합니다.

임시 가짜 바이트와 struct로 만든 NPY 헤더로 9항목 검사를 했습니다. split 경로만 작은 시험 한도/조각으로 검증했으며 production 조각은 고정 64MiB입니다. SOURCE 수치/배열/모델 decode는 0입니다. Git add/commit/push는 이 도구가 수행하지 않습니다.
