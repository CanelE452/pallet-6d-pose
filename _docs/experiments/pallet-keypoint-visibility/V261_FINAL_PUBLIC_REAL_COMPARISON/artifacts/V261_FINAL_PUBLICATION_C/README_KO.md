이 helper는 Root가 지정한 최종 1회 비교의 기록을 **원 바이트로 게시**하기 위한 도구입니다. 모델·라벨·배열을 해석하지 않고, 디렉터리를 탐색하지 않으며, 성공·채택 여부는 Root가 공급한 보고서·상태·registry prefix에 따릅니다. 준비 단계에서는 공개 원본·현재 평가 payload·Git·네트워크에 접근하지 않았습니다.

1. `BUILD_ENTRY_FALSE_TEMPLATE.json`에서 `files`를 정확히 열거합니다. 각 행은 `{relative_path, group, source:{path,sha256,size_bytes}}`입니다. `coverage_contract`는 각 group의 모든 source binding 목록이며 파일 수나 성과로 자동 선별하지 않습니다. Root는 승인 flags와 같은 entry 전체(`root_scope` 제외)를 담은 별도 scope binding을 제공합니다. 출력은 고정 `R/V261_FINAL_PUBLICATION_PACKAGE`의 새 폴더입니다.
2. `size_bytes <= 90 MiB`인 원본은 identity 복사합니다. 그보다 큰 원본만 gzip(level6, mtime0, 빈 filename header)으로 압축해 `transport_gzip/<원경로>.gz.part0000…`에 저장합니다. 각 part는 최대64 MiB이며 마지막을 제외하면 정확히64 MiB입니다. 원본 SHA·크기, 압축 stream SHA·크기, 각 part SHA·크기를 manifest에 보존합니다. 빌더도 압축 원본의 복원 SHA·크기를 검사합니다.
3. `tools/restore_artifacts.py`는 빌더/복원기와 동일한 고정 코드입니다. package 내부 manifest·part만 사용해 **모든** 원본을 새 폴더에 복원하고 `FULL_RESTORE_QA.json`을 기록합니다. 원래 절대경로는 provenance 문자열이며 fallback은 없습니다. Root는 실제 전체 복원을 완료해야 stage를 실행할 수 있습니다.
4. stage entry는 정확한 MANIFEST와 전체 복원 영수증을 bind합니다. `registry_prefixes`는 정확히 세 기존 registry 경로를 key로 하고 각 값은 `{utf8,sha256,size_bytes}`입니다. UTF8 인코딩한 prefix의 SHA·크기를 확인한 후 기존 HEAD 바이트 앞에 붙입니다. 기존 tail은 한 바이트도 바꾸지 않습니다. 문구를 코드에 하드코딩하지 않습니다.
5. staging은 고정 clone/parent/destination, 원격 main 일치·tracked clean/index empty, 선언한 package+manifest+registry3의 exact NUL pathspec만 사용합니다. 복사·working bytes·index blob의 SHA·크기와 명시한 core whitespace를 확인하고 staged tree 영수증을 남깁니다. commit·push는 수행하지 않으며 Root가 정상 non-force `HEAD:main` 게시를 완료합니다.

```sh
python -B /ABS/R/V261_FINAL_PUBLICATION_C/portable_publication.py --build-entry /ABS/ROOT_BUILD_ENTRY.json
python -B /ABS/R/V261_FINAL_PUBLICATION_PACKAGE/tools/restore_artifacts.py --package /ABS/R/V261_FINAL_PUBLICATION_PACKAGE --output-dir /ABS/FRESH_FULL_RESTORE
python -B /ABS/R/V261_FINAL_PUBLICATION_C/stage_publication.py --entry /ABS/ROOT_STAGE_ENTRY.json
```

고정 parent는 `ff18ff3c489cd49a64956aaeab25b01278519f65`, clone은 `/home/minjae/Documents/pallet-visibility-publication-20261024`, 게시 경로는 `_docs/experiments/pallet-keypoint-visibility/V261_FINAL_PUBLIC_REAL_COMPARISON`입니다. archive 원본은 whitespace 정규화하지 않습니다. 실패는 새 출력·부분 파일·영수증에 남기며 자동 재시도·덮어쓰기·기존 시도 삭제는 없습니다.

검증은 기존 V256 빌더/복원기와 V259 stage 패턴의 폐쇄 계약을 바탕으로 한 가짜 바이트 identity 복사, 작은 toy 한도에서 gzip·shard·복원 및 결정성, 실제 고정 한도의 identity 전체 복원, false flag 차단, arbitrary raw registry suffix 보존입니다. 작은 toy 한도는 순수 `transfer` 테스트 인자에만 적용되며 Root 실제 build는90/64 MiB 상수를 강제합니다. 모델 성능이나 실제 대형 산출물 복원 성공을 준비 검사로 대신 주장하지 않습니다.
