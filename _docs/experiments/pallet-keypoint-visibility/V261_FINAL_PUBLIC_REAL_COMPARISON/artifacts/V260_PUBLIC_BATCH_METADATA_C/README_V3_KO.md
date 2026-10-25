# V3 explicit resumption with53 verified completed files

V1/V2 code와 실제 실패/중단 파일은 그대로 보존합니다. 사용자의 마지막 공개실사 비교1회 승인 뒤 Root가 새 trueentry와 fresh `public_284_raw_v3` 출력에서 명시적으로 재개하는 transport이며 automatic_retry=false입니다.

Root entry는 기존284 exact bundle/1142 file allowlist와 V2 redirect scope를 그대로 유지하고 `explicit_user_authorized_resumption=true`, `previous_download_entry`, `previous_failure`, `expected_reused_completed_files=53`, `reuse_files`를 선언해야 합니다. 재사용 각 항목은 `{publisherfile_id,source:{path,sha256,size_bytes},original_MD5}`입니다. 이 목록은 exact V2 FAILURE.completed_downloads와 metadata에 모두 일치해야 합니다. `.part`는 절대 재사용하지 않습니다.

Source53개를 실제 copy하는 단계에서 ancestor/leaf symlink와 금지경로를 차단하고 stream SHA256/size/publisherMD5를 전부 확인합니다. mismatch는 partial보존/즉시실패이며 GET로 대체하지 않습니다. 검증된53개는 새out에 copy하고 나머지1089개만 같은 공식dataset HTTPS scope로 GET합니다. 기존 source/partial은 수정하거나 삭제하지 않습니다. 새 complete에는 reuse/GET 수와 파일별 operation/sourcebinding을 남깁니다. signedquery값과 fullsignedURL을 기록하지 않는 V2 규칙도 동일합니다.

C는 Root의 V2entry와 FAILURE **metadata만** 읽어53-binding/1142-plan을 검증했습니다. 실제53 sourcefile/payload 접근0, network0입니다. 별도 owned 임시폴더의 invented bytes로 정상 copy와 badSHA 실패/noGETfallback만 검증했습니다. Root intentional SIGINT는 data-quality failure로 재해석하지 않습니다.

`DOWNLOAD_ENTRY_FALSE_TEMPLATE_ALL284_V3.json`에는53 reuse sourcebindings와1142 원allowlist가 모두 고정돼 있습니다. Root trueentry에서 download/redirect/resumption flags를 명시적으로 활성화한 뒤 실행합니다.

```sh
python -B public_batch_v3.py --download-entry ROOT_TRUE_DOWNLOAD_ENTRY_V3.json
```

metadata split/label/query/모델/threshold/recipe 변경, decode, 학습 또는 평가를 이 downloader가 수행하지 않습니다. 이후 classifier 비교의 경계는 별도 Root policy입니다.
