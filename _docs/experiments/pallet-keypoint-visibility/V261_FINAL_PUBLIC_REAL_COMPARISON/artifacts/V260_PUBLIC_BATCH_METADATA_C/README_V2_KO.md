# V260 opaque downloader V2: same-dataset official redirect only

V1 code/metadata/false entry와 Root의0-file 실패를 보존했습니다. V2는 metadata collector를 바꾸지 않고 공식 file-ID URL에서 지정된 KU 저장소로 가는 정상 redirect만 허용합니다.

Root trueentry는 `approved_actual_public_allowlist_download=true`, `Rootapproved_same_dataset_official_redirect=true`, `opaque_only=true`, `training_admission=false`, `automatic_retry=false`와 다음 exact scope를 선언해야 합니다.

```json
{"scheme":"https","host":"rdmo.icts.kuleuven.be","port":443,"path_prefix":"/dataverse/10.48804/DJLDFS/","userinfo":null}
```

최초 요청은 기존284 bundle/1142 fileID allowlist의 `https://rdr.kuleuven.be/api/access/datafile/<ID>`와 일치해야 합니다. 한 번의 official→위storage redirect만 허용하며 host/DOI/port/userinfo 변경, path traversal, 연쇄 redirect는 거부합니다. 최종 HTTP200은 원 exactURL 또는 같은dataset storage prefix만 가능합니다. 기존 원 publisher MD5/size 검증과 파일별SHA256, freshout/noresume/noautoretry는 유지됩니다.

실제 signed URL은 HTTP client 메모리에서만 전달합니다. 영수증에는 scheme/host/port/path와 **query 전체 SHA256·키 이름만** 기록하고 query값이나 full signedURL은 저장/출력하지 않습니다. 실패도 URL이 포함될 수 있는 exception 문자열/원인 chain을 억제하고 error type/status/stage만 남깁니다. 실제 credential/query token은 C가 보거나 테스트에 사용하지 않았습니다.

`test_redirect_scope_v2.py`는 가짜 URL만으로6개 범위를 검증했고8478d6 exit0/PASS입니다. 네트워크/Source bytes 접근0이며 collector 및 기존 nondownload 함수 AST가 동일함을 확인했습니다. 전체 line diff와 역patch byteexact 증거는 별도 receipt에 있습니다. 이 수정은 다운로드 transport만이며 decode/pose/label/model/training 범위를 늘리지 않습니다.

Root는 `DOWNLOAD_ENTRY_FALSE_TEMPLATE_ALL284_V2.json`을 바탕으로 codepin/flags/freshoutput을 봉인한 trueentry를 작성한 뒤 실행할 수 있습니다.

```sh
python -B public_batch_v2.py --download-entry ROOT_FUTURE_TRUE_DOWNLOAD_ENTRY_V2.json
```

C는 actualdownload를 실행하지 않았습니다.
