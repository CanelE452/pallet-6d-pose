# Portable package verifier

표준 라이브러리만 사용하며 실제 프로젝트 자료를 자동으로 찾지 않습니다.

```sh
python3 verify_package.py --package /absolute/path/to/package --manifest MANIFEST.json
```

`--manifest`는 package 기준의 상대 POSIX 경로입니다. Manifest 형식:

```json
{"files":[{"path":"reports/REPORT.md","sha256":"<64 hex characters>","bytes":123}]}
```

Manifest 자기 파일은 `files`에 넣지 않습니다. 검증기는 모든 선언 경로를 먼저 검사한 뒤 각 파일의 SHA-256과 실제 byte 수를 확인합니다. 절대 경로, `..`, `.`/빈 경로 부분, 역슬래시, 중복 경로, symlink 및 일반 파일이 아닌 항목을 거절합니다. 대소문자와 무관한 `v112`, 모든 `.pyc`와 `__pycache__` 경로는 파일 open/stat 전에 거절합니다. 따라서 별도 제외 대상으로 지정된 두 pyc도 포함됩니다. Package 경로의 조상 디렉터리부터 `O_NOFOLLOW`로 열며 symlink를 따라가지 않습니다.

성공 시 JSON을 stdout에 출력하고 exit 0, 실패 시 JSON을 stderr에 출력하고 exit 1입니다. 파일을 수정하지 않으며 디렉터리 tree walk·추가 파일 열거·문서 안 pointer 추종을 하지 않습니다. **Manifest 밖 파일의 존재/부재는 인증하지 않습니다.** Root가 manifest를 정확한 allowlist로 구성해야 합니다.

```sh
python3 -B toy_verify.py
```

토이는 임시 디렉터리에 직접 만든 bytes만 사용합니다. SHA/크기, CLI 성공, lexical 경로 차단, manifest 자기파일·중복, leaf/ancestor/package symlink 거절을 검사했습니다. 실제 패키지·SOURCE·모델·prefix·보호 자료 접근은 0입니다.
