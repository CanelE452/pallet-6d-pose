# 정적 코드 확인 — 원본 변경 없음

아래는 첨부한 실행 소스에서 그대로 읽은 줄이다. 코드의 존재와 실제 GPU 재실행은 구분한다.

## cli/driver.py:86–113
원본 SHA-256: `5d8ccc4ec71fe6667ac8e19126ba01fc95f3db8dbc77208086456a30897b5ff9`

발췌 파일: `source_excerpts/cli__driver.py.txt`

## cli/export.py:96–155
원본 SHA-256: `0b027e9158c9a3de451b50ea22460a65de93185e8e2f6f714f2b39f02e1f6434`

발췌 파일: `source_excerpts/cli__export.py.txt`

## cli/preflight.py:27–117
원본 SHA-256: `23b0924ab3955de1856109210861e9fc3e9d100210e89e8f296cbbba2f38a50e`

발췌 파일: `source_excerpts/cli__preflight.py.txt`

## pointline_v4/score_cache.py:1–35
원본 SHA-256: `1868efdef683a424cebe752fba423da7f8becddfa614338c311cff9c0027af6d`

발췌 파일: `source_excerpts/pointline_v4__score_cache.py.txt`

## cli/audit.py:10–59
원본 SHA-256: `39b652d499d6109ae6a541363eb0f07c39c51c69dfa06331782d20b5031c87de`

발췌 파일: `source_excerpts/cli__audit.py.txt`

## 판정

1. `cli/driver.py:86–113`: 각 군·seed마다 calibration→정책→validation 평가를 이어서 수행한다. 지시문의 12개 정책 전체 동결 후 validation 개봉 순서와 다르다. 정책 알고리즘이 validation을 직접 입력받거나 결과에 따라 재튜닝했다는 증거는 발견하지 못했다.
2. `cli/export.py:102–150`: 새 predictor의 반환값을 버리고 새 P3/P4만 캡처한다. 비교는 P4, 출력 점·confidence·선·교점은 기존 캐시다. 이는 P4 재현 확인이며 전체 추론 반환값의 동일성 검사는 아니다. 실제 불일치가 발생했다는 결론도 아니다.
3. `cli/preflight.py:27–74`: affine과 그 역행렬의 왕복, 같은 선 식으로 만든 점의 거리를 확인한다. 내부 산술 일관성 검사이지 실제 raw 영상과 feature의 정렬에 대한 독립 검사는 아니다. content mask의 수를 기록하지만 PASS 조건은 왕복/선식 오차만 본다.
4. `cli/preflight.py:79–117`: frame ID 교집합은 검사하지만 원본 이미지의 중복은 상류 boolean을 읽는다. 읽은 image-disjoint 값은 PASS의 직접 조건에도 포함되지 않는다. 현재 값이 true이고 데이터가 중복됐다는 증거는 없다.
5. 같은 파일:98–113은 ExportDataset.observation을 8회 읽는 경로를 감시한다. 전체 score_cache 모델 추론을 감시한 것은 아니다. 제공 코어 score_cache.py의 supervision_files_opened=0은 계측값이 아니라 선언 상수다. 정적 소스상 GT를 모델로 전달하는 경로는 보이지 않지만 런타임 계측 완료로 과장하면 안 된다. 이 코어 한계는 패키지를 제공한 분석자 측에도 책임이 있다.
6. `cli/audit.py:12–14`: CANDIDATE_CONTRACT.json이 REQUIRED 목록에서 빠져 있어 missing_required_artifacts=[]는 지시문 전체 준수 증명이 아니다. 후보 정의의 실질 내용은 상류 PROPOSAL_SPEC.json과 DATA_CONTRACT.json에 있으므로 파일 하나 결손을 곧 후보 변경으로 해석하지 않는다.
7. 같은 파일:35–39는 _docs/와 challenge/ 전체를 작업 범위에서 제외한다. unrelated_working_tree_changes=[]만으로 관련 없는 변경이 전혀 없었다고 보증할 수 없다.
8. 첨부 리뷰 manifest와 history에 외부 알림 실행 기록이 있으나 원래 지시문은 외부 알림을 금지했다. CLI에서 별도로 승인받았는지는 이 ZIP에 없다. 별도 승인이 없다면 범위 이탈이며, 숫자 재계산/성능 판정과는 별개다.

이 발견들은 12개 결과 숫자가 틀렸다는 증거가 아니다. 계산 재현, 데이터·추론 연결, 실행 순서, 문서 해석의 보증 범위를 각각 제한한다.
