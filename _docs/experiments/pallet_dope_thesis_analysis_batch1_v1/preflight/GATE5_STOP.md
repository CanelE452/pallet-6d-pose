# 보호 FINAL 경계 확인으로 게시 중단

[확인] main push를 수행하지 않았다. 사용자의 게이트5 “보호 FINAL의 영상·정답 내용을 열지 않는다(메타데이터 집계만)”의 적용 범위를 확인해야 한다.

[확인] T0 메타데이터의 선언 이미지 SHA를 대조하면 PLASTIC184 중56장이 물리 FINAL과 같다. DEV292와 FINAL의 선언 이미지 SHA 중첩은0이다. frame ID는 서로 다른 alias이므로 이미지 SHA로 확인했다. 근거: `cache/METADATA_RECORDINGS.json#/populations/PLASTIC184`, `#/protected_final_image_sha256`, `preflight/GATE5_STOP.json`.

[확인] T2는 `code/t2_bootstrap.py:52,62`에서 기존 `PLASTIC184_SCORES.json`을 사용했다. `code/t2_bootstrap.py:76`의 모델 간 기준점 대응 검사에서 `code/t1_cached_bias.py:32–33`의 `frame_signature`가 복사된 `reference.xy`를 읽었다. CI 계산 자체는 기존 `error_px`의 정답 수·관측 수·전체 기준점 수만 사용했다. 독립 감사도 같은 파생 SCORES를 읽었다.

[확인] 보호 FINAL의 원본 RGB·annotation 파일 본문 열람은0회다. 그러나 “FINAL 기준점 좌표를 전혀 소비하지 않았다”는 표현은 사실이 아니다. 기존 파생 SCORES의 좌표를 확인한 사실과 원본 파일 미열람을 구분한다.

[확인] 계산 전 `FIXED_PLAN.json`에 기존 파생 점수 캐시 사용을 허용한 문구는 어시스턴트의 해석이었다. 사용자의 별도 승인이 아니다. T5 예외 허용을 FINAL 파생 정답 좌표 사용의 허용으로 확대하지 않는다. T2 실행 전에 이 경계를 확인하고 멈췄어야 했다.

[확인] 원본 두 실험 폴더598개 파일과 R0의 작업 전후 SHA는 모두 같았다. 근거: `hashes/COMPARISON.json`의 `all_unchanged=true`, `differences=[]`.

현재 필요한 결정은 기존 파생 SCORES 사용 사실을 명시하고 PLASTIC184 보조 결과를 포함해 게시할지, PLASTIC184 보조 결과를 게시 대상에서 제외할지이다. 답변 전에는 추가 분석·결과 확정·push를 진행하지 않는다. 기존 요청의 stop 기본값을 적용한다.
