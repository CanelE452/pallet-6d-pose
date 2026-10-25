# V223 저장 SOURCE 산술·보고서 QA

PASS. 두 모델의 저장 SOURCE 점수 6개를 원행에서 다시 계산하여 CM·D/E precision/recall·U, native별 표, paired 회복/손상과 triplet 판정이 모두 일치했다. JSON metric 40개와 CSV16행을 검산했다.

- CF DEV: C68 [[14,4,0],[6,3,0]], C69 [[13,5,0],[5,4,0]]. E1 회복/D1 손상, 전체17/27로 같고 triplet2/9→1/9다. natural DEV12는 판정 전행 동일9/12다.
- CF FIT117·natural FIT45·strong TRAIN6096은 양쪽 모두 정답이다. strong 점수는 C68 PART/C69 POINT라는 실제 감독 경로를 사용하며, 약감독 primary PART와 골드 권한을 합치지 않는다.
- 저장 canonical input6297개에서 signature3638개·중복 unordered6132쌍·반대 target 그룹0을 독립 집계했다. 이번 검사는 저장 digest 메타데이터를 재집계했으며 원 RGB/NPZ를 재해시하지 않았다. 모든 데이터의 무모순이나 목표 혼용 무해를 입증한 결과가 아니다.
- C68=V222 C67 logit6297개 bit-exact는 Root 제공 SOURCE replay 기록을 인용했다. B가 이번 범위에서 과거 점수 파일까지 다시 열어 비교한 결과는 아니다.

최종 ROOT/PORTABLE Markdown 바이트 일치, ROOT_RESULT 바인딩·예산·분모·z>=0 decoder·표 수치·scope 설명이 정합하다. 보고서는 warm start/공유 encoder/작고 노출된2-component DEV, GT-derived q8, weak finite-part≠physical native gold, U0≠confidence, 채택·새 private 추론·PnP 없음, 목표 미해결을 명시한다. ROI centroid의 RAW96 raster 거리와 사용자 GT3px 지표를 구분한다.

A 모델/로그/CF CPU replay 검산과 Root4장 실제 뷰는 보고서 및 Root가 제공한 증거로 인용한다. B의 실제 이미지 뷰·model/NPZ/PT/원GT/보호 payload 열람·새 forward는0이다. A의 원본 tensor/로그 재감사, ROI 통계·공개 teacher 역사 원자료, 외부 HTTP/브라우저 render는 이 QA에서 재검산하지 않았다. phase 전체 protected metadata 접근0을 주장하지 않으며 C의 held360 eager-parse 예외 공개를 확인했다.

상대링크16개 중 이미 존재하는 로컬10개는 regular/no-symlink이며 외부2개는 미접속이다. provenance4개는 Root가 예고한 보존 복사/manifest 생성 대기다. 이것을 모두 해결된 링크라고 주장하지 않는다. 최종 공개 전에 Root가 생성할 항목이다. 갤러리9장 PNG는 이 QA에서 열지 않았다.

[전수 산술 결과](NUMERIC_QA.json) · [최종 QA](REVIEW.json) · [산술 코드](check_saved_arithmetic.py) · [최종 text/link 코드](finalize_report_QA.py)
