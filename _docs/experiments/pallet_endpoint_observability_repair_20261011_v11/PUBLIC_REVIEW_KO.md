# V11 공개 자료 복원과 독립 검토

**공개 파일만 사용하는 새 폴더에서 실제 검산을 완료했다. 정확도 목표는 미해결이다.** [결과 보고서](RESULT_KO.md)의 숫자와 정책은 그대로 보존했다. 보고서의 게시 대기 상태는 당시 기록이며, 이 문서가 후속 실제 공개 검토의 권위다.

[공개 복사 receipt](PUBLIC_COPY_CHECKS.json)는 존재하지 않았던 새 목적지에 공개 코드·증거 386개를 실제 복사했다. 기존 V9 검토 폴더의 해시가 같은 파일 199개를 재사용했다. 새 Git clone이나 새 모델 실행으로 표현하지 않는다. V11 새 gzip은 모두 Git 파일 크기 한도 아래라 분할 아카이브 복원 호출은0이다. 이미 공개된 V7의 큰 원 gzip은 기존 fixture의 복원 바이트와 원 ARCHIVE_MANIFEST 권위를 대조해 복사했다.

[새 폴더에서의 실제 재검산과 원 영수증](PUBLIC_REPRODUCIBILITY_CHECKS.json)은 별도 freeze 후 통계1470행·972 moment scalar·135 CI, runtime150행 중130 측정행·33,905검사를 모두 PASS로 확인했다. 원행 재추론·GT 읽기·fit·새 시간 측정·학습·RGB 생성0이다. 이는 알려진 DEV의 저장 수치 검산이며 독립 물리 정답이나 정확도 개선을 증명하지 않는다.

Runtime inspector의 최초 freeze 명령은 상대 입력 경로를 사용해 public path 대조에서 거부됐다. protocol 생성과 산술 시작 전 오류를 별도로 보존했다. 절대 경로로 실행한 실제 freeze와 검산은 통과했다. inspector·원행·솔버를 고쳐 통과시킨 것이 아니다. 원 수치 실험의 두 runtime 실패와 선택 검사 KeyError도 삭제하지 않았다.

[원본·사용자 변경 보호](FINAL_PROTECTION_CHECKS.json)와 [최종 게시 파일 목록](PUBLICATION_MANIFEST.json)을 함께 확인한다. 정상 게시 대상은 `research/observation-refiner-robust-pnp-20261009`이며 main을 변경하거나 force push하지 않는다. 실제 게시 commit과 원격 일치는 Git 기록 및 게시 직후 별도 receipt로 확인한다. 이 문서 자체가 아직 시작하지 않은 push를 완료했다고 주장하지 않는다.
