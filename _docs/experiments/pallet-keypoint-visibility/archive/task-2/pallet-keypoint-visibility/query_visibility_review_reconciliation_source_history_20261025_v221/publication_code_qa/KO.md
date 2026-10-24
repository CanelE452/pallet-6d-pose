# V221 게시 코드 독립 검토

판정: 정적 검토 PASS, 중대 차단사항 0. 게시 스크립트와 Git 명령은 실행하지 않았습니다. 원장·SOURCE·모델·평가 데이터는 읽지 않았습니다.

검토 코드: `V221_ROOT_REVIEW_PREPARATION/publish_review.py`

SHA256: `7527abc6e631033d5c945c5466b80bfb8e8a3ba248fc10927c3e6bf38c7e44bf`

- 원 HEAD와 원격 main이 고정 BASE `59cf1e512fc8084c66f1eedc5195b4434ab9e28d`인지 확인하고 기존 tracked/staged 변경을 거부합니다.
- 독립 QA의 completed/PASS와 고정 보고서 SHA를 확인한 뒤 정확한 V221 폴더만 동결·복사합니다. source/복사본/staged blob SHA와 staged 경로 집합을 비교하고 push 이후 원본·복사본을 다시 확인합니다.
- 경로의 금지 문자열을 checked()에서 FS 접근 전에 거부하고 부모부터 symlink를 검사합니다. 기존 누락이던 ROOT_REVIEW_BINDINGS leaf도 checked()로 보호됐습니다. MD 링크는 문자열 정규화 후 REPO 내부 경계를 확인하고 checked()를 거쳐 존재 여부를 검사합니다.
- README 앞에 새 안내를 붙이며 기존 내용을 suffix로 보존합니다. force 옵션 없이 HEAD:main을 push하고 ls-remote와 commit 일치를 확인합니다. main 게시 receipt는 불변이며 handoff 이후 SHA는 별도 turn receipt에 기록해 순환 바인딩을 피합니다.

한계: 실제 파일 집합·QA·링크·Git 상태를 확인한 실행 증거가 아닙니다. HTTP HEAD 실패는 코드에서 receipt에 기록하므로 main push 성공과 HTTP 확인 성공은 별도 항목입니다. 실제 성공 여부는 root 실행 receipt로 판단해야 합니다.
