# 원영상 수동 코너 검수

새 업무용 스키마입니다. 기존 846개 예측을 사람 주석으로 바꾸지 않습니다. 서버는 Python 3.12 표준 라이브러리만 사용하고 `127.0.0.1`에만 바인딩합니다. 원본 영상·모델·제어 코드를 import하지 않습니다.

프로젝트 루트에서 아래 실제 명령으로 시작합니다.

```powershell
py -3.12 -X utf8 scripts/research/pallet_lifter_case_review_20261003_v1/start_review.py
```

브라우저 주소는 <http://127.0.0.1:8765>입니다. 원영상 120장·재검수 24장과 외부 코너 계약을 읽고 원본 이미지의 SHA-256·해상도 및 계획/계약 해시를 검사합니다. 자료가 없거나 해시가 다르면 시작을 거부합니다.

검수자는 식별자, 기계 도움, 과거 모델 예측 노출, 이전 수동 주석 노출을 직접 입력합니다. 선택 프레임의 **검수 시작 / 재개**를 누르고 대상 객체의 존재·식별자·대응을 확인합니다. 각 ID의 가시성, 외부 가림, 자기 가림, 화면 밖, 정의 미확인을 별도로 기록합니다. 여러 가림 속성을 함께 선택할 수 있습니다.

개념 도식의 ID는 외부 계약의 X/Y/Z 좌표에서 생성되며 원영상 방향이나 가시성을 나타내지 않습니다. 외접 직육면체 코너가 실제 재료의 꼭짓점인지, 대상의 canonical 방향과 해당 ID를 대응할 수 있는지 확인해야 합니다. 확인되지 않으면 **판단 불가 / 직접 클릭 불가**로 남기세요. 실제 코너에 대응함을 확인한 **직접 가시** 점만 원본 내부를 클릭합니다. 숨은 점·가상 꼭짓점의 좌표를 추정하지 않습니다.

휠 또는 확대/축소 버튼으로 확대하고 우클릭 드래그·이동 모드로 이동합니다. 좌표는 CSS 배율·캔버스 표시 크기·기기 픽셀비와 분리된 원본 픽셀 좌표로 저장합니다. 저장한 직접 클릭만 파란색으로 표시하고 모델의 점·박스·오차는 불러오지 않습니다. **최근 변경 취소**, 수정 사유, 건너뛰기 사유를 지원합니다. 제출 기록을 수정할 때 이전 전체 기록은 `annotations_in_progress.history.jsonl`에 남습니다.

**진행 중 저장**은 미검수 상태를 보존하며 정확도 참조로 내보내지 않습니다. 8개 ID를 모두 검수한 뒤 **사람 검수 완료 저장**을 누릅니다. 객체가 없거나 가시점이 0개여도 표본을 유지하고 상태를 기록합니다. 건너뛰기는 사유가 필요하며 참조 클릭을 포함하지 않습니다.

실제 저장 위치는 다음과 같습니다.

```text
data/pallet/results/pallet_lifter_case_review_20261003_v1/review/annotations_in_progress.json
```

브라우저를 닫았다가 동일 시작 명령으로 재개하면 같은 검수 차수·같은 검수자의 저장 기록을 불러올 수 있습니다. 재검수 차수는 첫 검수 좌표를 불러오지 않습니다. 다른 사람의 재검수 저장도 새 검수자에게 미리 채우지 않습니다. 같은 사람의 재검수는 `same_person_repeat`, 다른 사람은 `different_person_repeat`로 기록하며 기계 도움·첫 주석 노출이 있으면 독립 재검수로 표시하지 않습니다.

화면의 **제출된 주석 JSON 내보내기**를 눌러 `LIFTER_REFERENCE_REVIEWED.json`을 받습니다. 제출된 범위만 포함하며 진행 중 기록은 제외합니다. 실제 제출 기록이 없으면 `human_in_progress`, `WAITING_HUMAN`, 빈 목록으로 반환되어 정확도 참조 검증에서 거부됩니다. 내보내기에 일부 표본만 있으면 미완료 수와 `complete_primary=false`가 남습니다. 파일을 받은 뒤 아래 한 가지 명령으로 검증·병합과 평가를 재개합니다.

```powershell
py -3.12 -X utf8 scripts/research/pallet_lifter_case_review_20261003_v1/resume.py "받은_JSON의_경로"
```

로컬 서버 저장만 이용해 재개할 때는 같은 명령에서 JSON 인자를 생략합니다. 고정 가중치가 누락된 경우 추론은 해당 입력이 확보될 때까지 `BLOCKED_CONTRACT`로 남고, 수동 검수와 자료 감사는 진행 가능합니다. 사람이 제출한 참조가 없으면 정확도는 `WAITING_HUMAN`이며 완료 수치를 만들지 않습니다.

아래 독립 명령은 서버 구현의 실제 검증·import·export 기능입니다. 경로를 줄이기 위한 PowerShell 변수입니다.

```powershell
$taskCode = 'scripts/research/pallet_lifter_case_review_20261003_v1'
$taskOut = 'data/pallet/results/pallet_lifter_case_review_20261003_v1'
$bindingArgs = @('--manifest', "$taskOut/review/MANIFEST.json", '--plan', "$taskOut/LIFTER_EVALUATION_PLAN.json", '--contract', "$taskOut/review/CORNER_CONTRACT.json", '--store', "$taskOut/review/annotations_in_progress.json")
py -3.12 -X utf8 "$taskCode/review/serve.py" validate @bindingArgs --input '받은_JSON의_경로'
py -3.12 -X utf8 "$taskCode/review/serve.py" import @bindingArgs --input '받은_JSON의_경로' --merge
py -3.12 -X utf8 "$taskCode/review/serve.py" export @bindingArgs --output "$taskOut/review/LIFTER_REFERENCE_REVIEWED_새버전.json"
```

Import는 해시·ID·좌표·상태·사람 provenance를 엄격히 검사합니다. 같은 frame/pass의 충돌 기록을 덮어쓰지 않습니다. 동일 기록 재import는 허용합니다. Export는 기존 파일을 덮어쓰지 않고 실제 제출 기록이 0개일 때 파일 생성도 거부합니다. 서버 저장→재시작→검증→내보내기→다른 저장소 import 왕복은 임시 테스트 폴더에서 검증했습니다.

스키마 `lifter_reference_review_v1`의 `bindings`는 manifest/plan/contract 해시와 `corner_definition_version`을 포함합니다. 레코드는 `review_pass`, `status`, `source_kind`, 이미지 해시·원본 치수, 객체 판단, 여덟 `corners`, 직접 입력한 `reviewer`, 서버가 실제 측정한 `review_time`을 포함합니다. `corners[].visibility`와 가림 축은 별개이며 직접 가시일 때만 `x,y`를 허용합니다. 서버가 `started_at`, `finished_at`, monotonic 경과 시간을 기록하고 검수자 이름·확인 이력·클릭을 만들지 않습니다.

자동 검산 명령:

```powershell
Push-Location scripts/research/pallet_lifter_case_review_20261003_v1/review
py -3.12 -m unittest -v test_review
node test_coords.cjs
node --check app.js
Pop-Location
```

합성 시험은 `TemporaryDirectory`에서만 실행하며 실제 주석 저장소에 저장하지 않습니다. 전체 시험은 저장·HTTP·재개·필터링·재검수 독립 시작·편집 이력과 미검수/누락/중복/범위/해시 오류 차단을 검사합니다. JS 시험은 96개 CSS/DPR/확대/이동 왕복과 알려진 좌표값을 검사합니다.
