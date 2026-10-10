# V14 공개 자료와 fresh 검산의 실제 완료

별도의 공개 자료 복사본에서 **1470개 저장 score 원행의 972개 통계값·81CI와 150개 저장 호출의 timing56값을 실제로 검산해 PASS**했다. 새 영상 추론이나 하드웨어 처리시간 재측정은 하지 않았다. 위치·회전 개선 목표는 고정된 [RESULT snapshot](RESULT_KO.md)의 판정대로 미달이다.

이 문서는 RESULT/README/BUILD 이후의 공개 검토 권위다. 고정된 세 문서의 bytes는 바꾸지 않았다. 이 늦은 문서는 [PUBLIC_COPY_PROTOCOL](PUBLIC_COPY_PROTOCOL.json)에 명시된 제외 항목이며, 실제 복사 당시의 세 snapshot은 복사 closure에 포함된다. 따라서 이 문서 자체가 앞선 복사본에 있었다고 주장하지 않는다.

## 실제 확인한 자료

|항목|완료 근거|범위와 제한|
|---|---|---|
|보고서 숫자·원행 조인|[REPORT_SCALAR_CHECKS](REPORT_SCALAR_CHECKS.json): 972 moment 값·81CI·56 timing 값, 1470 score 원행과 24 case panel 조인 PASS|새 moment/CI/timing 산술은 하지 않은 전사 검사다. 원 scalar 단위는 위치 cm·회전 degree·ADDsym m이다. [공개 CSV](REPORT_PUBLIC_FRAME_SCALARS.csv)도 저장 scalar 전사다.|
|공개 의존 closure|같은 영수증: 1586 declared binding·290 unique public files·569 external metadata binding|외부 metadata는 열지 않았다. 경로·SHA·bytes가 일치하는 명시적 external 선언으로 분류했고, 파일 존재 여부로 private/public을 추측하지 않았다.|
|처음 없던 공개 byte bundle|[PUBLIC_COPY_CHECKS](PUBLIC_COPY_CHECKS.json): 처음 없던 target에 **344 files·1,254,416,688B** 실제 복사 PASS|397은 public input alias 수다. 전체 코드·문서를 병합한 344는 고유 파일 수다. Git clone이 아니며 GT·원 RGB·weights를 가져오지 않았다. 이전 큰 gzip의 ordered archive parts와 provenance도 보존했다.|
|V14 원 gzip의 부재 후 복원|[PUBLIC_RESTORE_ABSENT](PUBLIC_RESTORE_ABSENT/RESTORE_CHECKS.json): 원 gzip 8개가 처음 없었다가 실제 8개·54,549,247B 복원 PASS|[ARCHIVE_MANIFEST](ARCHIVE_MANIFEST.json)의 direct original 8개, split0·recompression0이다. 원 bytes를 바꾸지 않았다.|
|fresh 저장 통계 검산|[PUBLIC_FRESH_STATISTICS/VERIFICATION](PUBLIC_FRESH_STATISTICS/VERIFICATION.json): 1470 원행·162분포·972값·27 paired group·81CI PASS|72 nonempty·9None. 기존 13세션의 10,000 draw를 재사용했다. 새 draw/GT/score/fit/model0이다. 최대 numeric 차이 2.7284841053187847e−12.|
|fresh 저장 timing 검산|[PUBLIC_FRESH_RUNTIME/RUNTIME_CHECKS](PUBLIC_FRESH_RUNTIME/RUNTIME_CHECKS.json): 150 저장 호출·공식130회·8block·56slot PASS|저장 clock·출력/pose·counter의 일관성 검사다. 최대 절대 차이 1.8189894035458565e−12. GPU나 모델을 다시 실행하지 않았고 새 latency를 측정하지 않았다.|
|실제 fresh 결과의 복사 기록|[PUBLIC_REPRODUCIBILITY_CHECKS](PUBLIC_REPRODUCIBILITY_CHECKS.json): fresh protocol/STARTED/CHECKS 6JSON의 실제 복사와 SHA 기록 PASS|새 물리 GT·edge ownership·하드웨어 시간·새 데이터 일반화의 인증은 아니다.|

[PUBLIC_COPY_MANIFEST](PUBLIC_COPY_MANIFEST.json)에서 복사된 경로·bytes·SHA와 inherited archive provenance를 확인할 수 있다. 각 checker는 먼저 own protocol을 고정하고 STARTED를 남긴 뒤 CHECKS를 썼다. 저장 원행의 독립 산술 검산과 모델 포함 원 실험은 서로 다른 작업이다.

## 실패와 수정의 보존

원 보고서 inspector와 두 metadata review의 freeze 실패 세 건을 보존했다: [external 분류 누락](REPORT_PUBLIC_METADATA_FIRST_FAILURE.json), [legacy origin 분류 누락](REPORT_ORIGIN_REVIEW_FIRST_FAILURE.json), [namespace CODE 누락](REPORT_ORIGIN_NAMESPACE_FIRST_FAILURE.json). 모두 own protocol/STARTED/수치 산술 전에 실패했다. 최종 [report_scalar_origin_ready.py](../../../scripts/research/pallet_stable_incidence_guard_20261011_v14/report_scalar_origin_ready.py)는 원 표·scalar 산술을 바꾸지 않고 명시적 external provenance 분류와 module CODE 연결만 보완했다. 원 실패를 PASS로 덮어쓰지 않았다.

fresh 검산 준비에서도 [통계 출력 디렉터리 부재](FRESH_STATISTICS_SETUP_FIRST_FAILURE.json)와 [timing 출력 디렉터리 부재](FRESH_RUNTIME_SETUP_FIRST_FAILURE.json)를 남겼다. 두 실패 모두 own protocol/STARTED/산술0이다. 의도한 빈 출력 디렉터리를 만든 뒤 같은 checker와 같은 input bytes로 실제 검산을 완료했다. 성능 설정이나 추정 결과를 바꾼 재실험이 아니다.

## 다른 사람이 확인하는 최소 절차

공개 저장소의 기존 출력은 보존하고 별도 빈 출력 디렉터리를 사용한다. 아래 코드는 표준 라이브러리만 사용한다. `freeze`와 `run`은 순서대로 한 번씩 실행하고 실패 영수증이나 prefix를 지우지 않는다.

```bash
mkdir -p /tmp/pallet-stable-incidence-guard-private-20261011-v14/reviewer-statistics
python -I -S scripts/research/pallet_stable_incidence_guard_20261011_v14/verify.py freeze --output /tmp/pallet-stable-incidence-guard-private-20261011-v14/reviewer-statistics
python -I -S scripts/research/pallet_stable_incidence_guard_20261011_v14/verify.py run --output /tmp/pallet-stable-incidence-guard-private-20261011-v14/reviewer-statistics
mkdir -p /tmp/pallet-stable-incidence-guard-private-20261011-v14/reviewer-runtime
python -I -S scripts/research/pallet_stable_incidence_guard_20261011_v14/runtime_check.py freeze --output /tmp/pallet-stable-incidence-guard-private-20261011-v14/reviewer-runtime
python -I -S scripts/research/pallet_stable_incidence_guard_20261011_v14/runtime_check.py run --output /tmp/pallet-stable-incidence-guard-private-20261011-v14/reviewer-runtime
```

실제 fresh 통계/시간 검산은 위 절차와 같은 고정 checker의 별도 복사본에서 완료했으며, 명령 예시 자체를 추가로 실행한 것은 아니다. private weights·source GT·원 RGB는 이 저장 결과 검산에 필요하지 않다. 모델 포함 전체 경로 재실행은 원 실험 환경과 자산이 필요하다.

[ROOT_VISUAL_REVIEW](ROOT_VISUAL_REVIEW.json)의 실제 8이미지·24case panel 검토는 가독성 증거다. 그림이나 낮은 pixel residual이 물리 대응의 소유 관계와 pose 정확성을 보증하지 않는다. 직접 가시점·숨은 점과 전체 운용 결과의 수치 해석은 고정 RESULT를 따른다.

공개 byte 복사와 fresh 검산은 완료됐다. 이후의 정상 research commit/push·remote SHA 일치 영수증은 아직 이 문서에 기록하지 않았다. main 변경·자동 병합·force push는 하지 않는다. 게시 확인이 생기면 이 별도 문서에 추가하고 고정 snapshot은 유지한다.
