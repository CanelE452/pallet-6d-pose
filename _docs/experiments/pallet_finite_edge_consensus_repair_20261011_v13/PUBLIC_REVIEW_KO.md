# V13 공개 재현 검토: 실제 완료와 게시 상태

공개된 원행과 코드만 있는 별도 디렉터리에서 **저장 통계810값·81CI와 timing56값 검산이 실제 PASS**였다. 새 모델 실행이나 자세 재추정, 하드웨어 처리시간 재측정의 성공을 뜻하지 않는다. 전체245장 위치·회전 개선 목표는 [결과 snapshot](RESULT_KO.md)의 판정대로 여전히 미달이다. 원격 게시 확인은 아직 대기다.

이 문서는 고정된 RESULT/README/BUILD 이후의 공개 재현 권위다. 그3개 문서는 바꾸지 않았다. 이 늦은 문서는 [공개 복사 protocol](PUBLIC_COPY_PROTOCOL.json)의 제외 항목으로 기록돼 복사 closure에 포함되지 않는다. 실제 복사 당시의3개 snapshot bytes는 포함된다.

## 실제 수행한 공개 검토

|확인|실제 근거|범위와 제한|
|---|---|---|
|원 저장소 공개 자료 검사|[PUBLIC_LOCAL_CHECK](PUBLIC_LOCAL_CHECK/PUBLIC_REVIEW_CHECKS.json):1225 score 원행·5방법·245장·15 status group;338 공개 input alias·1124 declared binding PASS|외부 의존58 binding은 metadata로 남기고 열지 않았다. CSV는 원 scalar 전사다.810/81 산술은 별도 검산이 담당한다.|
|V13 원 gzip 부재 후 복원|[PUBLIC_RESTORE_ABSENT](PUBLIC_RESTORE_ABSENT/RESTORE_CHECKS.json):7개 모두 initially_present=False→실제7개·58,839,290B 생성·기존 일치 파일0·bytes 동일 PASS|[ARCHIVE_MANIFEST](ARCHIVE_MANIFEST.json)의 V13 원 gzip7개는 direct original이고 split0이다. recompression0, 원본은 보존했다.|
|의존 자료의 fresh byte copy|[PUBLIC_COPY_CHECKS](PUBLIC_COPY_CHECKS.json):처음 없던 target에295 unique files·1,231,013,042B 실제 복사 PASS|338은 alias 수이고295는 정렬 병합된 고유 파일 수다. Git clone이 아니다. public protocol과 V13 코드/문서만 복사하며 GT·원 RGB·weights를 가져오지 않았다. 큰 이전 gzip은 아카이브 provenance/parts도 함께 검증했다.|
|fresh 통계 검산|[PUBLIC_FRESH_STATISTICS/REVIEW_VERIFICATION](PUBLIC_FRESH_STATISTICS/REVIEW_VERIFICATION.json):1225 원행·135분포·810값·27 paired group·81CI PASS|72 nonempty/9None, 기존13세션10,000 draw 재사용. 새 draw0·새 GT/score/fit/model0. 원 표본·paired 집합·분모를 공개 stdlib 코드로 재검산했다.|
|fresh timing 원행 검산|[PUBLIC_FRESH_RUNTIME/RUNTIME_CHECKS](PUBLIC_FRESH_RUNTIME/RUNTIME_CHECKS.json):150 저장 호출·공식130회·8block56slot·37142검사/0FAIL PASS|저장 clock·출력/pose·counter의 일관성 검산이다. GPU 또는 detector를 다시 실행하지 않았고 새로운 latency 측정도 아니다.|
|복사 및 검산 영수증의 조인|[PUBLIC_REPRODUCIBILITY_CHECKS](PUBLIC_REPRODUCIBILITY_CHECKS.json):실제 fresh 결과6JSON의 독점 bytes copy와 SHA 조인 PASS|이 조인 영수증 자체의 새 수치 작업0/private asset read0. remote_publication_verified=False다.|

fresh 통계의 최대 scalar 차이는2.7284841053187847e−12, fresh timing 검산의 최대 절대 차이는2.842170943040401e−14였다. 원행으로부터 기존 통계와 저장 timing을 검산할 수 있다는 증거이며, 물리 정답 pose·edge ownership·새 데이터 일반화를 인증하지 않는다.

## 다른 사람이 확인하는 절차

[PUBLIC_REPRODUCE_KO](PUBLIC_REPRODUCE_KO.md)의 독립 출력 디렉터리와 명령을 따른다. 공개 데이터 bytes·ID/status 검사는 [public_review.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/public_review.py),810/81 산술은 [verify_review.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/verify_review.py), 저장150호출/56timing은 [runtime_check.py](../../../scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/runtime_check.py)가 담당한다. stdlib 검산에는 private weights·source GT·원 RGB가 필요하지 않다. 모델 포함 전체 실험 재실행은 별도 원 환경을 필요로 한다.

모든 실행은 own protocol→STARTED→CHECKS로 묶고 기존 출력을 덮어쓰지 않는다. [PUBLIC_COPY_MANIFEST](PUBLIC_COPY_MANIFEST.json)로 복사된 정확한 경로·bytes·SHA를 볼 수 있다. copy/restore는 학습·RGB 생성·PnP·성능 실험이 아니다.

## 실패 보존과 아직 완료라고 말하지 않는 항목

원 CPU14/15와 별도15PASS, 원 geometry713802/74FAIL과 별도713803PASS, 통계 adapter firstFAIL과 별도 정직한 metadata adapter, renderer 끝부분의 binding-shadow FAIL과 별도8그림/24panel 검토를 보존했다. 원 FAIL을 PASS로 덮어쓰지 않았다. [ROOT_VISUAL_REVIEW](ROOT_VISUAL_REVIEW.json)의 실제8이미지 검토는 가독성 확인이며 자세 정확도 보증이 아니다.

공개 fresh 재현 검토는 완료됐지만 최종 보호·정상 research commit/push·remote SHA 일치의 새 게시 영수증은 아직 이 문서에 기록하지 않았다. main 수정/자동 병합/force push는 하지 않는다. 이후 게시 영수증이 생기면 이 별도 문서에 추가하고 고정 snapshot은 유지한다.
