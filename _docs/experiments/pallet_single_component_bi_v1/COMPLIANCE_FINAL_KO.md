# 원문 이행 감사: 추가 실험 후 상태

원문497줄 전체와 산출물을 대조한 v1 감사를 보존하고 이번 추가 실험 상태만 덧붙였다. **이전에도 모든 세부가 다 끝났다고 말하면 과장이었다.** 원4경로 핵심은 완료했지만 개별 성분과 기록상의 공백을 구분해야 한다.

원문166줄은 N3/SubPix 묶음 제거 설계다. 따라서 N3 단독/SubPix 단독은 이전952행의 완료 항목이 아니었고, 현재 새 요청에서 전체1904행과 독립 검산으로 보충했다. 과거319장의4초기 캐시 존재와 현재192장 B/I 교차 평가는 다르다.

원문403줄은 전체 호출 연결 불가 시 전체 시간 NA,348줄은 기존 검정 정의 미검증 시 p/Holm NA를 허용한다. 두 NA 자체가 지시 위반은 아니지만, 전체 속도와 통계적 유의성을 측정했다는 뜻으로 포장하지 않는다.

245/110는 이전 contract에서 미확정이었다. 이번 감사로 확인된245행은49unique×5방법이고,110은 동일192장의 성공 분자다. 원문이 따로 다른245/110unique 영상 표를 가리켰는지는 여전히 확정되지 않았다. 전체 ID 차집합·출처 SHA를 별도 추가 문서에 남겼다.

일부 초기 읽기·짧은 probe·프로세스 시작/종료의 정밀 시간은 회복할 수 없다. 기존 기록은 주요 stage와 보수적 포함 구간의 상한 확인이지 모든 개별 초수의 완전 계측은 아니다.

새 사용자의 main 게시 지시는 원문의 별도 브랜치 지침을 대체한다. 이전 minimal/guarded 작업의 main 반영은 감사의 read-only 원격 확인에 기록했다. 이번 추가 결과의 최종 main push 완료는 별도 root 게시 영수증으로 확인하며 보고서 생성 시 완료를 미리 선언하지 않는다.

| 감사 항목 | 현재 판정 |
|---|---|
| scope_no_training · §0 | COMPLETE |
| source_contract · §1 | COMPLETE_WITH_RECORDED_LIMITS |
| workspace · §3 | COMPLETE |
| fixed_population · §5.1 | COMPLETE |
| legacy245_110 · §5.1 | PARTIAL_RESOLVED_BY_ADDITIVE_AUDIT |
| same_detector_cache · §5.2-5.3 | COMPLETE |
| four_paths · §4 | COMPLETE |
| minimal_module · §6 | COMPLETE |
| single_cli · §6/14 | COMPLETE |
| parity · §7.1 | COMPLETE |
| boundary9 · §7.2 | COMPLETE |
| pilot · §7.3 | COMPLETE |
| full952 · §8 | COMPLETE |
| failures · §8-9 | COMPLETE |
| metric_tables · §9.1 | COMPLETE |
| four_contrasts · §9.2 | COMPLETE |
| p_holm · §9.2 | ALLOWED_NOT_COMPUTED |
| stage_diagnostics · §10.1 | COMPLETE_WITH_RECORDED_LIMITS |
| coordinate384 · §10.2 | COMPLETE |
| callcounts · §11 | COMPLETE |
| timing · §11 | COMPLETE_WITH_LIMITED_BOUNDARY |
| fulltimeNA · §11 | ALLOWED_NOT_MEASURED |
| resources · §12 | PARTIAL_INSTRUMENTATION_WITH_BOUNDS |
| outputs · §14 | COMPLETE |
| gallery · §14 | COMPLETE |
| recommendation · §13/14 | COMPLETE_WITH_RECORDED_LIMITS |
| old_publication · §15 | COMPLETE_AS_ORIGINALLY_REQUESTED |
| new_main_push · §new user | PRIOR_WORK_MAIN_PUBLISHED_NEW_RESULT_PUSH_SEPARATELY_RECORDED |
| single_components · §4/new user | COMPLETE_NEW_INDIVIDUAL_COMPONENT_EXPERIMENT |
| guarded_followup · §separate followup | SEPARATE_AUTHORIZED_FOLLOWUP_NOT_ORIGINAL_REPLACEMENT |

[전체30항목기계판독감사](COMPLIANCE_AUDIT_FINAL.json), [245/110근거](LEGACY245_110_SOURCE_ID_AUDIT_KO.md). 기존 봉인 파일·사용자 수정·실패·논문을 고치지 않았다. 새 독립 촬영 평가·운영 기본값 교체는 수행하지 않았다.
