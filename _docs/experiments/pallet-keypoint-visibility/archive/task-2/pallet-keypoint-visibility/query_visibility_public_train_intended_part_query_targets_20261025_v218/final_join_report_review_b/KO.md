# V218 독립 join·최종 보고서 QA

**PASS.** 원 A/C120행의 봉인·UID·원 판독·근거/ROI를 보존한 join과 360행 jitter 지원 조건, condition32 전체를 독립 재계산했습니다. 수치 실행은 575c8f exit0 한 번이고 원 join/관찰/보고서를 변경하지 않았습니다.

- 합의101/120, 화면 안51/70. 최종 nominal D19/E8/부위·기하 미해결24/UNKNOWN19/화면 밖50입니다.
- 두 ROI를 −24 sample씩 옮긴 후 잘림 없이 strict-valid/target지원/query가용 조건을 확인한57변형은 D48/E9입니다. FIT45(D39/E6), DEV12(D9/E3)이며 DEV는 모든 fit/normalizer에서 제외합니다.
- 원96/U11과 새 GT-query360을 구분했고, GT에서 유래한 q8 자체의 권한을 명시했습니다. 다른 GT pose/ref/ROI/mask를 배포 feature에 추가하지 않는다는 사실과 혼동하지 않았습니다.
- 최종4보고서의36개 집계필드, HTML scalar33행·nominal120개 article, MD 주요표2개가 일치합니다. 상대 링크는 ROOT MD15/PORTABLE MD15/HTML364개 모두 존재합니다. 원96 condition 링크는 `deployable_query_condition_b/actual_conditions_v1/ALL96_DEPLOYABLE_CONDITION32.jsonl`로 정확합니다.

약한 assistant SOURCE 부위 판독을 사람/native physical gold나 blind 검증으로 승격하지 않았습니다. 새 학습·모델추론·보호 평가·PnP·admission은0입니다. 이 QA에서는 이미지 본문·배열·모델·원 GT·보호 payload를 읽지 않았습니다. 기존 V213 지표는 보고서의 인용값만 읽었고 원 점수나 benchmark를 재열지 않았습니다.

브라우저 file 접근 제한 때문에 실제 HTML 렌더는 확인하지 않았고 DOM/파일 링크만 검산했습니다. root의 한 번의 광역 filename metadata 열거 deviation도 보고서에 공개됐으며 phase 전체가 zero broad enumeration이었다고 주장하지 않습니다. 두 HTML 표시형식 가정 오류(scalar표의 dict행 제외, 영어 enum의 한국어 label표시)는 최초 checker/실패를 보존하고 최소수리했습니다. 수치 QA나 원 데이터를 다시 실행한 것이 아닙니다.

[실제 join QA](actual_join_v1/REVIEW.json) · [최종 text/link QA](FINAL_REPORT_TEXT_AND_LINK_QA.json) · [root PORTABLE](../PORTABLE_REPORT_KO.md)
