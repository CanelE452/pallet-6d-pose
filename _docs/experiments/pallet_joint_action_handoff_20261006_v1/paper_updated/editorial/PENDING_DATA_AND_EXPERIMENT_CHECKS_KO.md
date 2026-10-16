# 원고 셀 연결과 남은 x (2026-10-06)

[확인] 아래 표는 closeout 단계의 역사 기록이야. 이후 실제 대상 대응12건과 원래 G 저장본96좌표(직접66·PnP30)의 탐색 패널을 원고에 연결했고, 후속72점 초안은 별도 버전 민감도로 보존했어. 따라서 아래 과거의 대상대응0·67좌표 표기는 현재 완료 상태로 읽지 않아. 공식120/24장 계약, 확인된 상대 정지, 독립 물리 자세 참조의 x는 현재도 유지해. 이번 GEO/PERM 6 fit·동결 I/J·전체 경계 runtime 및 실제 PDF 빌드는 [C 상태](../../C_status.json), [A 상태](../../A_status.json), [A 통합](../../A_PAPER_INTEGRATION.json) 영수증이 기준이야.

| 원고 위치 | 확보 결과 | 실제 원고 반영 | 상태·제한 | 원시→지표 근거 |
|---|---|---|---|---|
| tab:composition | 319장153/92/74,119장3/85/31 | 반영 | 사람입력등급(기준 미확인); 보조분석 | static/LABEL_PROVENANCE_AUDIT → STATIC/SQUARE_REAGGREGATION |
| tab:occlusion_results | YOLO 최신등급 Base/N3 | 반영 | 전체319 headline·실패 분모 유지 | static/STATIC_REAGGREGATION |
| sup:occlusion_results | 세기반×3등급 Base→N3 9행 | 반영 | seed통계평균; 실패분모·악화보존; 등급 기준미확인 | 같은JSON/per_seed |
| aux:yolo_all_arms_current | YOLO Base/P/N2/N3 최신등급 전체행 | 별도TeX/CSV 완성 자료 | AVAILABLE_AS_SUPPORTING_EVIDENCE; 추가삽입은 선택 | 같은JSON |
| sup:visibility | 2499점 전체 가시성 Base/N3 | 반영 | 원래참조 유지; 독립블라인드 아님 | STATIC_VISIBILITY_MERGE_AUDIT → VISIBILITY_RESULTS |
| tab:square_results / sup:square_results | 119장602/600,세기반 | 반영 | 단일session; N3대Base/N2 구분 | visibility_square/SQUARE119_RESULTS |
| 정사각형119 T/R | x | x 유지 | 독립6D 참조없음 | SQUARE119_RESULTS.reference_6D |
| 정사각형 세션간 CI | NA | NA 설명반영 | 단일session으로 정의되지않음 | 같은JSON.intervals |
| 세기반·재질×등급·각seed·정사각형 등급 세부행 | 계산·서식 완료 | 상세CSV/54행 검토MD·JSON | AVAILABLE_AS_SUPPORTING_EVIDENCE; 필수 미완료 아님, 추가삽입은 선택 | final_review/SUBGROUP_REVIEW + STATIC_ALL_BACKBONE_GROUPS |
| 과거외부가림128 | 기존숫자 보존 | 별도조각만 | 최신319등급과혼합안함 | HISTORICAL_EXTERNAL128_SOURCE.csv |
| 과거정사각형150 | 별도103/44/3; 이전R0/N0/N2 2D | 별도CSV/계약만 | 현재N3·DOPE·ResNet/6D x; 참조검수미완료 | HISTORICAL_SQUARE150_SOURCE.json |
| tab:case frame/fresh/no-pose/인접변화 |8910/8772/138 및8737쌍 | 기존숫자보존 | 모델연속출력; 정확도아님 | 기존 LIFTER_CONTINUITY_SUMMARY + lifter/AUDIT |
| 리프터입력 |12장96상태=67좌표+5가시무좌표+24자체 | 본문설명반영 | 공식참조0·객체대응0 | lifter/AUDIT.classification/formal_reference |
| 리프터 가시코너오차 |x|x 유지| 공식참조·대상대응미확인,5점좌표없음 | lifter/AUDIT.formal_reference |
| 리프터정지변동 |x|x 유지| 사람확인정지구간없음 | lifter/AUDIT |
| 리프터물리 T/R/방향오차 |x|x 유지| 독립물리참조없음 | lifter/AUDIT.independent_physical_pose_reference |
| tab:backbones/절제/비교군/128학생/비용 | 기존동일계약결과검산 | 기존표보존 | 319와128혼합없음; 환경패널 유지 | static회귀검산·기존고정CSV |

`x`는 근거·계약 부족, `NA`는 정의되지 않는 통계, `0`은 실제 측정된 영입니다. 이번원고마감에는 새학습0·새클릭요구0회입니다. 리프터정확도를 별도로 완성할 때에만 현재67점재사용과좌표없는5점·이력·대상대응이 필요합니다. 현재추론8910과사람가시성분류96개는 완료상태로 보존합니다.
