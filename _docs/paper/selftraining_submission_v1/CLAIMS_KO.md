# 문장별 주장–근거

| 원고 문장/주장 | 근거/범위 |
| --- | --- |
| 보정 pseudo 가시점 PCK10 개선 | PSEUDO_LABEL_QUALITY.json: FINAL_V2 fixed-ID44/66→50/66; 217 unlabeled 학습영상 자체의 GT 품질 측정은 아님 |
| 동일 조건 corrected 학생 > raw 학생 | CORE_COMPARABILITY_AUDIT.json + CORE_RESULTS.json: LR5 PCK10/공통D9 AUC; 모든LR/order도 보존 |
| R0보다 추가 가치 | CORE_RESULTS.groups.ALL: PCK10/AUC 개선; 전체 지표 우월성 아님 |
| 분산·실패·악화 | PAIRED_ANALYSIS:92/28/8,51/12; TABLE3 severe/tail, TABLE6 recording |
| 실사 감독 공개 | 기존 teacher INPUT_LOCK/TRAIN_SUPPORT:9이미지38점; 66평가점과 분리 |
| 순수 teacher-free raw 정책 비교 아님 | RAW/REF 동일 보정후 승인집합. 좌표 intervention만 격리; 선택정책효과는 측정하지 않음 |
| 독립 일반화/물리6D 검증 아님 | INDEPENDENT_CONFIRMATION_AUDIT + geometry reference 계약 |
| 신규기법 주장 아님 | PoseFix/STAC/Self6D/BOP 원문 확인; application-specific controlled evidence |
| Hard8/selector/occlusion main으로 섞지 않음 | EXPERIMENT_ROLE_MAP.md |

위 표는 기존 Plastic320 비교의 주장–근거이다. 여기서 R0 대비 추가 가치는 Plastic에 한정하며 Wood까지 확장하지 않는다. 원래 수치의 source path/hash는 generated_tables/NUMBER_PROVENANCE.json 및 MANUSCRIPT_NUMBER_PROVENANCE.json에 있다.

## Wood material 확장으로 추가한 주장

| 주장 | 근거와 제한 |
| --- | --- |
| 두 평가 재료에서 corrected−raw의 pooled 방향이 같다 | Plastic +3.959pp / AUC +0.02429, Wood +0.578pp / AUC +0.00860. 각 재료 내부 RAW/REF matched 비교이며 모든 material 일반성 주장이 아님 |
| Wood 학생 비교는 성립한다 | Wood45장/346코너, 동일 교사9장38점·R0 초기값·고유361장·support·augmentation·source512·320update. 교사/학생과 평가의 ID·SHA·recording 중복0 |
| Wood에서는 R0보다 좋다고 할 수 없다 | Corrected PCK10 165/346, AUC0.66503; R0 167/346, AUC0.67050 |
| Wood pseudo-coordinate 품질은 검수 기준 미확정 | 적격 평가45장에서 직접 클릭 출처를 확인할 가시점0개. Legacy teacher 점수를 trusted quality로 승격하지 않음 |
| 작은 이득과 손상을 함께 보고한다 | Wood10px 진입8/이탈6, Clean 동률, Moderate 순증2; 한 recording PCK10 악화. P90 67.839→68.985px. 두 recording의 반복DEV이고 Severe 없음 |

수치 출처는 material namespace의 WOOD_RESULTS / WOOD_PAIRED_ANALYSIS / MATERIAL_NUMBER_PROVENANCE JSON이다. 사용자가 재현 메타데이터와 평가 오버레이6장의 공개를 승인했고, 원본RGB·원시좌표·카메라행렬·가중치는 비공개로 보존한다. 논문 PDF13페이지와 테스트30개/자동감사는 완료했다. 실제 Git 반영 확인은 material namespace의 PUSH_VERIFICATION.json을 참조한다.
