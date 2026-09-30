# GitHub 게시 전 보완 및 검증

최초 완료 보고 이후 사용자 질문에 대한 재감사에서 실제 누락을 확인했다. [당시 감사 기록](INSTRUCTION_COMPLIANCE_AUDIT_KO.md)은 역사적 기록으로 남기고, 아래를 이번 GitHub 검토판에 반영했다. 기존 자연99/clean29의 주 T/R 결과는 바꾸지 않았다.

| 지적 | 이번 조치 | 검증 근거 |
|---|---|---|
| Source256은 파일 hash만 재확인 | side table hash, K와 padding, renderer R/t, 치수·중심, 8코너 전단사/재투영, 현재 평가 코드의 대칭 계약을 256장 검사 | [Source 계약](SOURCE_CONTRACT_REVIEW.json) |
| E2 CI/LORO 누락 및 E7 일부만 적용 | 69개 pose 비교 전체에 모집단별 paired recording 분석. 모든 비교의 recording·단일 metadata 층별 표 작성 | [통계 설명](STATISTICS_KO.md) |
| E4 recording 표에 natural/clean 혼합 | 두 모집단×recording으로 분리 | [E4 요약](E4_SUMMARY.json), [전체 recording CSV](E7_BY_RECORDING_ALL.csv) |
| held_identity 33행의 후보명 오표기 | 실제 사용 후보와 자유 GEO 후보를 분리 | [필드 수정 검증](CSV_FIELD_REPAIR.json) |
| E4 개입 768행에 baseline 2D 표시 | 실제 교체 좌표로 2D 오차 재계산. 입력과 출력 의미 분리 | [수정 CSV](FRAME_RESULTS.csv) |

Source의 최대 renderer 재투영 잔차는 약 0.000451 px, 준비된 label과는 약 0.000825 px다. 현재 Source256의 허용 대칭은 C1 62장/C2 194장이며 자연 평가의 C2와 동일하지 않다. 기존 Source의 rotation 집계값을 자연 평가 값으로 복사하지 않는다. renderer의 정확한 3D 대응점으로 한 PnP 검사는 기하 연결 점검이며 모델 정확도 결과가 아니다. FULL125의 해당 operational xy 캐시가 없어 Source 모델 T/R 신규 비교는 조건부 생략 상태로 유지한다.

검증은 실제 T/R 2,964행의 최초 독립 재계산에 더해, 이번 E4 1,024행의 pose 재검산·T/R 변경 없음·33행 후보명·768행 2D 대상 수정까지 포함한다. 새 모델 추론·학습·촬영·GT 변경은 0회다. 보완 분석의 CPU 비용은 REVIEW_COST 파일에 별도 기록한다.

자료 부족으로 남은 자연 가림의 대응 clean 타깃 정확도, RGB 효과의 일반화, 박스 불일치의 물체 정체, E5-B의 어려운 오류 전이, REALFT_A 선택 이력, 실측 재클릭 noise floor를 완료로 바꾸지 않았다. [현재 질문 상태](QUESTION_STATUS.json)를 확인한다.

최초 완료 시점의 manifest는 [history](history/RUN_MANIFEST_INITIAL_COMPLETION.json)에 보존하고, 변경된 산출물·코드의 이전 바이트는 로컬 `data/pallet/results/pallet_pose_diagnosis_20260930_v1/before_github_review/`에 보존했다. 최신 파일 hash는 PUBLICATION_MANIFEST.json의 경로에 적용한다. 과거 manifest의 hash를 최신 수정 파일에 적용하면 일치하지 않는 것이 정상이며, 변경 연결은 PUBLICATION_VALIDATION.json에 기록한다.
