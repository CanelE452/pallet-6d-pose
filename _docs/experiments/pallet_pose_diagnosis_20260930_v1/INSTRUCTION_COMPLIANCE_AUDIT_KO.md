> 역사적 기록: 아래는 보완 전 재감사 결과다. 이후 조치와 현재 상태는 [GitHub 게시 전 보완 기록](REVIEW_CORRECTIONS_KO.md)을 확인한다.

2026-09-30 사용자 질문 “지시문 다이행한거야?”에 대한 재감사

핵심 실험과 주요 T/R 계산은 수행했지만, 세부 지시까지 전부 이행한 상태는 아니다. 앞선 완료 보고는 아래 누락과 결과표의 필드 문제를 반영하지 못했다. 이번 재감사는 읽기 전용으로 수행했고, 이 문서를 추가한 것 외에 기존 실험·산출물을 수정하지 않았다.

| 구분 | 확인된 내용 | 근거 |
| --- | --- | --- |
| 미완료: Source256 기하 계약 | image/label/renderer 768개 파일 hash를 확인했다. 과거 생산 코드는 K/padding, renderer pose, indexed projection을 검사했으나, 이번 실행에서는 geometry side table 자체의 binding과 현재 K/pose/인덱스/대칭 계약의 일치를 별도로 확인하지 않았다. 신규 추론 생략과 이 필수 확인의 누락은 구별해야 한다. | `scripts/research/pallet_pose_diagnosis_20260930_v1/provenance.py:92–108`, 과거 `SOURCE_GEOMETRY_BINDING.json` 및 생산 코드 |
| 미완료: 모든 비교에 E7 적용 | E2 oracle 및 best-box 비교에서 paired 계산에 `False`를 전달해 recording bootstrap/LORO를 생략했다. E4는 전체 natural99/clean29의 불확실성 분석은 있지만 recording별 표에서 두 모집단이 섞인다. 통합 모집단×recording 표와 metadata 층별 표는 E1/E6 위주이며 모든 비교를 포괄하지 않는다. | `run.py:210`, `scoring.py:61`, `close.py:70–78`, `provenance.py:74–90`, `E4_SUMMARY.json` |
| 수정 필요: 후보 필드 | E1 held_identity 256행 중 33행에서 CSV의 GEO_name은 재선택 후보를 표시하지만 T/R은 identity의 후보를 고정해서 계산했다. 고정 후보와 자유 재선택 후보를 별도 필드로 표시해야 한다. | `close.py:42–44`, `FRAME_RESULTS.csv` |
| 수정 필요: 2D 필드 | E4 개입 768행의 2D 필드는 개입 후 좌표의 오차가 아니라 입력 baseline의 값을 그대로 담았다. 입력값임을 명시하거나 개입 후 값을 별도 계산해야 한다. T/R은 실제 교체 좌표로 계산했다. | `close.py:60`, `scoring.py`의 visibility 계산, `FRAME_RESULTS.csv` |

이 문제들은 2,964행의 독립 scalar T/R 재계산이 통과했다는 사실과 구별된다. `VALIDATION.json`의 passed는 지시문 전체 준수나 모든 CSV 필드 의미의 검증을 뜻하지 않는다. 기존 `E7_statistics: ANSWERED` 또한 모든 비교에 적용했다는 의미로 읽으면 과도한 판정이다.

지시문에서 허용한 생략과 자료 부족은 별도다. 기존 질문표는 ANSWERED 10, UNRESOLVED 5, SKIPPED 3, BLOCKED 1로 기록되어 있다. 이 개수는 실행자가 세분한 질문의 개수이며 지시문 준수율이 아니다. 좌표 보간·Source 신규 추론·ROI 재추론의 조건부 생략, 자연 가림의 대응 clean 타깃 부재, 독립 반복 클릭 부재, REALFT_A 선택 이력의 모순 등을 남긴 것 자체는 지시문이 허용한다. 다만 Source 신규 추론의 생략이 위의 계약 확인 누락까지 정당화하지는 않는다.

E3는 기존 마스크의 크기·종횡비·색상 규칙을 재사용하고 위치 선정 절차를 cover/avoid 쌍으로 조정했으며 이를 추론 전에 기록했다. qC 예측 위치 기준의 마스크와 참조 코너의 실제 겹침이 일부 다르다는 한계도 보고했다. 문서가 GT 기준 배치를 명시하지 않았으므로 이 부분을 확정적인 지시 위반으로 판정하지 않는다.

남은 보완은 Source 계약 확인, E7의 누락된 집계·불확실성 분석, CSV 필드 의미 수정 및 그에 맞춘 상태·보고서 갱신이다. 이번 재감사에서 이 보완까지 완료했다고 주장하지 않는다. 신규 학습·촬영·GT 변경은 이번 실행 범위 밖이며 수행하지 않았다.
