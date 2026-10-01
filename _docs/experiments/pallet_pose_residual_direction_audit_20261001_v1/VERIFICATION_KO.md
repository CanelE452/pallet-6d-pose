# 잔차 방향 입력 감사 독립 검산

**검산 PASS. 입력 진단이며 학습·T/R 개선 성과가 아니다.** TRAIN2598행과 실패1행을 유지했다. 새 fit·weight 적용·후보 선택·PnP·VAL 품질·실사 참조 조회는 모두0이다.

생산자의 원래 실행은 두 번 중단됐다. 첫 시도는 threadpool 초기화에서 모델 루프 전에 멈췄고, 두 번째는4모델 계산과 NPZ 저장 뒤 자신의 출력을 읽어 hash를 만드는 guard에서 멈췄다. 별도 공개 finalizer가 같은 봉인 수학을 재계산하고 기존 NPZ25개 key의 dtype·shape·bytes 및 파일 SHA 불변을 검증한 세 번째 진단 시도에서 완료했다. 원래 guard가 성공했다고 기록하지 않으며, 독립 검산도 이 복구 체인과25개 해시를 확인했다.

원래 고정 R_cf/t/치수/K를 성분별 투영식으로 재구성하여4개 expert의 유효20,776후보,186,984점을 확인했다. 저장된 signed18과 기존94의 잔차9·corner8 캐시를 사전 허용오차에서 대조했다. 정규화18은 R0 TRAIN 유효5,194후보만 사용하며 기존253과타깃6해시는 모델마다 정확히 일치한다.

입력/확장 입력의 ±0만 통일한 byte 그룹, 모든 반복 그룹의 T/R 범위·부호 충돌·분할·남은 충돌·anchor 구분을 독립 재구성했다. 열공간 투영은 label 없이 pivoted QR와 작은 R의 GESVD를 사용해 생산자의 직접 SVD와 교차했다.

| 모델 | 기존 rank | 확장 rank | 독립 잔차 비율 | 1e−4 이하 |
|---|---:|---:|---:|---|
| R0_ONLY | 200 | 218 | 0.189748594 | False |
| UNION_s1 | 203 | 221 | 0.422550821 | False |
| UNION_s2 | 203 | 221 | 0.433444381 | False |
| UNION_s3 | 203 | 221 | 0.436207306 | False |

추가 입력의 비중복성은 정답 정보·성능 향상·도메인 전이의 증거가 아니다. 같은 입력의 타깃 충돌이 없더라도 표현 충분성을 입증하지 않는다. 이번 감사는 후속 학습을 승인하지 않으며 원래 source45/실사5 조건을 변경하지 않는다.

기존 source 컨테이너에는 VAL 입력도 있지만 투영·진단은 사전 적격TRAIN2598행만 사용했다. target은 동결 TRAIN 전용 캐시에서 해시와 충돌 설명에만 사용했으며 새 물리 오류를 계산하지 않았다.

[독립 검산 JSON](VERIFICATION.json) · [방향 특징 감사](FEATURE_AUDIT.json) · [표현 감사](REPRESENTATION_AUDIT_KO.md) · [고정 프로토콜](REPRESENTATION_PROTOCOL.json) · [운영 중단·복구](RUNTIME_INITIALIZATION_KO.md) · [복구 계획](RECOVERY_PLAN.json)
