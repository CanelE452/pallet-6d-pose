# 후속 원인 진단 — 목적과 기준

## 이번 목적

[확인] 기존 material/visible-transfer 종료 뒤 사용자가 새 탐색을 승인했다. 기존 논문 main·GT·checkpoint·평가 membership을 보존하고 별도 namespace에서 진행한다. 목표는 REF−RAW, REF−R0, NEW−기존REF를 분리해 회수 가능한 오류와 미확정 원인을 밝히는 것이다. 좋은 결과를 얻기 위한 무제한 재시도는 아니다.

| 행동 | 필요한 이유 / 생략 시 공백 | 더 단순한 대안 | 결과별 결정 |
| --- | --- | --- | --- |
| 현재 후보 fixed-set oracle | 생성 부족과 선택 손실 분리 | 저장 native 출력·기존 solver 재사용 | gap+GT-free cue 있으면 고정 후보 단순 대조; 없으면 선택기 개발 보류 |
| whole-output/per-point oracle | 교사/학생의 상보적 좌표 정보 계량 | 같은 4개 출력만 사용 | 선택 cue가 없으면 oracle 이득을 배포 성능으로 주장하지 않음 |
| TRAIN217/361 타깃 추종 | 타깃 모방과 실사 전이를 구분 | Plastic 기존 native 잔차 재사용 | augmentation/source/표현을 경쟁 설명으로 두고 단일변경 대조 |
| source/real gradient·노출 | 동일512슬롯이 같은 감독세기인지 검사 | optimizer step0 측정 | 음의 cosine 하나로 PCGrad 채택하지 않음 |
| 문헌/과거 실패 대조 | 이미 반박한 조건 반복 방지 | 실행결과·코드와 계약을 연결 | 다른 정보 요구 대형 방법은 작은 원리로 축소하거나 보류 |
| 참조입력 sanity | solver/참조 순환성 분리 | 기존 정확 합성·legacy좌표 | 순환 consistency를 물리 GT 정확도로 해석하지 않음 |
| 실제 최대3사이클 | 같은정보·한변경의 효과 측정 | 가능한 무학습 표준 대조 우선 | 부정적 결과 보존; 다른 지지 가설은 계속, 근거 소진 시 종료 |

## [확인: 직전 보고값; 재계산은 별도 parity 검사]

| Material / population | R0 PCK10 / AUC | RAW320 PCK10 / AUC | REF320 PCK10 / AUC |
| --- | --- | --- | --- |
| Plastic128 /985점 | 49.14% /0.33796 | 47.51% /0.33472 | 51.47% /0.35902 |
| Wood45 /346점 | 48.27% /0.67050 | 47.11% /0.65643 | 47.69% /0.66503 |

Replay9장38코너는 Wood에도 노출됐다. 재료 zero-shot 검사가 아니다. Plastic검수66은 RAW/REF43/43, 이전640연장44/44로 동률이었다. Wood45 direct-visible 출처가 검증된 점은0이며 좌표 자체가 없다는 뜻은 아니다. 양쪽 모두 반복 DEV, 독립 physical6D아님.

## 운영 잠금

[추정·운영상한] 최대3사이클/12fits/7680updates/GPU6시간/wall10시간. 교사·scorer·smoke도 합산한다. 기본320/fit, capability·재현최대640. GPU fit한번에1개. 원본RGB·좌표·카메라행렬·가중치는 공개하지 않고 이미 승인된 사례만 사용한다. 외부프로세스 종료·재부팅·환경변경 없음.

새 가설은 개별 SPEC를 먼저 저장하고 평가예측을 scoring 전에 고정한다. 과거 조건이 다르면 supplementary로만 사용한다. 평가를 본 후 다음 cycle을 설계한 이력은 DEV개발로 공개한다. CLAIM_IMPACT는 수정 제안이지 기존 main 대체가 아니다.
