# 상태별 관측 전달·대응 진단

원래 typed 관측을 보존한 두 경로는 기존 e를 복원했다. e보다 추가로 좋아진 결과는 아니다. 후단 재관측 경로는 일부 회복과 함께 기존 성공을 손상했다. 운영 기본값·가중치·논문은 바꾸지 않았다.

[전체 결과·평균·표본분산·SD·최대·모든 실패](REPORT_KO.md) · [실제8사례 단계별 갤러리](gallery/index.html) · [재현 안내](REPRODUCE.md)

| 경로 | PRIMARY192 성공 | TEST49 성공 | PRIMARY 위치 평균±SD cm | PRIMARY 회전 평균±SD ° |
|---|---:|---:|---:|---:|
| N0 | 108/192 | 40/49 | 11.306±26.877 | 10.631±25.131 |
| POSEFIX_RAW | 80/192 | 34/49 | 13.836±27.578 | 11.640±26.525 |
| E0 | 114/192 | 49/49 | 14.112±30.077 | 13.001±28.388 |
| REOBS_MULTI | 109/192 | 42/49 | 16.207±33.166 | 17.296±32.725 |
| REOBS_SINGLE | 110/192 | 42/49 | 14.698±29.177 | 15.586±30.732 |
| TYPED_MULTI | 114/192 | 49/49 | 14.112±30.077 | 13.001±28.388 |
| TYPED_SINGLE | 114/192 | 49/49 | 14.112±30.077 | 13.001±28.388 |

성공은 같은 pose의 위치≤5cm AND physical/C2 회전≤5°다. PRIMARY192=TEST49+REMAINDER143이며 CAL46은 별도다. 전체19경로×4모집단을 보고하고 outlier·실패·severe를 새로 넣거나 빼지 않았다. N0와 PoseFix가 e보다 평균이 낮은 손익도 그대로 보존했다.

PRIMARY의 typed SINGLE−e 위치는192장 모두 같고 회전의 작은 부호 차이는 최대3.04e-12°의 roundoff다. typed SINGLE↔MULTI는 원 audit상 `NOT_ISOLATED`여서 순수 시작 개수의 인과효과로 부르지 않는다. 전체238 예측·후보를 봉인한 뒤 별도 프로세스에서만 참조를 읽었다.

원 e가 실제 소비한 후보 범위는 전238장에서 확인했다. PRIMARY feasible joint oracle146장에서 최종114장으로 줄며, 최종실패78장은 관측·제안 제한46/선택손실32로 나뉜다. 원 caller 밖의 역사적 전체 생성 기록 완전성과 구분한다. 이는 고정된 기하 재구성 참조 기준의 상한이며 물리적 불가능성 증명이 아니다. [후보 범위](ORACLE_LIMITS.md), [전량 oracle](oracle/ORACLE_SUMMARY.json).

PoseFix RAW는 기존 synthetic-only seed1 last6000 uncapped PASS1이다. 두 기존 SubPix cap은 공통 CAL46/TEST49에서만 각각 보고하며 전체192 단일 규칙은 미확립이다. [소스·과거 실패·학습지원 감사](POSEFIX_TRANSFER_AUDIT.md), [선행과 기여 범위](CONTRIBUTION_EVIDENCE.md). CAL의 투영 잔차는 독립 2D 클릭/의미 owner 정답이 아니며 학습 오류지원은 `COVERAGE_NOT_VERIFIED`다.

새952 pose 슬롯을 실행했고 실제 worker CPU는 7676.278s, 계측 CPU solver/analysis 단계 elapsed 합은 3176.841s다. 범위는 관측 캐시 이후 재계산이며 E0 상위 비용은 추가다. [실제 비용](evidence/COST.json)을 full RGB FPS로 바꾸지 않는다.

독립 재집계4,522행·76집계를 확인했고 [최종 전체 소스 테스트](evidence/FINAL_SOURCE_TESTS.json)는 72개 통과했다. 선행71개 영수증도 보존했다. 로컬 동결 fixture 경로는 `PALLET_GUARDED_DIAG_FIXTURES`로 지정할 수 있다. [보호 계약](evidence/FINAL_CONTRACT_CHECK.json)과 [메타데이터 v1/v2 수치 동등성](evidence/SCORE_V1_V2_METADATA_BRIDGE.json)을 별도로 남겼다. 자료는 반복 개발에 노출됐으며 새 독립 test가 아니다.

보고서 생성 완료와 실제 원격 게시 완료는 별개다. 실제 commit SHA·push 확인은 최종 게시 영수증을 우선한다.
