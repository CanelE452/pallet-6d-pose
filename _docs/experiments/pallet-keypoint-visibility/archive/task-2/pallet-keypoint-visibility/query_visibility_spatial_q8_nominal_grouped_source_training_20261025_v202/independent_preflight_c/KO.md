# V202 독립 준비 검토

**PASS.** 최종 A adapter와 B head/trainer의 정확한 SHA를 결합했다. 독립 가상 검사는 scalar14개, A pure API35개, B CPU head·훈련 순서38개가 통과했다. 실제 source 배열·헤더·타깃·가중치·평가 결과는 읽지 않았고 production 학습·forward는 실행하지 않았다.

두 팔 모두 원 native와 availability mask를 유지한다. self XY/mask와 inputU context는0이며, 다른7 좌표는 F64 차분/정규화 후 [-8,8] clip, 마지막에 FP32다. control만 XY16을0으로 바꾼다. 원 encoder와3×3 pooling 출력, 공유64와115222개 기존 파라미터를 보존하고3137개 residual을 더해 각118359개다. 초기 residual 마지막 층0으로 기존 세 출력이 정확히 일치했다. 가상 비영 residual에서는 q8이 direct에만 직접 주입되지만 공유 trunk gradient에도 영향을 줄 수 있었다. 이는 실제 학습 효과 증거가 아니다.

원 component TRAIN/HOLDOUT은 그대로이며 별도 fitmask만 TRAIN∩nominal이다. 제외된 TRAIN jitter를 HOLDOUT으로 바꾸지 않는다. 모든 원5 loss/direct draw에 fitmask를 쓰고 FULL/TRAIN/HOLD 저장에는 원 role을 쓴다. A의 저장 요약 산술은 원73976행/TRAIN64160, fit14880/inputU697, 제외TRAIN jitter49280과 맞았다. C가 전수 메타데이터 census를 재실행한 것은 아니다. seed1810/direct1811, 양팔4000회, 두 최종 상태의 barrier 뒤 source 예측 순서를 코드로 확인했다.

C 검사기의 주석·tuple 표기·동일행 AST 순서에 대한 과도한 조건으로 발생한 세 가상 실패를 provenance에 보존했다. 최종 검사는 operational AST와 실제 statement 순서를 사용한다. A/B 동결 코드는 바꾸지 않았다.

같은 저장 파라미터 수가 같은 유효 용량을 보장하지 않는다. 기존 weak direct-end는 physical native/quality gold가 아니며 새 성능 개선·팔 선택·모델 채택·목표 완료를 주장하지 않는다. 실제 fit은 별도 root entry 후 가능하다.

[정확한 계약·SHA·한계](REVIEW.json) · [A 독립 가상검사](A_PURE_API_INVENTED_CHECKS.json) · [B 독립 CPU/훈련검사](B_CPU_AND_TRAINER_INVENTED_CHECKS.json)
