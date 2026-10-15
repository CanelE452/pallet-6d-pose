# 완료된 비교·통제 진단의 최종 독립 해석 검토

현재 최종 문서의 결론과 한정 범위는 결과에 맞는다. 미해결 material finding은 없으며 원 봉인 결과·코어·모델을 수정하지 않았다. 새 추론·학습·자세 풀이는 실행하지 않았다.

현재 방법은 같은 TEST49의 기존 RGB/경계 방법보다 위치 평균과 동시 성공 수가 좋지 않다. O/T 자체가 무효인 것은 아니다. 같은 corrected BI-only와 비교하면 4.692→3.514cm, 성공38→39로 개선된다. 다만 평균 위치 개선1.178cm 중 약1.057cm(89.7%)는 pallet09 한 장에서 발생한다. 이를 광범위한 우월성으로 읽지 않는다.

보정 효과도 조건부다. 같은 Σ에서 OT를 켠 corrected BI는 raw BI의3.797cm보다 좋지만, OT를 끄면 corrected4.692cm가 raw4.623cm보다 크고 회전도 악화한다. 보정×OT interaction을 구분해야 한다. 고정 covariance를 raw q에 붙인 것은 좌표 개입 통제이며 새로운 실제 raw noise calibration이 아니다.

가장 직접적인 진단은 **reference를 허용하는 것과 정확한 자세를 선택하는 것이 다르다**는 점이다. compatible TEST42에서 budget6은 reference 가능 수를8→23, 공동 채택 수를29→44로 늘려도 성공은39→37로 줄었다. 현재 covariance/관측/등록 모델/주석 reference 사이의 불일치가 크다는 근거이며 budget6이 해결책이라는 근거는 아니다. 이 reference는 독립 물리 GT가 아니다.

Full-human raw 상태는 같은 RAW_STANDARD_COV3.790cm/39와 비교해야 한다. Full-human3.838cm/38, 추가budget6 3.707cm/38은 전체 격차를 해소하지 못했다. Binary는 초기S/T 유지+knownD/E 교정+나머지SELF/UNANNOTATED abstention의 지원점 선택 실험이며 순수 이진 classifier-only 효과가 아니다. 알려진 falseOT 삭제와 reference O center도 다른 오차·제약을 남긴다. 단일 개입이 개선하지 못한 결과로 상태/영역 오류가 중요하지 않다고 결론내릴 수 없다.

Reference 시작점 추가는 **95건 모두 actual R/t가 정확히 같다**. 이 시작점이 해결하는 검색 부족은 관측되지 않았다. 이는 전체 SE3 검색이 완전하다는 증명은 아니며, reference가 infeasible하거나 관측 목적이 reference를 선호하지 않으면 정확한 초기값도 결과를 바꾸지 못한다. MODEL_IDEAL은95/95 구성 자세를 찾고 수치 오차 수준으로 복원하지만 여러 입력을 동시에 이상화했고 reference start도 포함한다. 내부 모델 일관성 점검이며 실제 RGB O/T 추론이나 OT 우월성의 증거는 아니다.

후보 선택은 일부 위치 손실을 설명하지만 전체 해결책은 아니다. 같은 pool의 nearest-coarse는3.326cm로 내려가도 회전/ADD가 악화하고 성공39는 그대로다. 개선은 주로 pallet09 한 장이다. GT-minimum-translation의3.092cm가 현재 저장 pool의 위치 오차 하한이며 기존 RGB2.619cm보다 높다. 저장된 모든 후보와 같은 fallback을 직접 확인하면 동시 성공이 가능한 최대도 TEST39이다. 따라서 이 **고정 유한 pool 안**의 selector만 바꾸어 기존 RGB42/경계43 성공이나 위치 평균을 달성할 수 없다. 전체 연속 feasible set에 더 좋은 자세가 없다는 뜻은 아니다.

실제 성공→실패4건이 모두3BI이고 local Jacobian rank6이라는 사실은 sparse noise/P3P branches에 대한 취약성을 뒷받침한다. 네 건을 모두 연속 미결정 가족이라 부르거나 residual0을 정확한 물리 자세의 증거로 부르지는 않는다. 관측·모델·가정 covariance와 적은 지원점/선택을 주요 병목으로 지목할 근거는 있으나, 전체 격차를 각 원인에 유일하게 배분한 것은 아니다. CAL joint calibration·coarse prior·새 공간 근거는 아직 검증되지 않은 후속 가설이다.

최종 root README와 FINAL_FINDINGS의 matched control,89.7% outlier, oracle/ideal/search의 경계 및 reusedTEST 한계를 확인했다. 1,520행·16arm×95, 현재 실행21개 source/snapshot와 prediction/protocol seal은 유지되어 있다. 최종 문서·source SHA와 근거는 [FINAL_RESULT_REVIEW_V1.json](FINAL_RESULT_REVIEW_V1.json)에 보존했다.
