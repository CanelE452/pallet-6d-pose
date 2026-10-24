# V213 고정 실사 평가 코드 준비

C63은 사전 지정 주 모델이고 C62는 dense-point만 사용하는 대조 팔입니다. 두 팔은 같은 C61 RAW·native·q8 구조(118,359개 파라미터)와 4,000회 갱신 상태를 사용합니다. C63의 source weak-bridge 가중치 0.25는 학습 프로토콜의 차이이며, 평가 시 입력·문턱·구조를 바꾸지 않습니다.

원래 RAW10/META6·F64 q/L·native8·input U를 유지합니다. 다른 7개 초기 query의 availability와 F64 상대좌표 계산도 V203 함수를 그대로 사용하며, GT endpoint·reference·라벨은 특징에 들어가지 않습니다. 두 SOURCE 최종 상태와 독립 exact-fit PASS를 먼저 확인하고, 두 실사 출력 전체가 봉인된 뒤에만 별도 entry로 저장 평가를 실행합니다.

현재 14개 모델과 public weak 13개 모델을 비교합니다. 기존 v61/legacy/C52~C61은 저장 출력만 사용하고 이전 12개 모델의 표는 완전히 같아야 합니다. 주평가 1,237개(D 1,147/E 90), 원1784/U45, self 전체136과 paired21, reference ≤3px human-D 461과 manual-near376을 구분합니다. Public360의 weak204/UNKNOWN156 및 legacy 결측도 보존합니다.

예측 가상검사 18개, 저장 평가 가상검사 24개, 새 source 봉인·역할·기존 C60/C61 경로 검사 15개가 통과했습니다. 원 입력·문맥·추론·순수 평가 커널의 함수 AST 동일성도 기록했습니다. 실제 데이터·가중치·라벨·점수·NPZ 헤더/본문을 열지 않았고, 실제 forward는 0회입니다. 두 actual entry는 approved=false입니다.

실제 실행은 root가 V212 최종 COMPLETE·독립 REVIEW·trainer/protocol·두 최종 checkpoint barrier의 정확 바인딩과 terminal exit0를 채운 후에만 가능합니다. 출처의 point/finite-end 약한 감독은 인간 native visibility gold가 아니며, 반복 노출 평가를 새 blind 검증이나 개선 원인 증명으로 해석하지 않습니다. 결과에 따른 모델 채택·선택은 하지 않습니다.
