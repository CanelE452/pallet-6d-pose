V1 실제 감사는 원 query_xy=null을 scalarNaN으로 비교한 schema adapter 오류로 실패했다. 원 unavailable query의 저장 형식은 float64 [NaN,NaN]이며 L=null은 scalarNaN이다. V1 code/protocol/READY/entry/stdout/실패 산출물은 그대로 보존했다. V2는 새 폴더와 출력 경로를 사용하며 production metadata identity helper 한 곳만 이 저장 형식에 맞췄다.

18GT 선언 whitelist·576native·1152매핑·원 q/L/native·component HOLDOUT·inputU80는 V1과 동일하다. 선택 및 whitelist와 원행 파일 네 개의 바이트 동일성을 확인했다. GT·점수·예측 배열·가중치는 V2 준비 중 다시 읽지 않았다. 원 12개 가상검사와 실제 identity helper의 null/finite/shape/원 unavailable 13개 회귀가 모두 exit0로 통과했다. 실제 V2 평가는 아직 0회이며 별도 root entry를 기다린다.
