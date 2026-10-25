# V253 SOURCE 저장 실패·reference 분포 진단

이 결과는 저장된 세 모델 점수와 두 reference9 cache만의 기술적 진단입니다. 새 모델 forward·학습·bank 적합·문턱 조정·픽셀 열람은 없습니다.

{
  "FIT": {
    "rows": 5876,
    "D": 5406,
    "E": 470,
    "D_TREATfalseE_total": 574,
    "D_BASEcorrect_to_TREATfalseE_new": 544,
    "D_BASEfalseE_to_TREATfalseE_existing": 30,
    "D_BASEU_to_TREATfalseE": 0,
    "D_TREATfalseE_rate_originalD": 0.10617832038475768,
    "E_BASEmiss_to_TREATcorrect": 0,
    "E_BASEcorrect_to_TREATmiss": 39,
    "E_CONTROLmiss_to_TREATcorrect": 31,
    "E_CONTROLcorrect_to_TREATmiss": 10
  },
  "DEV": {
    "rows": 1741,
    "D": 1628,
    "E": 113,
    "D_TREATfalseE_total": 212,
    "D_BASEcorrect_to_TREATfalseE_new": 193,
    "D_BASEfalseE_to_TREATfalseE_existing": 19,
    "D_BASEU_to_TREATfalseE": 0,
    "D_TREATfalseE_rate_originalD": 0.13022113022113022,
    "E_BASEmiss_to_TREATcorrect": 37,
    "E_BASEcorrect_to_TREATmiss": 9,
    "E_CONTROLmiss_to_TREATcorrect": 3,
    "E_CONTROLcorrect_to_TREATmiss": 6
  }
}

trueD가 TREAT E가 된 전체 수와 BASE D정답에서 새로 손상된 수를 구분합니다. 원 분모를 유지한 모든 가족/FIT·DEV/D·E/원SYN주소/세 view 표와 빈 셀을 보존했습니다. score 분포차이는 원인·라벨 오류·정보 불가능성의 증명이 아닙니다. FIT-D reference 자기포함과 반복 노출된 SOURCE DEV의 한계를 유지합니다. SYN 원native 입력주소는 physical canonical gold가 아니며 YCB associated-r 보조 라벨의 native·SELF는 UNKNOWN입니다.

FIT_TILE_SELECTION은 36개 사전 고정 셀에서 sample index 최소 두 개와 빈 셀을 보존한 시각화 후보일 뿐, 픽셀 판독·새 라벨·학습 admission이 아닙니다. GT/r/owner를 추가하지 않았으며 현재 실사 개선이나 채택을 주장하지 않습니다.
