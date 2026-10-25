# 고정 crop에서 확인한 점

이 문서는 고정 정책으로 선정된 8개 사례를 설명합니다. 모집단을 대표하는 비율이나 실패 원인을 확정하는 분석은 아닙니다. 정량 분모는 [주 보고서](PORTABLE_REPORT_KO.md)와 전체 ledger를 사용합니다.

- [WOOD D→E, g221](figures/CASE_01_G0221.png): 수동 클릭 reference 오차2.09px이며 human D, legacy D, 후보 E입니다. 원 crop에는 목재 끝과 회색 바닥이 함께 있고, 원 q 십자 중심은 경계의 바닥 쪽에 놓입니다. 96×96 입력의 pixel support는 전부 유효합니다. 좌표 오차가 작아도 경계에서 중심 픽셀과 주변 부위 보임은 다를 수 있음을 보여줍니다. GT 위치로 crop을 옮기지 않았습니다.
- [SQUARE E→D, g996](figures/CASE_07_G0996.png): 수동 reference 오차1.14px이며 human E, legacy E, 후보 D입니다. 원 crop 안에는 녹색 팔레트와 주황색 콘이 함께 보입니다. 팔레트 일부가 crop 안에 존재하는 사실과 해당 native의 외부 가림 라벨은 구분해서 읽어야 합니다. 이 그림만으로 어느 픽셀이 물리 native 정답인지 또는 어노테이션 오류인지 확정하지 않습니다.

두 사례에서 보이는 단서는 목표·라벨 의미 차이와 전이 실패를 따져 볼 근거이지만, 어느 한 원인으로 모든 오류를 설명하지 않습니다. 현재 결과를 보고 모델·입력·문턱을 바꾸지 않았습니다. 실제 8개 panel에는 원 q/L, 원 native, pixel support, reference provenance, 세 모델의 저장 응답을 나란히 표시합니다.
