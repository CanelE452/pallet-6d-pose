# 방법 근거와 열람 범위

이론은 실사 안전성 보증으로 전용하지 않았다. 아래 공식 초록/출판 페이지를 실제 열람한 범위에서만 연결한다. 모든 PDF 전문을 읽었다고 주장하지 않는다.

- [Safe Policy Improvement with Baseline Bootstrapping, ICML2019](https://proceedings.mlr.press/v97/laroche19a.html): offline 정책 변경에서 baseline과 정보 부족을 분리한다는 배경. 현재 neural score-gap 정책에 논문의 보증이 자동 적용되지 않는다.
- [Regression with Multi-Expert Deferral, ICML2024](https://proceedings.mlr.press/v235/mao24d.html): 고정 예측기와 다른 전문가 사이의 regression deferral 배경이다. 논문의 일관성 결과를 본 합성 split의 실사 안전 보증으로 쓰지 않았다.
- [Two-Player Games for Efficient Non-Convex Constrained Optimization, ALT2019](https://proceedings.mlr.press/v98/cotter19a.html): 비볼록 제약 최적화의 배경. 논문의 게임 기반 최적화를 구현한 것은 아니며 유한 표본 확인과 실제 물리 안전성을 구분한다.
- [Structured Domain Randomization, NVIDIA 공식 페이지](https://research.nvidia.com/labs/amri/publication/prakash2019structured/): 생성 가능한 조건을 근거로 증강을 설계한다는 배경. 이번에는 필요한 인과 조건/원 scene 입력과 fit 예산이 미충족이다.

SRT3D·Blender mask encoding 관련 공식 출처와 실제 열람/미인증 범위는 [관측 계약 문서](OBSERVATION_CONTRACT.md)에 별도 기록했다. 특정 mask의 0/255 의미나 물리 edge 소속은 일반 문헌으로 인증할 수 없다.
