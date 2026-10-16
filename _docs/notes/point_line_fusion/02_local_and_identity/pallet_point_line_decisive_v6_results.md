# 점·선 공동 자세 v6 — 완료, 현재 구조 NO_GO

고정한 3개 구조×3개 seed×2,000 update = 18,000 update를 완료했다.
정본 결과는 `data/pallet/results/pallet_point_line_decisive_v6/`의 `DECISION.json`,
`INTERPRETATION_KO.md`, `REPORT_KO.md`, `index.html`이다.
사전 protocol 문서는 SHA 보존을 위해 수정하지 않고 결과를 이 문서에 기록한다.

- Hough 대 점 전용 주 지표: +0.95% 개선 / −7.94% / −5.85%. 세 seed 모두 gate 실패.
  새 catastrophic frame0/2/8. 직접 선도 안정적 통과 실패. 실사 평가0, 원본GT 변경0.
- **융합의 부분 효과는 관측됨**: 같은 Hough 모델의 선을 끄는 ablation 대비 예측선을
  사용하면 calibration253 주 오차13.65/5.53/10.78% 감소.
- **번호를 제외한 기하도 개선됨**: 새 점 전용 대비 val505 공간 배치 평균5.47/15.43/9.69%
  개선. 그러나 원래 stock YOLO4.18px 대비 새 Hough30.88/27.47/32.14px로 크게 뒤짐.
- 새 점 head는 train128에서도 역할 포함78.7~91.8px, 역할 제외32.7~36.1px. 순수 도메인
  차이 또는 수작업 realGT 오류만으로 설명 불가. 학습/표현/역할 판단이 미완성이다.
- 실제 초기값+예측선보다 GT 근처 초기값+동일 예측선이44~48% 나음(특권 진단).
  GT선은 실제 초기값에서도 개선. 선 mode/역할과 초기 자세의 상호작용이 유력 병목.
- Hough 긴 edge top-mass normal median17.1~23.5px vs 각 edge별 GT-best mode2.0~2.6px.
  GT-best 후보들의 공동 자세 정합을 보장하지 않고, 실제 solver의 mode 결정과도 구분한다.
- 17기하/gradient tests, 실제8-update smoke PASS. NumPy로9개 평가를 별도 재계산해
  primary 최대차6.93e-8, rawframe 최대차5.27e-5px; catastrophic0/2/8도 재현.

이 결과는 YOLO에서 Hough를 사용할 수 없다는 뜻이 아니다. 실제로 YOLO26 특징에서
Hough를 학습하고 선의 부분 이득을 확인했다. 현재로서는 원래 YOLO의 점 정확도를
보존하지 못한 새 head를 채택하지 않는다는 결정이다. DOPE로 돌아갈 근거도 아니다.

원본 외부 v5 번들은 미전달이므로 그 코어의 재현/성능 검증을 주장하지 않는다.
full YOLO 공동학습이 아닌 frozen-backbone 새 head 파일럿이다. 모든 점·선 구조의
불가능성을 증명한 실험도 아니다. 기존41.49px와 지표/모델/모집단을 섞지 않는다.

보고서 Chrome 창 `0x62003e1`이 실제 IsViewable/active이고 제목이 일치함을 확인했다.
기본 xdg-open 시도만으로는 창 확인에 실패해 Chrome을 직접 열어 확인했다.
