# 점 주변으로 선 탐색 안내 — 완료

사용자 정정: 점끼리 연결하는 것이 아니라, 점 주변에 선이 있을 가능성을 DHT에 이용한다.
기존 YOLO 점 좌표와 DHT를 고정한 빠른 가능성 검사를 수행했다.

코드: `scripts/research/pallet_point_neighborhood_v8/`
결과: `data/pallet/results/pallet_point_neighborhood_v8/index.html`, `REPORT_KO.md`

합성 재사용 calibration 매칭253/256장, Hough3개 seed. 후보 직선과 가장 가까운
YOLO 점의 거리 d로 prior=.1+.9exp(-d²/(2sigma²))를 만들고 log prior를 기존 DHT logits에
더했다. 점 쌍/연결선/GT anchor/GT pose 입력은 없다. 후보별 약한 위치 편향이며
학습형 attention 또는 feature voting 자체의 재학습이 아니다.

반경은 관측 박스 대각선1/3/10%, 주 비교3%를 실행 전에 고정했다.
대조군: 다른 이미지 점의 ROI 정규화 위치를 현재 ROI로 옮긴 것, 박스 코너와 변 중앙.
추론 중 GT 파일0개, 원래 YOLO 좌표 그대로 보존, 원본 모델 SHA 일치.

동일 일대일 선 매칭 normal-distance 평균(px), baseline→3% 안내:

- seed1:23.697→24.239
- seed2:20.962→21.354
- seed3:22.584→22.741

1/10% 반경도 3seed 일관된 개선 없음. 사전 파일럿 조건 실패.
이 지표는 점/자세 오차와 다르다. 실제 자세 및 실사 개선을 평가한 것이 아니다.

후속 진단: 역할이 틀린 선(그 역할의 GT선 끝점 normal error>10px) 중
반경3% 안의 점 근처를 지나가는 비율97.2/98.1/99.0%.
어느 GT선에든 가까운 예측선 비율96.9~98.0%와 달리, 전체GT 경계포착은68.8~74.5%.
후자는 역할 중복/누락을 의심하게 하는 자료이며 전자는 느슨한 any-GT 일치율이다.
“선 검출 정확도98%”로 오해하면 안 된다. 점 근접만으로 방향/역할/중복 문제는 해소되지 않는다.

단일 점은 여러 방향의 선을 동일하게 지원, 점 순서 불변, 유한 비영점 gradient,
soft prior 범위 검사 통과. 독립 동차직선 수치 감사4,554개 조건·프레임 PASS,
최대 지표 차이1.14e-13px. HTML 로컬 참조8개, 사례2장 육안 확인.

판단: 이 고정 후보가중치 방식은 개선 근거 없음. 점 주변 영상 특징과 선 방향/역할을
함께 학습하는 별도의 아키텍처는 아직 검증하지 않았다. 원래 YOLO는 변경하지 않았다.

실행:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_neighborhood_v8/run.py
MPLCONFIGDIR=/tmp/pallet-neighborhood-v8-matplotlib /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -u scripts/research/pallet_point_neighborhood_v8/report.py
```
