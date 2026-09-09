# low_angle_diversity_v1 — 저앙각 structural diversity 가 레버인가

## 1. 제안

가설: R0 의 저앙각(<8도) 학습 프레임은 한 asset 에 쏠려 있고
(`SOURCE_REAL_GAP_AUDIT.md`: 유효자산수 1.01/4), 미사용 oblique pool 로
**총 장수를 고정한 채** 교체하면 real translation/depth 가 줄어든다.

방법: matched replacement (추가 아님). R0 train N 을 정확히 유지하고
저앙각 dominant-asset 프레임을 층화 대응시켜 diverse pool 로 swap.
변수는 **data composition 하나**. 새 loss·self-training·DiffPnP·solver·
architecture·새 render·real GT FT 전부 금지.

판정 지표 (결과 보기 전 freeze): D1 vs C0 에서
depth median >= 5% 개선 AND translation median >= 5% 개선 AND
plastic 의 t·depth 둘 다 개선 AND p90 악화 <= +5% AND
rotation·corner median 악화 < 5% AND detection 손상 없음.

예상 실패 모드: (a) pool 이 저앙각이 아님 (b) 옛 keypoint 규약 (c) eval 누수
(d) 층화 매칭 실패로 SMD > 0.10 (e) 개선 없음.

중단 기준: (a)(b)(c) 중 하나면 학습 전 STOP. (d)면 인과 해석을 낮추고 진행.
screen 실패면 full retrain 하지 않고 `NEXT_RECOMMENDATION = MATCHED_CAPACITY`.

## 2. 결과

(진행 중 — 결과가 나올 때마다 이어 쓴다)
