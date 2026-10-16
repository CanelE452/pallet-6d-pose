# V27 점/Hough 실제 사례 시각화

V27 완료 결과를 읽기만 하며, GT oracle8.78px의 크기·개선 사례·후보 선택 실패·oracle도 실패하는 사례를 별도 산출물로 만든다. 선은 새로 검출한 Canny 선이 아니라 실제 DHT 캐시의 점별 top3다. 원본/모델 입력/확대를 분리하고 이미지 밖 연장은 점선으로 표시한다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_direction_utility_v27_cases/show_cases.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_direction_utility_v27_cases/deliver_cases.py
```

[사례 보고서](../../../data/pallet/results/pallet_direction_utility_v27_cases/index.html) · [좌표와 출처](../../../data/pallet/results/pallet_direction_utility_v27_cases/CASES.json) · [MD](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_direction_utility_v27_cases.md)
