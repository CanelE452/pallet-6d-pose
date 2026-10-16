# V28 선 전용 추론 개입

V27 사례3의 점 보정 후보를 제거하는 추론과, DHT 앞의 점 주변 가이드를 우회하는 추론을 분리한다. 새 학습과 운영 모델 변경은 없다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_line_only_v28/probe28.py --phase score
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_line_only_v28/probe28.py --phase dht
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_line_only_v28/report28.py
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_line_only_v28/deliver28.py
```

`dht`는 GPU가 필요하다. `score`는 저장된 후보·가중치·이동량을 재조합하며 기존 점과 점에 따른 선 선택을 유지한다. GT는 평가 및 별도 oracle에만 사용한다. `gate_bypass`는 명시적 점 힌트 없는 선 생성이며8점 포즈를 디코딩하지 않는다.

[결과 MD](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_line_only_v28.md) · [HTML](../../../data/pallet/results/pallet_line_only_v28/index.html)
