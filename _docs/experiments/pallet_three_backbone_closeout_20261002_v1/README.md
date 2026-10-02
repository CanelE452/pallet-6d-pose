# 세 백본 실험 마감 자료

[한국어 전체 보고서](REPORT_KO.md) · [출처·출력 manifest](REPORT_MANIFEST.json) · [검증 결과](TEST_RESULTS.json)

세 고정 backbone의 별도 보정 학습에서 재사용 DEV의 조건부 2D 중앙값 감소를 관찰했다. 치수 입력의 안정적 T·R 공동 개선과 독립 TEST 일반화는 입증되지 않았다. 정확도는 세 seed 통계 평균이며 통합 runtime은 대표 seed1이다.

[전체 JSON](SUMMARY.json) · [정확도 CSV](ACCURACY_SUMMARY.csv) · [runtime CSV](RUNTIME_SUMMARY.csv) · [runtime 차이 CSV](RUNTIME_OVERHEAD.csv) · [paired CSV](PAIRED_SUMMARY.csv) · [direct 치수 CSV](DIRECT_DIMENSION_SUMMARY.csv) · [direct pose CSV](DIRECT_POSE_SUMMARY.csv)

그림: [정확도 PNG](figures/within_backbone_accuracy.png) / [PDF](figures/within_backbone_accuracy.pdf), [runtime PNG](figures/runtime_overhead.png) / [PDF](figures/runtime_overhead.pdf). 기존 RGB 갤러리는 전체 보고서에 상대경로로 연결되어 있다.

저장소 루트에서 기존 공개 source artifact를 유지한 채 CPU로 재생성한다. Python에 NumPy, Matplotlib, Pillow가 필요하다.

```sh
python -B scripts/research/pallet_three_backbone_closeout_20261002_v1/report.py build
python -B scripts/research/pallet_three_backbone_closeout_20261002_v1/test_report.py
```

SOURCE_BINDINGS.json은 고정된 원본 hash다. source가 달라지면 중단하며 자동 재봉인하지 않는다. 원영상/GT 재평가·GPU·checkpoint·대용량 raw 캐시는 필요하지 않다. 기존 artifact 및 manuscript를 수정하지 않는다.
