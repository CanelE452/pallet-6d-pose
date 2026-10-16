# ResNet18 weight decay 고정 진단

동일 TRAIN256·fresh ImageNet/seed42·배치순서·augmentation RNG를 쓴 폐기용2fit입니다. 각500updates 이후 같은TRAIN16장을 평가했으며 실사/VAL평가·full60epoch 재학습은 하지 않았습니다.

| Adam WD | step | loss | zero-map loss | max confidence | valid/144 | matched/16 | conditional original pixel median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0 | 0.000511305014 | 0.000511305348 | 4.30959e-06 | 0 | 0 | NA |
| 0.0001 | 100 | 0.000510143203 | 0.000511305348 | 0.0132742 | 0 | 0 | NA |
| 0.0001 | 500 | 0.000507580553 | 0.000511305348 | 0.00548503 | 0 | 0 | NA |
| 0 | 0 | 0.000511305014 | 0.000511305348 | 4.30959e-06 | 0 | 0 | NA |
| 0 | 100 | 0.000509342673 | 0.000511305348 | 0.00840942 | 0 | 0 | NA |
| 0 | 500 | 0.000509229765 | 0.000511305348 | 0.00846379 | 0 | 0 | NA |

학습에 포함된 작은256장 중16장 결과이므로 일반화·논문성과를 입증하지 않습니다. threshold0.1/IoU0.5와 모든 누락 수를 유지했습니다. 초기동일state의 작은 GPU출력차로 첫실행이중단됐고 첫500회는보존한채 WD0의0→500만진행했습니다.

공식 [Simple Baselines Adam 분기](https://raw.githubusercontent.com/microsoft/human-pose-estimation.pytorch/master/lib/utils/utils.py)는 lr만 지정하고 WD는 SGD에만 전달합니다. 작은 통제실험에서 관측된 효과를 full분포로 확대하지 않습니다.

[사전프로토콜](DIAGNOSTIC_PROTOCOL.json) · [중단/수치비교 addendum](NUMERICAL_CONTINUATION_ADDENDUM_KO.md) · [전수결과와SHA](DIAGNOSTIC_RESULTS.json)
