# 미적용 원시 후보 진단

Both classifiers failed source safety calibration. Raw argmax is not promoted, not used as pseudo labels, not counted as solved goal.

| 학습기 | 변경 후보 이미지 | >20→≤10 복구 | >100→≤10 복구 | 정상 코너 손상 | PCK20 |
|---|---:|---:|---:|---:|---:|
| SYN | 34 | 46/281 | 31 | 85/511 | 69.33% |
| MIX | 32 | 46/281 | 31 | 74/511 | 70.39% |

의자·콘 사진: {'SYN': 23.16260127521826, 'MIX': 23.16260127521826}
현재 실제 채택 출력은 R0 그대로이며, 이 후보 진단으로 안전 기준을 완화하지 않았다. 새 정답/태그0. 반복 DEV의 진단이지 독립 검증이 아니다.
