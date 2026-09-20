# 기존 수도레이블 기반 큰 입력 오차 복구 학습

추가 수동 레이블·태그0. 기존 Replay를 복제하여 CLEAN과 DENOISE 각각300step. 같은217장과 합성 replay를 사용하며, DENOISE만 실사 입력 코너를 크게 교란한다. 목표 좌표는 기존 수도레이블 그대로이며 평가 GT는 학습·선택에 쓰지 않는다.

| 평가 | 모델 | 중앙 px | P90 px | PCK20 % | >20→≤10 복구 | <5→>10 손상 | 복구 이미지 |
|---|---|---:|---:|---:|---:|---:|---:|
| full194 | R0 | 7.509 | 43.637 | 77.86 | 0/281 | 0/511 | 0 |
| full194 | BASE | 6.083 | 46.251 | 79.91 | 7/281 | 5/511 | 6 |
| full194 | CLEAN | 5.642 | 43.298 | 81.30 | 5/281 | 7/511 | 4 |
| full194 | DENOISE | 5.943 | 50.516 | 80.24 | 14/281 | 13/511 | 13 |
| retained159 | R0 | 6.974 | 33.177 | 80.45 | 0/187 | 0/447 | 0 |
| retained159 | BASE | 5.625 | 37.209 | 82.08 | 7/187 | 5/447 | 6 |
| retained159 | CLEAN | 5.389 | 33.563 | 83.21 | 5/187 | 5/447 | 4 |
| retained159 | DENOISE | 5.510 | 39.768 | 82.24 | 11/187 | 7/447 | 10 |

사전 screen: {'CLEAN': False, 'DENOISE': False}. 기준별 상태: {'CLEAN': {'recovery5': True, 'frames3': True, 'damage1percent': False, 'pck20_preserved': True, 'source_PCK10_preserved': True, 'source_P90_preserved': True}, 'DENOISE': {'recovery5': True, 'frames3': True, 'damage1percent': False, 'pck20_preserved': True, 'source_PCK10_preserved': True, 'source_P90_preserved': True}}

32장 probe는 수도레이블과의 일치이지 실제 정확도가 아니다. 실제 자연 오류 복구는 위 평가로만 판단한다. 기존 교사 학습과 겹친3장을 포함한 반복 DEV이며 독립 검증이 아니다. 최종 모델은 교체하지 않았다.
