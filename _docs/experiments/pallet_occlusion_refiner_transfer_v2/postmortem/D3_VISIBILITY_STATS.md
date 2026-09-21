# D3 — 가시성/가림 출처 감사

[확인] **EXTERNAL_OCCLUSION_CORNER_RECOVERY_UNVERIFIED**. 외부 가림 복구가 없다는 뜻이 아니라 외부/자기 subtype을 확인할 독립 판정이 부족하다는 뜻이다.

[확인] 기존 prediction-blinded 사람 검토 169/713코너가 있다. 좌표 source=unknown과 가시성 검토 여부는 다른 정보다. 사람의 `o`는 occluded 일반 판정이며 subtype GT는 아니다.

## 검증 가능한 subtype

| [확인] 그룹 | 코너 | R0 % | N2 % | Replay % | A10 % | A11 % | 획득/손실 | hard 복구 | good 손상 |
|---|---|---|---|---|---|---|---|---|---|
| UNKNOWN | 669 | 43.20 | 49.78 | 49.63 | 50.37 | 49.48 | 19/25 | 9 | 1 |
| VISIBLE | 44 | 43.18 | 38.64 | 40.91 | 43.18 | 43.18 | 0/0 | 0 | 0 |

## 사람이 판정한 일반 가시성

| [확인] 그룹 | 코너 | R0 % | N2 % | Replay % | A10 % | A11 % | 획득/손실 | hard 복구 | good 손상 |
|---|---|---|---|---|---|---|---|---|---|
| OCCLUDED_UNSPECIFIED | 125 | 20.00 | 18.40 | 20.80 | 19.20 | 18.40 | 6/7 | 1 | 1 |
| UNKNOWN | 544 | 48.53 | 56.99 | 56.25 | 57.54 | 56.62 | 13/18 | 8 | 0 |
| VISIBLE | 44 | 43.18 | 38.64 | 40.91 | 43.18 | 43.18 | 0/0 | 0 | 0 |

## 기존 자동 기하/depth 후보 — 정답 아님

| [확인] 그룹 | 코너 | R0 % | N2 % | Replay % | A10 % | A11 % | 획득/손실 | hard 복구 | good 손상 |
|---|---|---|---|---|---|---|---|---|---|
| AUTO_SELF_OCCLUDED | 135 | 25.19 | 28.15 | 26.67 | 29.63 | 25.19 | 3/9 | 3 | 0 |
| EXTERNAL_OCCLUSION_CANDIDATE | 169 | 26.04 | 23.67 | 26.04 | 25.44 | 24.85 | 6/7 | 1 | 1 |
| SELF_VISIBLE_CANDIDATE | 409 | 56.23 | 66.50 | 66.01 | 66.75 | 66.99 | 10/9 | 5 | 0 |

## annotation 표기 — 혼합 provenance

| [확인] 그룹 | 코너 | R0 % | N2 % | Replay % | A10 % | A11 % | 획득/손실 | hard 복구 | good 손상 |
|---|---|---|---|---|---|---|---|---|---|
| occluded | 260 | 22.69 | 23.46 | 23.85 | 24.62 | 21.92 | 9/16 | 4 | 1 |
| visible | 453 | 54.97 | 63.80 | 63.58 | 64.46 | 64.68 | 10/9 | 5 | 0 |

[확인] A11 hard recovery 9개 중 manual external subtype으로 확인된 개수는 0개다. 자동 SELF_VISIBLE/SELF_OCCLUDED/EXTERNAL_CANDIDATE는 보조 자료로만 집계했다. image-level occlusion을 코너별 GT로 복사하지 않았다.

[확인] [검토 갤러리](D3_REVIEW_GALLERY.html): 65개 고유 코너, 48개 프레임. 모든 A11-only/A10-only/hard recovery/good damage와 고정 랜덤 20코너를 포함하고 중복만 합쳤다. full RGB, bbox crop, GT와 다섯 후보의 좌표/오차를 표시한다.

[확인] 현재 사람의 새 검토를 수행하지 않았으며 annotation도 변경하지 않았다. 모델 출력을 보여주는 사후 검토 자료이므로 향후 독립 visibility 정답으로 쓰려면 prediction-blinded review가 별도로 필요하다.
