# 원본 문제 사례: 저장된 21개 모델의 최종 점·선 검토

**사례:** `eval_pallet07:1778652166837872128` · 640×480 · DEV의 한 이미지. 새 12개 모델과 재사용 9개 대조군의 완료된 원본 예측만 읽었다. 모델 실행, 후보/순열 탐색, GT 수정, 보정 적용, 학습·선택·전체 판정 변경은 없다. 아래 PASS는 이 저장 자료 검증의 완료성을 뜻하며 정확도 성공이 아니다.

**공식 번호 오류는 남았다.** 21개 모두 최고 confidence 후보가 IoU≥0.5로 매칭되지만 공식 이미지 median은 **232.93–273.87px**이다. 점을 정답 근처에 배치하는 것과 camera-facing semantic ID를 맞추는 것은 다르다. 원본 스크린샷 R0는 267.929px이며, 같은 고정 진단 순열로만 재대응하면 25.706px다. 원래 GT 감독 마스크는 `[0,1,2,4,5,6,7,8]`: **7코너+중심, 총8점**이다. 잘린 3번 코너(v0)는 제외하고 가림 v1은 포함한다.

다음 순열은 기존 GT 기반 진단 `[1,5,6,2,0,4,7,3,8]`를 모든 모델에 똑같이 적용한 `P_diag[j]=P[perm[j]]`다. GT와 감독 마스크는 그대로다. **실제 추론에서 알 수 있는 교정 규칙이 아니고 공식 점수도 아니다.** 모든 seed를 제시하며 사례에서 좋은 모델을 고르지 않는다.

| Arm | 공식 median px (seed1 / 2 / 3) | 고정 GT 진단 순열 median px (seed1 / 2 / 3) |
|---|---:|---:|
| point_only | 260.68 / 250.98 / 262.46 | 34.02 / 43.58 / 34.45 |
| hough_features | 249.51 / 248.74 / 262.60 | 45.56 / 46.41 / 30.52 |
| hough_joint | 270.06 / 271.55 / 244.53 | 20.70 / 25.86 / 49.00 |
| balanced | 271.79 / 264.27 / 240.10 | 22.65 / 32.32 / 52.44 |
| pcgrad | 262.71 / 240.37 / 232.93 | 35.10 / 51.44 / 59.85 |
| balanced_pcgrad | 262.06 / 266.88 / 263.57 | 36.47 / 24.18 / 33.71 |
| incidence | 258.44 / 273.87 / 241.00 | 38.03 / 11.40 / 47.91 |

고정 순열 뒤에도 11.40–59.85px의 위치 오차가 남는다. 번호 대응 오류가 큰 비중을 차지한다는 관측이지 GT의 번호를 바꿔야 한다는 결론은 아니다. 정본 GT는 `camera_dynamic_0123_v4`이며, annotation의 canonical-axis migration 후보는 `CANDIDATE_ONLY_UNCONFIRMED_SIGN`이다. 그 상태를 실제 카메라 yaw 정답이나 새 GT 채택 근거로 쓰지 않았다.

**같은 역할7(GT4–7)을 따로 확인했다.** 정답은 `[21,347]→[22,391]`, 길이44.01px, 두 endpoint의 annotation v2다. 이 두 점의 표시가 전체 영상의 물리 edge 가시성/recall 정답을 만들어 주지는 않는다. 저장된 18개 DHT 파일의 채널7만 사용했으며 point_only의 선은 만들지 않았다. `hough_features`의 채널7에는 역할별 auxiliary 감독이 없어 정답 역할이 학습됐다고 가정하지 않는다.

`hough_joint`, `balanced`, `pcgrad`, `balanced_pcgrad`의 12개 모델 모두 같은 역할의 분포가 잘못된 원래 P4–P7보다 **GT4–7 pair를 더 지지**했다. 조건부 mixture compatibility의 GT/pred 비는 1.53–3.10배다. 특히 hough_joint seed1의 역할7 peak 선은 GT 두 endpoint까지 평균1.68px인데 원래 예측 P4–P7까지277.64px이고, 공식 점 median은270.06px다. 따라서 이 사례를 일괄적으로 “점과 선이 함께 같은 위치로 틀렸기 때문”이라고 설명하면 증거와 맞지 않는다. 정답에 맞는 일부 선 증거가 있어도 전체 점 ID가 교정된 것은 아니다.

**Incidence 모델의 역할7에는 다른 현상이 관측됐다.** 정답 선의 법선각은178.70°인데 세 seed의 peak 법선각은80°/80°/78°로, 방향오차가81.30°/81.30°/79.30°다. 정상적인 GT4–7 방향을 나타내는 peak가 아니다. 그럼에도 GT pair와 잘못된 원래 P4–P7 pair의 mixture cost는 비슷하다.

| incidence seed | GT pair cost | 원래 P4–P7 pair cost | Peak 선의 GT / pred endpoint 평균거리 px |
|---|---:|---:|---:|
| 1 | 1.660 | 1.571 | 21.75 / 16.57 |
| 2 | 1.496 | 1.641 | 21.75 / 17.57 |
| 3 | 1.675 | 1.632 | 21.62 / 16.88 |

Cost는 저장 sigmoid 분포를 실제 footprint 안에서 정규화한 뒤, **동일 후보 선에 대한 두 endpoint의 pseudo-Huber cost를 먼저 합쳐** 주변화한 값이다. σ는 실제544×640 입력 대각선의1%(8.40 입력px, 원본에서약11.02px)다. 학습과 같은 기하식을 사용했지만 최종 검출점은 학습의 TAL-positive dense anchor 집합과 다르므로 실제 학습 loss 재현값은 아니다. 독립 계산한 원본 좌표 affine·sigmoid·유효 격자도 확인했다. Backprojection의 GT segment 평균은 세 incidence 모델에서도 예측 segment보다 높았으므로, peak·분포·공간 평균을 서로 동일한 “정답 확률”로 해석하지 않았다.

**구조 한계와 관측을 구분한다.** [ARCHITECTURE_LIMITS](../architecture_limits/ARCHITECTURE_LIMITS.md)의 무한선/끝점/role permutation/거친 격자 한계와 이번 관측은 양립한다. 해당 incidence 역할의 방향 오류와 비슷한 pair cost는 무한선 일치만으로 원하는 semantic edge를 보장하지 못한 구체적인 단일역할 증거다. 그러나 이것만으로 전체12개 역할의 공동 오류, 실제 학습 중 원인, 다른 이미지의 실패 원인까지 증명하지는 못한다. PCGrad가 gradient 충돌을 조절하는 것과 이러한 semantic 모호함을 해소하는 것은 다른 측정 대상이다. 같은 이미지의21개 출력은21개 독립 실사 표본도 아니다.

**검증:** 21개 공식 CSV median을 저장 원본점으로 독립 재계산해 모두1e−9 이내 일치했다. 개별 CSV 점오차의6자리 저장 반올림 차이는 최대4.986e−7px. 입력 SHA를 분석 전후 재검사했으며 원본 raw와 정본 이미지 SHA는 같다. 전체 실험의 정확도/통계 판정은 별도 SUMMARY/VERDICT를 따른다.

산출물: [전체 좌표·선 통계 JSON](REVIEW.json), [21개 공식/진단 CSV](CASE_METRICS.csv), [입력 SHA](INPUT_HASHES.json), [실행 소스](review.py).

재현(CPU, 저장 파일만 읽음):
```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python data/pallet/results/pallet_dht_coupling_v2/provenance/problem_case_final_review/review.py
```
