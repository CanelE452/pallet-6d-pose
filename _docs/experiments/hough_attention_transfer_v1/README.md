# Synthetic-only Direct Hough attention A/B

사용자 요청: 합성 데이터로 attention과 선을 학습한 다음, 실제 영상에서도 선이 맞는지와 어디를 참조하는지 직접 확인한다.

## 결과 보기 전에 고정한 설계

- A: 기존 DirectHoughModel의 역할 query, cross-attention, Hough head를 선 공간 CE로 학습.
- B: 같은 모델, 같은 초기값과 배치 순서, 같은 optimizer에 `-log(sum(attention * visible_pallet_mask))`를 가중치 1로 추가.
- foreground 마스크는 합성 JSON의 실제 렌더링 `mask_rle`에서 복원한다. 팔레트 전체 영역 지도이며 각 선의 물리적 가시 영역 지도가 아니다. 가려진 cuboid 선에 대해서도 팔레트 전체의 문맥을 참조할 수 있다.
- frozen backbone: `weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth`의 VGG. G38 합성 학습 이력, real fine-tuning 없음. 기존 A1은 paper_4pallet 전체를 학습한 이력이 있어 재사용하지 않는다.
- 입력: 원본에 reflect padding 100px, 400×400 resize, ImageNet normalization. 합성·실사 동일. backbone 학습은 448px이었으며 이번 고정 readout은 기존 50×50 토큰 격자를 위해 400px을 사용한다.
- 학습: 합성 2,048장, 합성 validation 256장, 합성 test 256장. 250장씩의 실제 렌더링 실행 묶음을 서로 다른 split에 할당한다. split seed 17.
- 추가 합성: 명시적 `camera_dynamic_0123_v4` 라벨을 가진 v4_split_base 128장.
- 실사: `challenge.data_paths.EVAL_CANONICAL`의 open DEV 세 그룹 52장. `objects[0].split=eval`, manual GT만. migrated v2와 원본 좌표의 일치를 검증한다. 이미 존재하는 DEV 평가이며 새로운 final test가 아니다.
- A/B 각각 seed 1, 2, 3. AdamW lr 0.001, weight decay 0.0001, batch 12, 3,000 step 고정. 결과가 좋아 보이는 checkpoint를 사후 선택하지 않는다.
- sanity: 첫 합성 학습 32장, 별도 seed 101에서 1,500 step의 학습 추세를 확인한다. 이는 본 A/B 체크포인트와 다른 모델이다.

## 측정과 시각화

선 역할은 `[01,12,23,30,45,56,67,74,04,15,26,37]`의 camera-dynamic cuboid 경계선이다. 실제 판재의 모든 물리적 선을 검출하는 과제가 아니다.

주 지표는 원본 이미지에서 GT 선분의 양 끝점과 예측 무한직선 사이 평균 수직거리를 영상 대각선으로 정규화한 값이다. 원본 픽셀 각도 오차와 거리px, 각 중앙값·90백분위도 함께 저장한다. 기존 canonical50 격자 지표는 별도로 기록하며 원본 이미지 각도와 혼동하지 않는다. supported는 영상과 교차하는 유효 GT 선분이라는 뜻이며 물리적 가림 여부가 아니다.

실제 추론 입력은 RGB뿐이다. GT 점·마스크는 손실, 오차 측정, 그림의 정답 선에만 사용하고 crop이나 예측 입력을 만드는 데 사용하지 않는다.

시각화는 사전 고정한 seed 1에서 파일 ID hash 순서로 선택한다. 원본 입력(패딩 포함), A/B의 attention, GT와 예측 선을 나란히 표시한다. 12개 선 각각의 attention도 제공한다. 값은 cross-attention head 평균의 실제 가중치이며 A/B는 같은 색 척도를 쓴다. 패딩을 제거한 영상에 가중치를 억지로 늘려 표시하지 않는다.

## 해석 범위

관심 영역으로 attention이 더 모였다는 사실과 선이 더 정확하다는 사실을 구분한다. B의 attention이 개선되어도 실제 선 오차가 줄지 않으면 전이 개선으로 판단하지 않는다. Attention은 토큰 혼합 비율이며 인과적 픽셀 중요도의 증거는 아니다.

백본 G38 원본 학습 경로가 삭제되어 이번 합성 평가와 사전학습 이미지의 모든 byte 중복을 확인할 수 없다. 본 실험의 head 학습 split은 검증하지만 백본까지 완전히 미관측인 합성 test라고 과장하지 않는다. 실사 52장은 제한된 기존 개발 세션이며 야간·목재 전체 일반화나 독립 최종 성능을 주장하지 않는다.

실행과 결과: `data/pallet/results/hough_attention_transfer_v1/`. `CONFIG.json`, `manifest.json`, `CHECKS.json`, 각 HISTORY, `RESULTS.json`, `PER_ROLE.csv`, `REPORT.md`, `attention_gallery.html`에 저장한다.

```bash
/home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/hough_attention_transfer_v1/run.py --phase all
/home/minjae/anaconda3/envs/pallet-pose/bin/python scripts/research/hough_attention_transfer_v1/analyze.py --run-dir data/pallet/results/hough_attention_transfer_v1
```

## 완료 결과 (2026-09-07)

6개 본학습 모두 3,000 step 완료. 세 쌍의 초기 state SHA, 최종 checkpoint SHA, split 분리와 실제 attention gradient·forward parity·좌표 roundtrip 검증이 통과했다. 별도32장 sanity는 1,500 step에서 방향 중앙값0.435°, 거리 중앙값3.625px이었다.

3회 실행에서 각각 계산한 중앙값의 평균:

| 평가 | A 방향° | B 방향° | A 거리px | B 거리px |
|---|---:|---:|---:|---:|
| 합성 test256 | 3.30 | 3.41 | 20.16 | 18.30 |
| 다른 합성128 | 4.36 | 4.31 | 25.24 | 23.82 |
| 실사 DEV52 | 5.70 | 5.98 | 36.99 | 35.89 |

합성 test의 foreground attention 질량은 29.8%→98.7%. 그러나 실사 위치의90백분위 평균은102.98px→120.36px로 악화했다. 실사 outside 그룹도 위치37.97px→41.62px, 방향7.33°→8.38°로 악화했다. 팔레트 영역 집중은 학습됐지만 실사 선 정확도는 혼합 결과이며 전이 문제를 해결했다고 판단하지 않는다.

결과 해석 정본은 결과 폴더의 `ANALYSIS.json`과 `REPORT.md`. 12개 사전 선택 프레임(합성3, 교차합성3, 실사6)에 overview12장, 선별 PNG144장, 전체 role sheet12장과 오프라인 `attention_gallery.html`을 생성하고 개별 시각 검사를 마쳤다. 실제 이미지가 첫 화면에 열린다.

## 후속 실패 원인 진단

사용자 요청에 따라 재학습 없이 오차 분해, 기존 격자의 GT oracle, attention 역할 유사도, 고정 classical 선 후보를 추가 분석했다. 정본: 결과 폴더 `diagnosis/WHY_IT_FAILS.md`. 관련 재현 스크립트는 `diagnose_metrics.py`, `diagnose_lattice.py`, `diagnose_classical.py`.

핵심: 실사 B의84.08% 역할은 위치 이동 오차가 지배적이다. 동일 격자의 GT후보 오차는1.85px이므로36px 모델 오차를 격자 간격만으로 설명할 수 없다. 저장된 실사6장·seed1에서 B의 역할간 attention cosine은0.979로 거의 동일한 위치를 참조한다. 일반 Hough/LSD에서도 물리적 선 후보는 나오지만 GT12역할에는 가려진/가상 cuboid 선이 포함된다. 백본 동결·저해상도·학습 미수렴 가능성은 통제 비교 전이므로 확인된 단일 원인으로 단정하지 않는다.
