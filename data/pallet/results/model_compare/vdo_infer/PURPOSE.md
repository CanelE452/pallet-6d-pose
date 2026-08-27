# 논문 트랙 YOLO — 배포영상 4프레임 — PURPOSE

**[소비처]** 방향결정 — 같은 학습 데이터(BROAD 합성 40,000, real 0장)에서
**아키텍처만 다른 두 모델**이 적용범위 밖 영상에서 어떻게 갈리는지 본다.
FINAL40K(SPLIT_LATE, corner+line) 판본은 이미
`paper_s2_multihead/final_train/vdo_infer/` 에 있고, 이 폴더는 그 짝이다.

**[문장]** "정본 161장에서 YOLO 논문판이 116/161, FINAL40K 가 그와 다른 검출
프로파일을 보인 차이가, 학습 분포 밖(창고 실내·원거리·1280x720)에서도 같은
방향으로 나타나는가" — 같은 방향이면 아키텍처 차이고, 뒤집히면 정본 결과가
분포 안에서만 성립하는 것이다.

## 학습 아님

추론 + 그림만. 가중치를 만들지 않는다.
```
yolo26n_paper_generic_v1  runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt  60 epoch
yolo26n_broad40k_5ep      runs_broad40k/b_yolo26n_broad40k_5ep/weights/last.pt          5 epoch
```
둘 다 `datasets/broad40k/data.yaml` 로만 학습됐다. challenge 트랙의 `*_ft`
(real 파인튜닝)는 **넣지 않는다** — 섞으면 "같은 데이터, 다른 아키텍처" 라는
비교가 깨진다.

## 공정성을 위해 맞춘 것 / 못 맞춘 것

```
맞춤    K        internet_pallet_infer.fit_K (HFOV 스윕), FINAL40K 와 동일
맞춤    치수      ASSUMED PALLET_DIMS 1.1 x 1.3 x 0.11 m, FINAL40K 와 동일
맞춤    그림      ft_vdo_infer.draw / 색 / 엣지 그대로 재사용
못 맞춤  pose 팔  FINAL40K = F3 (line 이 회전 재적합) / YOLO = 점 PnP (line 없음)
못 맞춤  det 정의 DOPE = belief threshold 0.3 / YOLO = 박스 잡히면 9점 항상 출력
                 → YOLO 는 keypoint conf >= 0.5 로 세고 헤더에 기준을 박는다
```

## 읽을 때의 단서

**표본 4프레임이다.** 우열 판단 근거로 쓸 수 없다. cam_K 없음·치수 가정도
그대로 남아 있어 거리(m)와 reproj 는 그 가정 위의 값이다.
