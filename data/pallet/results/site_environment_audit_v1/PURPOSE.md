# PURPOSE — site environment audit v1

[소비처]
논문의 새 절 "post-hoc site-matched adaptation analysis" 의 데이터 계약.
기존 MAIN self-training(Day500+Night500 joint, R0~R5)은 건드리지 않고,
그 실패를 설명할 대안 설정이 **애초에 가능한지**를 판정하는 데 쓴다.
가능하다면 후속 실험은 namespace `paper_selftrain_site_v1/` 로 분리한다.

[문장]
"이 저장소의 real 촬영본에는 같은 물리 장소를 서로 다른 recording 으로 찍은
세션 쌍이 N 개 있으며, 그 쌍은 source recording 과 이미지 SHA 가 서로소라
adaptation/evaluation 으로 leakage 없이 나눌 수 있다" — 또는 그 반증.

## 판단 지표 (결과를 보기 전에 고정)

READY 로 세려면 다섯 개를 모두 만족해야 한다.

```
1  같은 site 임이 사람 확인으로 확정됐다 (자동 grouping 만으로는 불가)
2  adapt source recording ∩ eval source recording = 공집합
3  adapt image SHA ∩ eval image SHA = 공집합
4  adaptation 이미지가 실제로 존재한다
5  evaluation 이미지가 실제로 존재한다
```

논문용 강한 조건은 따로 표시한다: adapt recording >= 2 AND eval recording >= 2.

## 이번 단계에서 하지 않는 것

새 학습 · pseudo-label 생성 · teacher inference · student training ·
모델 성능 평가 · 기존 split 수정 · 기존 MAIN pseudo pool 재작성.
이번 실행의 끝은 site 분류 후보와 READY 개수 보고까지다.

## 사용 금지 신호 (결과 기반 clustering 차단)

GT keypoint 정확도 · GT pose · model error · R0/R5 결과 · IoU3D · ADD ·
AUROC · pseudo-label purity 를 site 판정에 넣지 않는다.
"R5 가 좋아지는 세션끼리 같은 site" 같은 grouping 은 순환이라 금지.
site 판정에 쓸 수 있는 것은 RGB · capture provenance · session metadata ·
timestamp · camera metadata 뿐이다.

## 자동화의 한계 (사전 선언)

자동 단계는 **후보 생성까지**다. `PROPOSED_SITE_GROUPS.json` 은 제안이고
`SITE_GROUP_LOCK.json` 은 사람 확인 뒤에만 만든다.
AUTO_GROUPING_IS_FINAL = NO.
