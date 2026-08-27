# PAPER GENERIC — PURPOSE

**[소비처]** 논문 main model. "target-free generic synthetic 만으로 unseen real
pallet 6D pose 가 되는가" 의 답이 그대로 논문의 주 실험이 된다.

**[문장]** 평가 대상 팔레트의 OBJ·target-specific synthetic·real positive 를
하나도 쓰지 않고 BROAD generic synthetic 만 학습한 YOLO26n point-only 파이프라인이
unseen real 팔레트에서 동작한다면, 논문의 unseen-object 주장이 성립한다.

## 판정 지표 (결과 보기 전 고정)

```
STRONG_PASS   challenge105 native availability >= 0.75
              AND canonical R median <= 5 deg
              AND canonical 5cm5 >= 0.20
              AND open56 5cm5 >= 0.45
WEAK_PASS     challenge105 5cm5 >= 0.10 AND R median <= 8 deg
              AND 5epoch 진단본보다 명확히 개선
FAIL          challenge105 5cm5 < 0.10 OR availability < 0.50
              OR generic set(outside/noapril/cad)까지 광범위 붕괴
```

## 하지 않는 것

Direct-Hough / F3 / CIGM 미사용. target OBJ·alias·target synthetic·real positive
미사용. negative 는 이번 main run 에서 제외. best.pt 미사용 — last.pt @ epoch 60.

## 사전 감사

`data_audit/TARGET_ASSET_EXCLUSION_AUDIT.md` = **PASS**.
누수는 없으나 두께비 coverage 구멍이 확인됐다 (target 0.0923, BROAD 에서 그보다
얇은 프레임 0.29%). 결과 해석 시 이 사실을 먼저 놓고 본다.

## 초기화 표기

COCO-pose pretrained 에서 출발한다. 논문에는 **"generic pallet synthetic only
after COCO pose pretraining"** 이라고 적는다. "완전 synthetic-only from scratch"
라고 쓰지 않는다.
