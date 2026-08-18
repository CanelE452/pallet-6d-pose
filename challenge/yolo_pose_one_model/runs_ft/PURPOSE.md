# PURPOSE — real finetuning (challenge 트랙)
# nano(yolo26n) 로 먼저 수행. medium(yolo26m) 도 같은 데이터·레시피를 쓴다.

[소비처] 과제 실배포 — 리프터(forklift) 장착 카메라 실시간 6D pose 추론.
         `challenge/pallet_jetson_deploy/infer_fps.py` 가 쓰는 최종 weight 를 교체한다.

[문장]   합성만으로 학습한 stage_a 에 검증 통과한 real GT 157장과 배포환경 negative
         259장을 섞어 finetuning 하면, forklift_raw 에서 리프터 포크·울타리를 팔레트로
         잡는 false positive 가 줄고 실사 도메인 검출이 개선된다.

## 판단 지표 (착수 전 고정)

1. **FP율** (주지표) — forklift_raw 의 팔레트 없는 259 프레임에서 conf>=0.4 검출이
   나오는 비율. base = 131/259 (50.6%, conf 0.05~0.2 구간이 전부 FP). 목표 대폭 감소.
2. **eval 정본 161장** — 학습에서 제외한 유일한 셋. 검출률·keypoint 정확도가
   base 대비 떨어지지 않을 것(회귀 감시용, 개선은 보너스).
3. **forklift_raw 육안** — 영상 전체에서 FP 구간이 사라졌는지.

## 데이터 (누수 규칙)

포함: capturenight01~07 88 + capturepallet02,03,04,05,08 44
      + forklift_20260528 25 = **real 157장**
      + forklift_raw negative(팔레트 없음, 빈 라벨) **259장** (conf<0.2, 육안 전수 검수 완료)
      + 합성 stage_a subsample

제외 (이유):
- **pallet11_gt 243장 — `gt_source: apriltag` 이고 GT 가 틀렸다.** manual_kps 없음,
  오버레이 12장 육안 검증에서 큐보이드가 팔레트에 맞는 프레임 **0/12**
  (`challenge/data/04_results/pallet11_apriltag_gt_check.jpg`). AprilTag→팔레트 변환 오류.
  memory 의 "manual 만 사용 결정(AprilTag 0·depth 안 씀)" 과도 일치.
  ※ 처음엔 최대 덩어리(real 의 61%)라 채택했다가, 착수 직전 검증에서 걸러냈다.
    그래서 real 이 400 → 157 로 줄었다. **실수로 빠진 게 아니라 의도적 제외다.**
- eval 정본 161장 — 평가 전용. CLAUDE.md 최상단 규칙.
- eval_canonical 내 non-eval 53장 — 같은 세션에 eval 프레임 존재 → 인접 프레임 누수.
- capturepallet07_augmented 275장 — SEALED final-test 파생 의심(stem 재번호로 추적 불가)
  + [-1,-1] sentinel 744개로 코너 정보 손실.
- _night_eval_manual_gt 43장 — night05/06/07 과 중복(같은 프레임 2배 가중 방지).
- wood 45장 — 나무 팔레트, 배포 대상(플라스틱)과 다른 물체.
- pseudo_gt 38장 — pseudo label, 품질 미보장.

## 근거

FP 원인 진단: `_docs/history/2026-08-16.md` "forklift_raw FP 진단".
학습셋 73,916장에 negative 0장 → 모델이 "팔레트 없음" 을 본 적이 없다.
