# hough_local_evidence_v1 — 국소 증거가 Hough 일반화의 병목인가

## 1. 제안

이 세션에서 확정된 사실 위에 선다.
```
능력 있음      OVERFIT32 (32장 암기) angle 0.60도   게이트 1.0도 통과
자기 dev       4.43도  (재현 검증 완료, 상대차 0.03%)
분포 밖        15~48도 (7 모집단), 균일분포 45도 수준
배제됨         전처리 오류 · 측정 배선 · sim2real · descriptor 용량 · 기록값 신뢰성
```
가설: `role @ f(theta,rho)` 에서 이미지가 **전역 벡터 하나**로만 들어오고 선분을 따라
픽셀을 확인하는 경로가 없어서, 학습 분포에서는 외우고 밖에서는 못 맞힌다.

## 2. 설계 (결과 보기 전 고정)

```
공통   데이터   mixed_v8_train 9,000 (렌더 단위 train/dev 분리)
       backbone FrozenA1 동결
       step     3,000 · seed 1 · 측정은 정본 batch_rows/decode/measure/summarise
H0     role @ f(theta,rho)                       현행 구조 재현
H1     H0 + SupportingLineHead raster + CoarseRadon 누적   국소 증거 경로
```
★**단일 변수가 아니다** — CoarseRadon 이 확률 raster 를 요구해 head 를 함께 붙인다.
따라서 "누적 경로" 가 아니라 **"국소 증거 경로 통째로"** 를 넣는 실험이다.

전제는 실행으로 확인됐다 — CoarseRadon 미분 가능(grad 30,000/30,000),
정본 로더가 mixed_v8_train 을 읽음(load_frame/load_pack/gt_lines).

## 3. 판정 (사전등록)

```
1차 관문  H0 가 자기 dev 에서 angle median 3~6도.  실패 시 STOP —
          파이프라인 재구축 실패이지 가설 검증이 아니다.
2차 관문  교차셋(v4_split_base, aug_squash_v2)의 angle median 에서  [추정][미검증]
          H1 이 H0 대비 30% 이상 개선 -> LOCAL_EVIDENCE_HELPS
          10~30%                      -> WEAK
          10% 미만 또는 악화           -> LOCAL_EVIDENCE_NOT_THE_CAUSE (계열 종료)
```
교차셋을 primary 로 두는 이유: 자기 dev 에서는 H0 도 4.43도라 **거기서는 안 갈린다.**

예상 실패 모드: (a) H0 재현 실패 (b) H1 이 메모리/속도로 안 돎 (c) 차이 없음.
중단 기준: (a) 면 즉시 STOP. (c) 면 계열 종료하고 재시도하지 않는다.

## 4. 결과
(진행 중)
