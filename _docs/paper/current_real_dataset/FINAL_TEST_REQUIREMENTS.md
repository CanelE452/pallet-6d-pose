# Final Test Requirements

작성 2026-08-27.

```
FINAL_TEST_STATUS = NOT_AVAILABLE
```

## 왜 현재 128 / 140 은 final test 가 아닌가

1. **매니페스트 자신이 그렇게 선언한다.**
   `REVIEWED_CLEAN_REALDEV_V2_MANIFEST.json` 의 `role` 필드가
   `"DEVELOPMENT — final test 아님"` 이고 `status` 는 `"CANDIDATE_PRIMARY_REAL_EVAL"` 이다.

2. **실제로 selection 에 반복 사용됐다.** architecture 비교, negative supervision
   스크린, hard-negative 스크린, conf threshold 탐색, model_compare 등에서 이 셋의
   수치를 보고 다음 실험을 정했다. 같은 셋에서 최종 수치를 보고하면 selection 에 쓴
   셋에서의 성능을 보고하는 것이 된다.

3. **session-level separation 이 보장되지 않는다.** `eval_outside` 22장이 다른 5개
   capture 세션에서 뽑아 모은 셋임이 뒤늦게 드러났고, 나머지 6개 set 의 세션 독립성은
   **디렉토리명 외의 근거가 없다**(capture metadata 부재).

## final test 가 갖춰야 할 조건

```
□ 새 frame        model / data / loss selection 에 한 번도 사용되지 않은 프레임
□ 새 capture      가능하면 새로 촬영한 세션.  기존 세션에서 잘라내면
                  eval_outside 와 같은 사고가 반복된다
□ session 분리    frame 단위가 아니라 **세션 단위**로 분리.
                  세션 동일성을 디렉토리명이 아니라 **capture metadata 로** 검증할 것
                  (현재 라벨에는 capture 일시·장비 metadata 가 없다 — 새 셋에는 넣는다)
□ GT-QA 완료      판정 사유를 남긴다.  현 GT-QA 는 reviewed_gt/fixed_gt/logs 가
                  0 files 라 "왜 21장을 뺐는가" 를 재구성할 수 없다
□ overlap 0       FT / adaptation / self-training 학습 셋과 교집합 0
                  ★디렉토리 비교로 끝내지 말 것 — frame 단위 해시로 검증
□ duplicate 0     동일 해시 0 + pHash near-duplicate 점검
                  (현 negative 셋에 동일 해시 1쌍이 있다)
□ freeze 시점     ★metric 과 threshold 를 정하기 **전에** membership 을 동결하고
                  sha256 을 기록한다.  동결 후에는 어떤 사유로도 추가·제외하지 않는다
```

## 추가로 권고 (현 셋의 결함에서 도출)

```
□ per-keypoint 가시성 flag 를 라벨에 넣는다
     현 셋은 object-level 스칼라 visibility 만 있고 140/140 전부 값 1 이다.
     가림 조건별 분석과 "보이는 점만 평가" 프로토콜이 불가능하다.

□ dimensions_m 규약을 하나로 고정한다
     현 셋은 (1.1,0.11,1.3) 81장 / (1.3,0.11,1.1) 59장 두 변종이다.
     프레임별 W/D 스왑이 평가에 GT 정보를 흘린다
     (memory `evaluator-receives-gt-per-frame-axis-assignment`).

□ NIGHT 비중을 늘린다
     현 셋 NIGHT 28장(20%). 야간이 실패 레짐인데 표본이 작아 CI 가 넓다.

□ 가림·잘림 프레임을 의도적으로 포함하고 라벨한다
     현 셋은 occlusion/truncation 이 라벨돼 있지 않아 커버리지를 주장할 수 없다.
```

## 동결 절차 (제안)

```
1. capture           새 세션 촬영.  일시·장비·조명 metadata 를 프레임마다 기록
2. annotate          camera_dynamic_0123_v4 + per-kp 가시성 flag + 고정 dims 규약
3. GT-QA             판정 사유를 파일로 남긴다
4. overlap audit     기존 전 학습셋에 대해 frame 해시 교집합 0 확인
5. duplicate audit   내부 동일 해시 0 + pHash 근접중복 점검
6. FREEZE            membership 목록 + sha256 을 커밋.  이 시점 이후 metric/threshold 결정
7. 평가              main table 5지표만 (README 참조)
```

동결 전까지 논문 최종 수치는 **보고하지 않는다.** 그때까지의 모든 real 수치는
development 결과로 표기한다.
