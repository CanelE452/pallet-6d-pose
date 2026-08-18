# PPD ARCHITECTURE DECISION — learned polarity head REJECT (real transfer)

> candidate set 은 oracle unsigned SAI-U 로 **고정**했고, learned map 은 **re-ranking 만** 했다.
> predicted line 기반 candidate generation 은 아직 검증하지 않았다.
> training 은 clean full-view **V8-only** synthetic(`paper_4pallet_mask_v1`) 단독.
> validation 은 group-disjoint, untouched 5,916 은 checkpoint selection 과 독립,
> N87 은 mechanism-val **one-shot**, final-test 미사용.
> historical H1(mask)/H2(pixel line) FAIL 은 그대로 유지하고,
> H3(candidate polarity)가 direct objective 이므로 long-run 을 진행했다.

## [관찰]

```
단계         L0                M0                M1
overfit(32)  0.969 (1 inv)     1.000 (0)         1.000 (0)
validation   0.980 (15/754)    0.980 (15/754)    0.977 (17/754)   전 arm PASS
untouched    0.951 (183/3716)  0.950 FAIL(경계)   0.952 (180/3716)
real N87     0.023 (84/86)     — (미실행)         0.012 (85/86)    전 arm FAIL
```

## [Process recovery]

worker(PID 562885)는 `[done]` 으로 정상 종료.  5916/5916 완주 후 real 까지 실행.
run_state 는 L0/M0/M1 모두 `completed: true`, epoch 20.

## [OOM and chunk integrity]

- [확인] 첫 untouched 실행은 5,916 프레임을 한 번에 적재해 **OOM Killed**.
- [확인] 두 번째 실행은 **400-frame chunk** 로 재실행.  metric 정의·threshold·candidate set 은
  바뀌지 않았고 **메모리 상주 방식만** 변경했다.
- [확인] OOM attempt 는 산출물을 남기지 못했다(`ppd_untouched_metrics.json` 은 chunked 실행이
  08:34:37 에 생성한 단일 파일).  partial merge 위험 없음.
- [확인] chunk 결과는 `_merge()` 로 가중 병합.  n_frames 5916, candidate_pair 3716,
  세 arm 모두 동일 수.

## [Untouched synthetic]

availability 3716/5916 = **0.628** (validation 0.721 대비 하락).
conditional polarity 0.950~0.952, indexed reproj 98.14 → 1.15~1.16 px.
L0/M1 PASS, M0 는 0.9497 로 **0.0003 미달** FAIL(경계).

## [Real N87]

전 arm gate FAIL.  다만 실패의 성격이 중요하다.

**0.023 은 성능 저하가 아니라 반-상관이다** (우연이면 0.5).

결정적 실험 — 동일 N87·동일 candidate 에서
**oracle 5-class map + long-run scorer = 86/86 (1.000)**.
따라서 scorer·candidate·reference pose·좌표 규약은 정상이고,
**learned map 자체가 real 에서 틀린다**.

원인은 top↔base 스왑이 아니라 **base 로의 붕괴**:

```
GT top_width 위치 → pred base 0.683 vs top 0.206
GT top_depth 위치 → pred base 0.781 vs top 0.146
predicted 면적: base 0.242~0.295  >  top 0.183~0.187
```

[추정] target 클래스 불균형(positive-frame rate base 0.975~0.985 vs top 0.620~0.685)이
synthetic 에서는 가려졌다가, top 근거가 약한 real(94% 저앙각 edge-on)에서 발현된 것으로 보인다.

## [Visual evidence]

오버레이 결과 real 에서 예측 map 이 **팔레트가 아니라 배경 전체**에 반응한다.
positive 중 팔레트 위 비율 synthetic 0.456 vs real 0.086, 면적보정 enrichment
4.35x vs 2.14x.  즉 real 실패의 1차 증상은 polarity 혼동이 아니라 **localization 소실**이다.
(`figures/overlays/`, `ppd_on_object_activation.json`)

## [Mask ablation]

mask 는 **pixel/mask 지표만** 올리고(macroF1 0.311→0.409, IoU 0.097→0.746)
목적 지표는 개선하지 않는다(real 은 L0 0.023 > M1 0.012 로 오히려 L0 우세).

## [CGR]

**NOT RUN** — real gate 를 통과한 arm 이 없다.  조건부 규칙을 따랐다.

## [지지 증거]

- [확인] validation·untouched 는 group-disjoint 이며 checkpoint 는 validation 으로만 선택.
- [확인] 학습이 32-frame 암기가 아니다: 미학습 3,716 candidate-pair 프레임에서 0.951~0.952.
- [확인] 평가 경로 무결성은 oracle map 대조 실험(86/86)으로 입증.

## [반증 증거]

- [확인] synthetic 0.95 가 real 0.02 로 **역전**된다.  clean V8-only 학습 분포와
  real 저앙각 분포의 간극이 결정적이다.
- [확인] untouched availability 가 validation 대비 0.721 → 0.628 로 떨어진다 =
  selection 이전에 **upstream SAI availability** 도 취약하다.

## [현재 판정]

learned 5-class polarity head 는 **synthetic 안에서는 일반화하지만 real 로 전이되지 않는다**.
oracle 표현이 real 에서 3/86 을 달성했으므로 **표현의 정보량이 부족한 것이 아니라
학습된 예측기가 real 에서 무너지는 것**이다.

## Architecture decision

```
SAI-U candidate generation      ACCEPT
5-class polarity representation ACCEPT (oracle upper bound: real 3/86, 86/86 재현)
L0                              REJECT (real transfer 실패)
M0                              DROP
M1                              DROP
Learned PPD                     REJECT (현 학습 데이터·target 조건에서)
CGR                             NOT RUN (real gate 미통과)
Point uncertainty               DEFERRED
```

## [다음 admissible experiment]

1. **target 클래스 불균형 교정** — top/base positive-frame rate 를 맞추거나
   base-우세 prior 를 억제하는 class weight/sampling.  이번엔 결과를 보고 바꾸지 않았다.
2. **저앙각 학습 분포 확보** — `paper_4pallet_mask_v1` 은 V8 full-view 100%,
   truncation 0% 다.  real 의 94% 저앙각 edge-on 이 학습 분포에 없다.
3. upstream SAI availability(0.628) 개선 — selection 이전 단계의 병목.
4. 위 1~2 없이 재학습·threshold 조정은 하지 않는다.
