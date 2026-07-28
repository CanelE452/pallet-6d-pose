# 4차 발표 슬라이드 골격 (skeleton)

작성: 2026-06-18 (sigma sweep 도는 중). 상태: **틀만 — 결과/데모 슬롯 비움**.
양식 무관(AIaaS 템플릿에 부어넣기용). 태그: `[재사용]` 3차 그대로 / `[갱신]` 3차 있으나 6월 방향전환 반영 / `[신규]` 4차 새 것 / `[빈슬롯]` 결과·데모 후 채움.

> ⚠ **narrative 분기 (채우기 전 확정)**: 3차는 anchor가 **v8_ablation_A_coord(이후 폐기)**. 6월에 v8 폐기 → camera-facing 0123 + paper_base 재구축 + "필터가 아니라 **base 정밀도(11px 천장)가 진짜 병목**" 발견. 4차 spine 두 안:
> - (A) **추천**: 3차 self-training 결과는 "동작 입증"으로 recap → **새 핵심 = base 천장 진단(PHASE1)+sigma로 깸(PHASE2a)** → 데모. v8→paper_base는 "convention 정정 재구축" 각주.
> - (B) 전면 재구성(base 정밀도 thesis 중심). 시간상 (A)가 안전.
> 아래 골격은 (A) 기준.

---

## 슬라이드 맵 (15~17분, ~16슬라이드)

```
#   섹션      슬라이드                                    태그        분량  figure
──────────────────────────────────────────────────────────────────────────────────
1   개요      표지 (4차)                                  [갱신]      1     -
2   개요      1·2·3차 보완 흐름 + 4차 한 문장             [갱신]      1-2   -
3   방법      시스템 전체 구조 + 데이터 흐름              [갱신]      1     재생성
4   방법      개발 환경 / 데이터셋                        [갱신]      1     -
5   방법      합성 데이터 (mixed_v8 9000 + v4_split + aug) [갱신]     1     -
6   방법      DOPE + camera-facing 0123 convention        [갱신]      1     재생성
7   방법      Geometric Filter + "2D기하 천장" 발견       [갱신]      1-2   -
8   결과      3차 recap: Self-Training R0→R1→R2 (도메인)  [재사용]    1-2   재생성×2
9   결과★     NEW 4-1: base 11px 천장 *진단* (PHASE1)     [신규]      1-2   신규
10  결과★     NEW 4-2: sigma로 천장 *깸* (PHASE2a)        [빈슬롯]    1     신규(~2h후)
11  결과      핵심 발견 (종합 메시지)                     [갱신]      2     -
12  시연★     5-1 영상 추론 데모 (6D cuboid overlay)      [빈슬롯]    1     영상
13  시연      5-2 (선택) truncation/도메인 robustness     [빈슬롯/선택] 1   몽타주
14  결론      한계 + 향후 (base v3 풀학습→필터 재가동→ADD) [갱신]     1     -
15  결론      기여 요약 (3차 대비 새 것)                  [신규]      1     -
16  Q&A       예비 (파손/가림, 처음본 파렛트, 거리추정)   [신규]      1     -
```

---

## 슬라이드별 내용 (채울 것)

### #1 표지 [갱신]
- 제목: 팔레트 6D 포즈 — 기하 제약 기반 준지도 self-training (4차/최종)
- 부제: "self-training 동작 입증 → **base 정밀도 천장 규명 + 해결**" (4차 새 메시지)

### #2 1·2·3차 보완 흐름 [갱신]
- 2차 한계: 동일 실내 도메인 / self-training 반복 미검증
- 3차 보완: outside(11seq 9894f)+night(10seq 9134f) 추가, R0→R1→R2 전 라운드, 도메인 robust 입증
- **3차가 남긴 질문(#12) → 4차 답**: "base가 핵심"이라 했으나 *왜* base가 막혔는지 미규명 → **4차 = base 11px 천장 진단+해결**
- 한 문장: "필터·self-training을 더 쥐어짜기 전에, **base corner 정밀도(real ~11px)가 천장**임을 진단으로 규명하고 sigma 레버로 깸"

### #3 시스템 전체 구조 [갱신]  · figure 재생성
- 3차 다이어그램 재사용하되 anchor를 v8 → **paper_base(camera-facing)** 로 갱신
- 합성(Isaac+Blender)→DOPE pretrain(paper_base)→R0 추론→기하필터→R1 ft. (그림: `_docs/figures/` 재생성 필요, 원본 round_curve/qualitative 부재)

### #4 개발 환경 [갱신]
- 3차 #4 표 재사용 + 정정: Pallet **KS T-11 → 실측 1100×1300×120mm**(config height 0.15→0.12), Ubuntu/Windows 양환경
- RTX 3080 10GB, conda pallet-pose, DOPE VGG-19 9 belief+16 affinity

### #5 합성 데이터 [갱신]
- paper_base 학습데이터 = mixed_v8_train 9000(Isaac2000+Blender7000) + v4_split 4000 + aug_squash/trunc/scale = 19308
- 도메인 랜덤화 + truncation 증강(L/R 측면), squash 비율증강

### #6 DOPE + convention [갱신]  · figure 재생성
- ★ v8(object-frame, 폐기) → **camera-facing 0123** (0~3 앞면, {0,1,4,5}위/{2,3,6,7}아래, 8 centroid)
- convention 정정으로 2D 기하 검증 가능(대각선 교점≈centroid)
- (근거: memory camera-facing-0123-convention)

### #7 Geometric Filter + 천장 발견 [갱신]
- 필터 재설계(RANSAC/diag/flip/loo 등) → **결론: dims-free 단안 2D기하 PL필터는 원리적 불가** (scale-skew/depth-collapse가 기하 invariant 통과). 6필터 반복확인.
- → "필터를 더 못 쥐어짠다. 천장은 **base 정밀도**" (4차 결과로 연결)
- (근거: memory dimsfree-2d-geometry-pl-filter-impossible)

### #8 3차 recap: Self-Training 결과 [재사용]  · figure 재생성×2
- 3차 outline #9 매트릭스 그대로: indoor R1 60.5%, outside R1_loo 39.5%, night R1_loo 33.3%
- 핵심: R1 단발 optimal, R2 over-iteration(confirmation bias), filter quality>양, outside-PL이 generic
- figure: round 곡선 + 정성 panel (원본 png 부재 → phase1_results.json으로 재생성)

### #9 ★NEW 4-1: base 11px 천장 *진단* [신규]
- 문제: paper_base real corner median ~11px → 어떤 필터·self-training도 그 위 못 감
- **학습 0 진단**(PHASE1)으로 천장 원인 분리:
```
종횡비(squash→aspect)  10.97→10.44  → 천장 아님(깔개만)
peak 윈도우 11→7→5     10.44→10.68  → 추출 병목 아님
입력해상도 512 forward  belief 50→64 → 해상도 레버 유효(2순위)
loss convention 검증    camera-facing 정합✓ → loss 켤 자격
→ 천장의 실체 = belief 뭉갬 = sigma (1순위 레버)
```
- 부가: edge loss λ=0.05가 belief MSE의 868% → 그냥 켜면 터짐(미리 잡음)
- (근거: memory base-11px-ceiling-phase1-diagnosis, history 2026-06-18 PHASE1)

### #10 ★NEW 4-2: sigma로 천장 *깸* [채움 — 2026-06-19]
- PHASE 2a: sigma 격리 sweep (변수 1개, v2레시피 동일·sigma만, ep60→69 finetune)
- **결론(1줄): belief Gaussian sigma 4→2로 base corner median 10.44→8.50px(−19%), 검출 손실 없이 11px 천장 돌파.**
```
model            corner_med   detect(/200)   PnP%   PCK@5
baseline s4        10.44         176           88     -
sigma3 ep69         9.90         179           90    0.979
sigma2 ep66 ★       8.50         170           85    0.992
```
- 핵심: ① sigma2가 8.5px로 천장 깸(PHASE1 "belief 뭉갬" 진단 입증) ② 검출수 유지(170≈176)→PL 가용량 손실 0 ③ sigma3는 무변=sigma2가 결정적
- 정직 단서: sigma2는 ep66 정점 후 ep69 악화 → 풀 학습 시 sweet-spot/조기정지 주의
- 메시지: "필터를 더 못 쥐어짠다(2D기하 한계) → 천장의 실체는 base 정밀도(sigma) → 학습0 진단으로 짚고 최소 레버로 돌파"

### #11 핵심 발견 (종합) [갱신]
1. Self-training R1 한 라운드가 도메인 robust (3차)
2. **단, 필터·반복의 천장 = base corner 정밀도(real ~11px)** (4차 신규)
3. 천장 원인 = belief 뭉갬(sigma), 학습 0 진단으로 규명 + sigma로 X px 개선 (4차)
4. dims-free 2D 기하필터는 원리적 불가 → base 정밀도가 정공법 (4차)

### #12 ★시연 5-1: 영상 데모 [완성 — 2026-06-19]
- **영상: `data/pallet/results/demo_4th_base.mp4`** (capture0403middle indoor, 440f, 22s, 640×480)
- paper_base_v2 + reflect-pad100, 6D cuboid wireframe(노란선) + 9 keypoint overlay, 연속 프레임 pose 추적
- 성능: det≥4 = **350/440 (79.5%)**, PnP 79.5%. 눈검증 3프레임 cuboid 정확(다각도·근접)
- 3차(정적 keypoint 이미지) → 4차(동작 영상) 진전. overlay 원본: data/pallet/results/demo_4th_base/
- (선택) 향후: sigma2_ep66로 before/after 영상이면 sigma 개선 시각화 가능

### #13 시연 5-2 (선택) robustness [빈슬롯/선택]
- truncation(모서리잘림, aug_trunc/annotate_truncation 결과) 정성 몽타주 — 기존 overlay 재활용
- ※ "파손" 데이터 없음 → "truncation/부분가림"으로 정직히. 시간 남으면.

### #14 한계 + 향후 [갱신]
- base v3 풀학습(검증된 레버=sigma[+해상도 단, 480px<512 크롭 블로커])
- 천장 깬 base로 self-training PL 재가동
- ADD/거리추정(실측 dims known, 좌표계 보정 후)

### #15 기여 요약 [신규]
- 3차 대비 새 것: ① base 천장의 정량 진단(학습0) ② sigma로 천장 돌파 ③ 영상 데모 ④ 2D기하필터 한계 규명
- "더 쥐어짜기 전에 진단으로 천장의 정체를 밝히고, 최소 레버로 돌파"

### #16 Q&A 예비 [신규]
- 파손/가림: truncation robustness(데이터 있음) + occlusion은 challenge 합성. 파손 전용 데이터는 향후.
- 처음 본 파렛트: squash 비율증강 + JSON 꼭짓점 동기화로 일반화 (논문 트랙)
- 거리추정: PnP(치수 known)에서만, 평가용 분리

---

## 채우기 작업 큐 (우선순위)
1. [~2h] sigma 결과 → #10 표·결론 (orchestration bu7dg2ibd 통지 후)
2. [오늘밤] #12 영상 데모 제작 (dope_predict_mp4_pad, 세션 택1)
3. [재생성] #3·#6·#8 figure (phase1_results.json + 파이프라인 그림 dope_pipeline.png 활용)
4. [확정] narrative 분기 (A) 확정, v8→paper_base 각주 문구
5. [정정] #4 dims, #6 convention — 6월 갱신 반영
6. [선택] #13 robustness 몽타주 (시간 남으면)

## 도구 메모
- python-pptx·marp 미설치. 실제 .pptx는 (a)`pip install python-pptx`로 생성 or (b)이 골격을 AIaaS 양식에 수기 이식.
- 3차 자료: archive/phase1_3rd_presentation.md, archive/phase1_presentation_outline.md (구조 출처).
