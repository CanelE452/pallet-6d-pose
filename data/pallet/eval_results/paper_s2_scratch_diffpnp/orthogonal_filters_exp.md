# 직교(orthogonal) 신호 필터 실험 — Stage B on filterval(123)

- weights: `weights/paper_s2_stageB/net_epoch_0057.pth` (squash-parity)
- set: filterval = outside44 + night43 + manual36 = 123 (real, low-angle)
- 목적: DiffPnP 와 순환 안 하는 비-기하 신호가 rear-collapse / half=whole 를 잡는가.
- signals: (1)photometric TTA front/rear/worst + rear-dropout  (2)heatmap spread/entropy  (3)ensemble(s1) disagreement
- GT-good=corner_med<10px, GT-gross=>20px. rear_err=back(index-aligned Hungarian, GT rear index).
- ★f4_tta_stab_ref = 기존 f4(8코너 평균 scalar) 대조군.

## 집합 요약
```
set           N  det  GT-good
overall     123   94       26
outside      44   30       12
night        43   29        3
manual       36   35       11
```

## Spearman(signal, GT오차) — 양수=신호↑일수록 오차↑ (좋은 필터=강한 양상관)
낮은 신호=신뢰. reject 규칙 = signal > tau. |rho| 클수록 분리력.

### outside+manual (base 품질 有)  (n_det=65)
```
signal              rho vs overall   (n)   rho vs REAR   (n)
------------------------------------------------------------
photo_front                  +0.34    65         +0.35    65
photo_rear                   +0.18    65         +0.24    65
photo_worst                  +0.25    65         +0.32    65
drop_rear                    +0.26    65         +0.14    65
spread_front                 +0.55    65         +0.38    65
spread_rear                  +0.24    65         +0.23    65
spread_worst                 +0.33    65         +0.21    65
ent_rear                     +0.27    65         +0.28    65
ens_front                    +0.04    59         +0.07    59
ens_rear                     +0.24    60         +0.31    60
ens_worst                    +0.25    60         +0.30    60
f4_tta_stab_ref              +0.28    65         +0.27    65
```

### outside  (n_det=30)
```
signal              rho vs overall   (n)   rho vs REAR   (n)
------------------------------------------------------------
photo_front                  +0.43    30         +0.41    30
photo_rear                   +0.40    30         +0.34    30
photo_worst                  +0.37    30         +0.29    30
drop_rear                    +0.65    30         +0.61    30
spread_front                 +0.69    30         +0.58    30
spread_rear                  +0.29    30         +0.31    30
spread_worst                 +0.43    30         +0.40    30
ent_rear                     +0.35    30         +0.36    30
ens_front                    +0.09    30         +0.05    30
ens_rear                     +0.38    30         +0.41    30
ens_worst                    +0.40    30         +0.37    30
f4_tta_stab_ref              +0.44    30         +0.39    30
```

### manual  (n_det=35)
```
signal              rho vs overall   (n)   rho vs REAR   (n)
------------------------------------------------------------
photo_front                  +0.24    35         +0.29    35
photo_rear                   -0.08    35         +0.01    35
photo_worst                  +0.11    35         +0.18    35
drop_rear                    -0.08    35         -0.22    35
spread_front                 +0.51    35         +0.34    35
spread_rear                  +0.11    35         +0.26    35
spread_worst                 +0.23    35         +0.15    35
ent_rear                     +0.14    35         +0.34    35
ens_front                    -0.13    29         +0.01    29
ens_rear                     +0.11    30         +0.18    30
ens_worst                    +0.15    30         +0.23    30
f4_tta_stab_ref              +0.10    35         +0.09    35
```

### night (base 붕괴, 참고만)  (n_det=29)
```
signal              rho vs overall   (n)   rho vs REAR   (n)
------------------------------------------------------------
photo_front                  +0.16    29         +0.11    29
photo_rear                   +0.23    29         +0.35    29
photo_worst                  +0.30    29         +0.27    29
drop_rear                    +0.20    29         +0.38    29
spread_front                 +0.09    29         +0.18    29
spread_rear                  +0.32    29         +0.36    29
spread_worst                 +0.24    29         +0.43    29
ent_rear                     +0.26    29         +0.37    29
ens_front                    +0.36    27         +0.25    27
ens_rear                     +0.20    27         +0.47    27
ens_worst                    +0.34    28         +0.47    28
f4_tta_stab_ref              +0.15    29         +0.17    29
```

### overall  (n_det=94)
```
signal              rho vs overall   (n)   rho vs REAR   (n)
------------------------------------------------------------
photo_front                  +0.31    94         +0.33    94
photo_rear                   +0.27    94         +0.35    94
photo_worst                  +0.31    94         +0.36    94
drop_rear                    +0.26    94         +0.25    94
spread_front                 +0.44    94         +0.37    94
spread_rear                  +0.21    94         +0.25    94
spread_worst                 +0.27    94         +0.30    94
ent_rear                     +0.25    94         +0.30    94
ens_front                    +0.18    86         +0.19    86
ens_rear                     +0.28    87         +0.41    87
ens_worst                    +0.31    88         +0.39    88
f4_tta_stab_ref              +0.28    94         +0.30    94
```

## ★ 핵심: 배포필터 통과했지만 GT-나쁨(=confidently-wrong) 프레임 잡기
배포필터 = f1&f2&f4&f5&f6&f7. 통과 중 GT-bad(corner_med>=10px) 를 각 rear-signal 이 상위 threshold(통과군 75-percentile)로 몇 개 flag 하는가.
이상적 = accepted-bad 는 많이 잡고, accepted-good 는 적게 버림.
```
배포필터 통과: 86  (good=23, bad=63)
rear-signal      tau(good p75)    bad잡음    good버림   net(잡-버)
--------------------------------------------------------------
photo_rear                5.06    29/63      6/23        +23
photo_worst               7.36    32/63      6/23        +26
spread_rear              30.65    25/63      6/23        +19
spread_worst             31.22    35/63      6/23        +29
ens_rear                 21.03    21/63      5/23        +16
ens_worst                27.12    26/63      5/23        +21
drop_rear                 0.00    16/63      2/23        +14
f4_tta_stab_ref           0.87    42/63      6/23        +36
```
(accepted-bad fids: outside/1778651518865604096(cm37,rear48), outside/1778651530557153024(cm13,rear13), outside/1778651534387246080(cm65,rear68), outside/1778651569891693056(cm57,rear60), outside/1778651577517179648(cm22,rear22), outside/1778651579432408064(cm84,rear95), outside/1778651583329453568(cm11,rear14), outside/1778651587361112320(cm62,rear68), outside/1778651651444080384(cm73,rear82), outside/1778653364045039360(cm19,rear27), outside/1778653390451955456(cm24,rear24), outside/1778653424183454208(cm11,rear16), outside/1778653458082420992(cm15,rear21), outside/1778653522084302848(cm50,rear82), outside/1778653524402450688(cm33,rear33), outside/1778653526754124544(cm10,rear15), outside/1778653554639401216(cm18,rear19), night/1779449171575465984(cm21,rear37), night/1779449173709591552(cm18,rear46), night/1779449178279663872(cm32,rear69), night/1779449180481178880(cm19,rear43), night/1779449182916275200(cm23,rear41), night/1779449191722656768(cm25,rear25), night/1779449194023912448(cm18,rear18), night/1779449196392532480(cm13,rear14), night/1779449231735721216(cm14,rear14), night/1779449236238168576(cm11,rear23), night/1779449240774612992(cm15,rear28), night/1779449245478053376(cm14,rear21), night/1779449247779662080(cm37,rear89), night/1779449254651112192(cm47,rear47), night/1779449256919425024(cm38,rear82), night/1779449263958007040(cm51,rear230), night/1779449266426633216(cm27,rear52), night/1779449302793770240(cm16,rear17), night/1779449305228291328(cm13,rear13), night/1779449309498034944(cm16,rear23), night/1779449314101526272(cm15,rear16), night/1779449316336256768(cm18,rear11), night/1779449318604401920(cm14,rear29), night/1779449327510928384(cm13,rear41), manual/1778654525414013952(cm11,rear8), manual/1778654528572052480(cm14,rear11), manual/1778654529143344384(cm17,rear25), manual/1778654533141360384(cm11,rear23), manual/1778654533578182400(cm11,rear27), manual/1778654533712330240(cm29,rear29), manual/1778654534619446016(cm11,rear25), manual/1778654534652969472(cm10,rear10), manual/1778654535896213248(cm10,rear15), manual/1778654535996968192(cm16,rear16), manual/1778654536232369664(cm12,rear15), manual/1778654538046514944(cm50,rear50), manual/1778654538483679232(cm14,rear14), manual/1778654539356538112(cm12,rear8), manual/1778654540767799296(cm45,rear35), manual/1778654544900225024(cm17,rear24), manual/1778654548864587008(cm50,rear50), manual/1778654548998988032(cm46,rear46), manual/1778654549805509888(cm11,rear11), manual/1778654550208514304(cm10,rear11), manual/1778654552425688832(cm14,rear14), manual/1778654553265639936(cm11,rear18))

## per-frame (detected, outside+manual, corner_med 순)
```
dom     fid             cm rear  ph_f  ph_r  sp_r ens_r   f4 acc good
---------------------------------------------------------------------
outside 177865342720   3.7  7.6   1.5   1.9  29.5  18.9  0.4   Y    Y
outside 177865356011   4.8 11.5   0.8   1.2  29.5  10.8  0.3   Y    Y
outside 177865334546   5.0 17.0   1.0   2.7  30.3  16.1  0.3   Y    Y
outside 177865345593   5.2 11.8   1.1   4.8  29.8  10.0  0.8   Y    Y
manual  177865450340   5.4  5.2   0.8   2.2  29.5    -   0.4   Y    Y
outside 177865152857   5.6 13.4   1.3   3.6  29.7   4.4  0.4   Y    Y
outside 177865353978   6.0  6.8   0.7   8.7  31.3   5.7  2.7   Y    Y
outside 177865355729   6.3 14.6   1.7   3.5  29.7   8.0  0.6   Y    Y
outside 177865342969   6.6 13.5   0.4   3.5  29.5  30.6  1.0   Y    Y
outside 177865151678   6.6 12.8   0.8   2.2  29.0   6.6  0.3   Y    Y
manual  177865449746   7.3 11.6   4.1   3.3  29.5    -   0.7   Y    Y
outside 177865333777   7.5 13.9  17.3   8.3  30.3  12.2  0.7   Y    Y
manual  177865449776   7.6  4.9   5.1   3.2  30.4    -   1.7   n    Y
manual  177865450357   7.8  7.4   1.5   2.8  29.2    -   0.5   Y    Y
outside 177865158534   7.9 15.3  38.0  12.9  29.9  15.7  7.2   n    Y
manual  177865453408   8.4  9.2   1.3   3.6  30.8   4.8  0.6   Y    Y
outside 177865349843   8.9 20.5   5.6  12.3  30.8  23.1  4.3   Y    Y
manual  177865455225   9.2 30.3   1.6   4.0  30.5  50.4  0.8   Y    Y
manual  177865453539   9.3 10.5   1.1   2.1  31.0   6.2  0.4   Y    Y
manual  177865453582   9.6  9.6   0.9   1.8  30.3   8.2  0.4   Y    Y
manual  177865451489   9.8  9.5   2.2   1.1  29.3    -   0.3   Y    Y
manual  177865453562   9.9 10.0   1.1   2.0  30.9   8.6  0.4   Y    Y
manual  177865455202   9.9 37.3   1.3   7.7  30.8  28.6  0.9   Y    Y
manual  177865453589  10.2 14.9   1.4   4.8  31.2   4.3  0.7   Y    n
manual  177865453465  10.2 10.3   2.3   3.2  31.9   3.1  1.2   Y    n
outside 177865352675  10.3 15.0   2.0   2.6  29.9   8.3  1.6   Y    n
manual  177865455020  10.5 11.5   1.3   2.4  30.1  15.1  0.4   Y    n
manual  177865453357  10.5 26.7   4.2   3.3  31.5   7.6  1.8   Y    n
manual  177865453314  10.6 23.2   9.2   4.9  31.6   7.4  4.7   Y    n
manual  177865455326  10.8 18.1   3.0   5.1  32.0  20.8  0.8   Y    n
manual  177865454980  10.8 10.6   1.9   2.8  30.6   9.4  0.8   Y    n
manual  177865452541  10.9  8.3   1.9  11.3  29.8  12.2  2.7   Y    n
outside 177865342418  11.1 15.8   0.8   2.4  29.1  13.9  0.5   Y    n
manual  177865453461  11.4 24.7   2.3   3.1  32.0   3.2  0.9   Y    n
outside 177865158332  11.4 14.2   6.2  13.4  30.0   6.7  2.9   Y    n
manual  177865453623  11.6 15.2   2.0   2.2  30.5  11.7  0.6   Y    n
manual  177865453935  11.9  8.0   1.2   2.5  29.9   9.9  0.4   Y    n
manual  177865454913  12.4 18.1  54.6  21.2  30.7  18.2 12.2   n    n
outside 177865153055  13.5 13.5   1.1   2.2  29.4  12.0  0.3   Y    n
manual  177865455242  13.5 13.5  11.2  15.3  30.5  44.3  3.8   Y    n
manual  177865453848  13.6 14.2   1.4   5.1  31.7  11.3  1.2   Y    n
manual  177865453811  13.8 18.9  43.3   2.3  30.6   7.1  9.9   n    n
manual  177865452857  14.3 11.0   1.0   3.0  30.8  16.1  1.2   Y    n
outside 177865345808  15.2 20.6   2.3   3.0  29.6  18.9  0.7   Y    n
manual  177865453599  16.1 16.4   5.0   4.0  31.0   6.1  1.2   Y    n
manual  177865452914  17.1 25.3   5.5   3.7  31.1  12.8  1.5   Y    n
manual  177865454490  17.2 23.8   1.4   6.0  30.7  20.1  1.0   Y    n
outside 177865355463  18.4 19.2   2.9   8.7  31.2  27.1  1.2   Y    n
outside 177865336404  19.1 27.5   1.7   6.6  31.5   5.1  1.6   Y    n
outside 177865157751  22.0 22.0   3.2   6.5  29.4  23.5  2.1   Y    n
outside 177865339045  24.1 24.1   3.1  13.2  31.0   7.9  1.9   Y    n
manual  177865453371  29.3 29.3   3.8   1.8  30.7   6.9  0.6   Y    n
outside 177865352440  33.0 33.0   0.6   8.4  29.3  57.9  1.3   Y    n
outside 177865151886  37.3 48.0   6.5   2.6  29.5   4.4  0.9   Y    n
manual  177865454076  45.0 35.2   1.8   1.9  29.8  22.6  0.3   Y    n
manual  177865454899  46.0 46.0   1.7   1.9  30.3  17.7  0.4   Y    n
outside 177865352208  49.6 81.7   0.9   2.0  29.3   7.7  0.6   Y    n
manual  177865454886  50.2 50.2   2.5   1.9  30.3   6.8  0.4   Y    n
manual  177865453804  50.2 50.2   1.2   1.3  30.0   6.2  0.4   Y    n
outside 177865156989  56.6 59.7   2.9   8.4  31.6  24.1  2.2   Y    n
outside 177865157190  57.1 61.4  12.9  32.3  30.4  60.8 11.1   n    n
outside 177865158736  62.4 67.7   1.8   7.8  31.0  26.0  1.8   Y    n
outside 177865153438  64.7 67.8   5.8   8.6  31.5  13.1  0.9   Y    n
outside 177865165144  73.0 81.7   2.3   5.2  30.1  78.0  1.0   Y    n
outside 177865157943  84.3 95.3   4.5  10.2  31.1 114.5  2.5   Y    n
```

★ caveat: real low-angle domain-mixed, heuristic tau, 소표본(도메인별 accepted-bad 특히 적음). ensemble-s1 = squash OOD 가능(domain-shift confound). night = base GT-good 붕괴라 어떤 신호도 순도 못 냄(참고만).

## 결론 (2026-07-10) — 직교 신호 3계열 검증

**검증셋**: filterval 123 (det 94), 판정은 base 품질 있는 **outside+manual**(accepted 60,
good 21/bad 39, base precision 0.35). night(GT-good 3/29)은 "이미지가 어둡다↔pose 나쁨"
confound라 신호가 자연히 높아짐 → 참고만.

### 1) rear-specific photometric TTA (사용자 #1) = 미확정/약함
- photo_rear precision 0.35→0.44 (bad 19/39), manual 도메인 rho ≈ 0 (-0.08~+0.01).
- ★기전: **collapse된 rear는 confidently-wrong일 뿐 아니라 stably-wrong** — gamma/WB/noise/
  JPEG/blur 교란에도 안 흔들림. 모델이 rear를 일관되게 hallucinate하기 때문. flip-consistent
  실패(f3)·confidently-wrong(stage19)와 같은 벽. "front 안정/rear만 흔들림" 가설은 outside만
  약하게 성립, manual서 무너짐.

### 2) heatmap evidence spread (#4) = 최고 rho이나 fragile
- spread_front precision 0.35→0.57(최고), rho +0.44~0.55. 그러나 **50×50 belief 양자화로
  spread std=0.84px(27.7~32.2 거의 상수)** — monotone은 실재하나 tau 0.3px 차이로 뒤집히는
  fragile 신호. 배포엔 belief 해상도↑ 선결. spread는 "rear가 틀렸다"가 아니라 "전체 이미지가
  어렵다"를 재는 global difficulty proxy (front spread가 rear보다 상관 높은 이유).

### 3) ensemble(s1) disagreement (#3) = OOD confound로 불신
- ens_rear precision 0.35→0.38, **manual서 net -1**. s1이 squash 입력에 OOD → 불일치가
  epistemic이 아니라 domain-shift. rear rho는 outside/night서 +0.4대이나 도메인 불안정.

### ★ 실질 결론 = 새 필터보다 기존 f4 재튜닝
- **f4_tta_stab(기존 배포필터, scale+brightness TTA)을 tau 5.0→~0.8로 조이면** outside+manual
  precision 0.35→0.55 (bad 26/39 잡고 good 5/21 손실) = spread_front(0.57)와 사실상 동급이고
  fragile하지 않음. 즉 **직교 신호는 이미 f4로 갖고 있었고, threshold가 너무 느슨(5.0)했을 뿐.**
- 어떤 신호도 precision을 ~0.57 이상 못 올림(accepted 43%는 여전히 bad) = "pass=truly-good"
  달성 실패. 순환-밖 신호들이 결국 **weak global difficulty proxy**지 rear-collapse의 clean
  separator가 아님. memory(필터 천장=base 정확도) 재확인.
- 순도 우선(memory) 관점: good 24% 희생하고 precision +20pp는 프로젝트 철학과 정합 → f4 tau
  조임은 채택할 만한 low-cost 개선. spread는 belief 해상도 올린 뒤 재검토.

### 미검증 남은 직교 신호 = temporal (영상 시퀀스)
- filterval fid가 연속 timestamp(영상) → temporal smoothness 검증 가능. 단 systematic
  rear-collapse는 프레임 간에도 일관될 수 있어 temporal도 못 잡을 위험(photometric와 동일 논리).
  transient 오류엔 유효. 별도 실험 필요.

★ caveat: 도메인별 accepted-bad 소표본(outside 17/manual 22), heuristic tau(good-p75),
spread 양자화 한계, ensemble OOD. 단일 weight(Stage B ep0057).
