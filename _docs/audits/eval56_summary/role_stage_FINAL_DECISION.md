# Depth-role stage fusion — static 전부 FAIL, PFDR 조건 충족

> 기존 8 corner + centroid heatmap, 기존 decoder, centroid 포함 canonical PnP 유지.
> **추가 학습 0 step**으로 E0~E5·F0~F4 를 실행하고 G1~G4 oracle 을 전부 계산했다.
> eval56(내 파렛트 56) / wood(처음 본 목제 45) 두 셋에서 재현되는 변경만 채택한다.

## [관찰] E0~E5 / F0~F4 (eval56)

```
                  arm  pnp    reproj   corner     near       far  far_near_ratio  t50  t100  nan_corner
              E0_base   50 11.557805 7.241083 4.675503 11.406327        2.439593   45    17         119
            E1_all_s5   50 13.145233 6.682566 5.084661  9.642184        1.896328   48    17         116
              E2_DRSF   50 11.743258 6.397502 4.675503  9.642184        2.062277   45    16         120
            E4_far_s4   51 13.921725 6.890918 4.675503 10.958519        2.343816   48    20         116
           E5_reverse   50 11.249220 7.306098 5.084661 11.406327        2.243282   48    18         115
           F0_sw_full   50  8.519145 7.416222 6.075932 10.980696        1.807245   45    15          73
  F1_base_near_sw_far   49  9.934825 6.875035 4.675503 10.980696        2.348559   42    13         115
F3_base_near_sw_s5far   49 10.808236 6.271505 4.675503  9.384093        2.007077   42    14         124
      F4_reverse_ckpt   53 10.602228 7.742025 6.075932 11.406327        1.877297   48    19          77
```

## [관찰] wood

```
                  arm  pnp   reproj   corner     near       far  far_near_ratio  t50  t100  nan_corner
              E0_base   44 9.283903 9.225494 6.732508 14.179799        2.106169   40    36          51
            E1_all_s5   44 8.695505 9.186070 6.779662 11.877580        1.751943   35    31          56
              E2_DRSF   44 9.032884 8.775418 6.732508 11.877580        1.764213   38    34          53
            E4_far_s4   44 9.826459 9.484098 6.732508 13.591344        2.018764   36    31          54
           E5_reverse   44 9.383798 9.530172 6.779662 14.179799        2.091520   37    33          54
           F0_sw_full   44 9.583033 9.488703 7.379296 13.467676        1.825063   40    35          43
  F1_base_near_sw_far   43 9.452757 9.215812 6.732508 13.467676        2.000395   34    29          60
F3_base_near_sw_s5far   43 9.695007 8.908184 6.732508 11.754992        1.746005   31    28          60
      F4_reverse_ckpt   45 8.826763 9.418441 7.379296 14.179799        1.921565   46    42          34
```

## H1 지지 — corner 수준에서는 near/far 최적 stage가 다르다

G1 role-stage grid 9 조합, corner median 오름차순:

```
eval56
 near_stage  far_stage  pnp    reproj   corner     near       far  t50  t100
          6          5   50 11.743258 6.397502 4.675503  9.642184   45    16
          5          5   50 11.763887 6.682566 5.084661  9.642184   48    17
          4          5   50 12.171502 6.848293 5.234151  9.642184   50    20
          6          4   51 13.921725 6.890918 4.675503 10.958519   48    20
          5          4   51 13.303286 7.195390 5.084661 10.958519   51    21
          6          6   50 11.557805 7.241083 4.675503 11.406327   45    17
          5          6   50 11.249220 7.306098 5.084661 11.406327   48    18
          4          4   51 13.748526 7.395566 5.234151 10.958519   53    24
          4          6   50 10.833206 7.926672 5.234151 11.406327   50    21

wood
 near_stage  far_stage  pnp    reproj    corner     near       far  t50  t100
          6          5   44  9.032884  8.775418 6.732508 11.877580   38    34
          5          5   44  9.140547  9.186070 6.779662 11.877580   35    31
          6          6   44  9.283903  9.225494 6.732508 14.179799   40    36
          4          5   44  9.647206  9.399035 7.655376 11.877580   35    31
          6          4   44  9.826459  9.484098 6.732508 13.591344   36    31
          5          6   44  9.383798  9.530172 6.779662 14.179799   37    33
          5          4   44 10.311480 10.000942 6.779662 13.591344   33    28
          4          6   44 10.441310 10.115628 7.655376 14.179799   37    33
          4          4   44  9.789473 10.332766 7.655376 13.591344   33    28
```

[확인] **near=H6 / far=H5 (E2) 가 두 셋 모두에서 9 조합 중 1 위**다.
far median 이 eval56 11.41 → 9.64 (-15.5%), wood 14.18 → 11.88 (-16.2%) 로 개선되고
near 는 정의상 정확히 불변이다.

[확인] E5 reverse control(near=H5 / far=H6)은 far 를 전혀 개선하지 못한다(0.0%).
즉 이득은 **stage 평균화 효과가 아니라 far 에 H5 를 쓰는 데서** 온다 — H1 을 지지한다.

## H2 기각 — corner 개선이 pose 로 전환되지 않는다

```
arm      far median        reprojection        판정
E2 eval56  -15.5%            -1.6% (악화)      FAIL
E2 wood    -16.2%            +2.7%             FAIL
```

Phase J gate 실패 항목:

```
  eval56  E2_DRSF                ['reproj', 'NaN 미증가', 'P>=0.90']
  eval56  E1_all_s5              ['reproj', 'near<=+5%', '>50 미증가', 'imp>wor', 'P>=0.90']
  eval56  E4_far_s4              ['far -10%', 'reproj', '>50 미증가', '>100 미증가', 'imp>wor', 'P>=0.90']
  eval56  E5_reverse             ['far -10%', 'reproj', 'near<=+5%', '>50 미증가', '>100 미증가', 'imp>wor', 'P>=0.90']
  eval56  F1_base_near_sw_far    ['far -10%', 'PnP>=50', 'P>=0.90']
  eval56  F3_base_near_sw_s5far  ['reproj', 'PnP>=50', 'NaN 미증가', 'P>=0.90']
  wood    E2_DRSF                ['NaN 미증가', 'P>=0.80']
  wood    E1_all_s5              ['NaN 미증가', 'P>=0.80']
  wood    E4_far_s4              ['far -5%', 'reproj', 'NaN 미증가', 'imp>wor', 'P>=0.80']
  wood    E5_reverse             ['far -5%', 'reproj', 'NaN 미증가']
  wood    F1_base_near_sw_far    ['reproj', 'PnP>=44', 'NaN 미증가', 'imp>wor', 'P>=0.80']
  wood    F3_base_near_sw_s5far  ['reproj', 'PnP>=44', 'NaN 미증가', 'imp>wor', 'P>=0.80']
```

[확인] **모든 static fusion arm 이 두 셋에서 FAIL** 이다.
E2 는 wood 에서 NaN corner +2 와 P(improve) 0.774(기준 0.80) 두 항목만 남을 만큼 근접하지만,
eval56 에서 reprojection 이 기준 -10% 대신 +1.6% 악화라 통과할 수 없다.

## H3 기각 — stagewise far 만 가져와도 안 된다

```
F1 (base near + stagewise far)  eval56 reproj -14.0% 이지만 PnP 50→49 로 FAIL
                                wood   reproj +1.8% 악화, improved 15 / worsened 28 FAIL
F4 reverse control              eval56 PnP 53 로 최고 — far 가 아니라 near 를 바꿨을 때다
```

[확인] F4(near 만 stagewise)가 PnP 를 53 까지 올린다.
stagewise 의 이득이 far 에만 국한된다는 해석은 이 데이터에서 지지되지 않는다.

## H4 — oracle 상한은 두 셋 모두 존재

```
                G2 per-corner stage    G3 per-role source (far)
eval56              +24.5%                 +14.5%  (far +27.7%)
wood                +16.1%                 +18.5%  (far +30.3%)
```

[확인] 고정 조합으로는 못 얻지만 **상한은 두 셋 모두 10% 이상**이다.
Phase M 의 PFDR 실행 조건 세 가지가 모두 충족된다.

```
static_all_fail    True
oracle_ge10_both   True   (G3 far +27.7% / +30.3%)
same_direction     True   (E2 가 두 셋 모두 1/9, far 개선 동일 방향)
-> PFDR 실행 필요 = True
```

## Phase R — learned router NOT RUN

```
        base    oracle-margin   개선     선택비율   판정
eval56  8.41 -> 6.63px         +21.1%    0.197     PASS
wood    9.82 -> 9.50px          +3.3%    0.096     FAIL (기준 -10%)
```

두 셋 동시 PASS 가 아니므로 **learned router 를 학습하지 않는다**.
eval56 단독 결과는 same-pallet oracle 로만 기록한다.

## [현재 판정]

```
H1 near/far 최적 stage 상이       ACCEPT (corner 수준, 두 셋 1/9)
H2 고정 fusion 으로 pose 개선     REJECT (reprojection 미개선)
H3 stagewise far-only 이식        REJECT (PnP 하락, wood 악화, F4 반증)
H4 near/far decoupled 상한        존재 -> PFDR 조건 충족, 미실행
learned coordinate router         NOT RUN (wood oracle FAIL)
채택 architecture                 base ep57 (변경 없음)
```

## [지지 증거]

- [확인] baseline parity 3 셋 재현 후 시작.
- [확인] E2 가 corner 기준 9 조합 중 1 위, 두 셋 일치.
- [확인] E5 reverse control 이 far 를 개선하지 못해 stage 평균화 가설을 배제.
- [확인] far-GT 치환 oracle 이 near 치환보다 큰 이득(48.3% vs 42.9%, 36.7% vs 20.9%).

## [반증 증거 / 한계]

- [확인] corner 가 15~16% 좋아져도 reprojection 은 개선되지 않는다.
  이 프로그램에서 반복된 "corner 이득이 pose 로 전환되지 않음" 과 같은 벽이다.
- [확인] F4(near 교체)가 PnP 최고 — far-only 해석의 반례.
- [확인] G2 per-corner oracle 의 far 선호 stage 가 두 셋에서 다르다
  (eval56 base4 0.38 최다 / wood base5 0.369 최다).  고정 규칙이 어려운 이유다.
- [확인] F2 표본이 eval56 8 / wood 5 로 얇아 F2 지표를 단독 근거로 쓰지 않았다.

## [다음 admissible experiment]

1. **PFDR 조건이 충족됐으므로 far-only adapter 3 epoch 학습(N1/N2/N3)이 다음 순서**다.
   near/centroid 는 base H6 로 exact 고정, far 만 H5 anchor + zero-init residual.
2. N3(near adapter negative control)를 반드시 함께 돌려 far-specific 여부를 판별한다.
3. PFDR 이 FAIL 하면 stage/coordinate 층위는 종료하고 base ep57 을 유지한다.
