# E3 3k seed1 복구 보고 — PHASE 0

## 무엇이 망가졌나

E3 의 checkpoint 를 다시 만들려고 재실행했는데, 러너가 mark 마다 결과를 덮어쓰는
구조라 **원본 JSON 이 같은 경로에 열렸고** mark 0 직후 kill 되면서 6개 mark 중 5개를
잃었다. 백업을 먼저 뜨지 않은 것이 원인이다.

```
손상본   splitlate_E3_SPLIT_LATE_seed1.json.damaged_backup_20260817_173612
         sha256 84acbad52fb1...   (damaged_backup_sha256.txt 에 기록)
         남은 mark: [0] 하나뿐
복구본   splitlate_E3_SPLIT_LATE_seed1_recovery.json
         mark: [0, 250, 500, 1000, 2000, 3000]  전부
```

재발 방지로 `mh_splitlate.py` 에 `--label` 을 넣어 **복구 실행이 원본 경로에 착지할 수
없게** 했다(`splitlate_{ARM}_seed{n}_{label}.json`).

## 복구본이 원본과 같은가 — 세 갈래로 확인

손상본에 mark 0 밖에 없어서 직접 대조로는 한 점밖에 못 본다. 그래서 **살아남은 파생
결과에서 원본 값을 역산해 미리 고정한 뒤** 복구본을 열었다.

### 1. 사전 고정한 예측값과 정확 일치

`capacity_control_compare.json` 이 `E3_vs_E4` cornerC = **+20.6279%** 를 들고 있고,
`capacity_control_seed1.json` 의 E4 @3000 cornerC = **0.9555** 다. 따라서 원본 E3 값은
`0.9555 × (1 − 0.206279) = 0.758400` 이어야 한다. 복구본을 열기 **전에** 적어두었다.

```
recovered cornerC = 0.758400
predicted (locked) = 0.758400        diff 0.00e+00
```

### 2. line 은 E0 와 6개 mark 전부 bit-identical

E3 의 line 무손실은 배선의 성질이지 측정 결과가 아니다. 복구본에서도 그대로 성립한다.

```
mark    E0 angle    E3 angle      diff   |  E0 offset   E3 offset     diff
    0   2.216270  2.216270  0.00e+00  |  1.078695  1.078695  0.00e+00
  250   2.250423  2.250423  0.00e+00  |  1.011897  1.011897  0.00e+00
  500   2.291849  2.291849  0.00e+00  |  1.077412  1.077412  0.00e+00
 1000   2.558907  2.558907  0.00e+00  |  1.115673  1.115673  0.00e+00
 2000   2.450531  2.450531  0.00e+00  |  1.083322  1.083322  0.00e+00
 3000   2.309444  2.309444  0.00e+00  |  1.086584  1.086584  0.00e+00
```

### 3. 손상본에 남은 mark 0 과 일치

```
@0  angle 2.216270 vs 2.216270   diff 0.00e+00
```

## 결론

```
E3_3K_SEED1_RECOVERED = True     결정적 재현, 3갈래 모두 0.00e+00
```

기존 판정 중 이 arm 에 의존하는 것은 다시 돌릴 필요가 없다. 참고로 복구본 기준
E3 vs E2 cornerC 는 **+18.82%**(point estimate)이고, 보고서의 +19.54% 는 paired frame
bootstrap 의 중앙값이다 — 서로 다른 추정량이며 +18.82% 는 그 CI [+15.67, +25.90] 안에
있다. 모순이 아니다.

## 비교 시 주의 (같은 실수를 두 번 했다)

3k arm(E0~E4)은 **A0 @18,000 에서 이어받은 continuation** 이다. from-scratch screen 인
`mh_screen_A0_LINE_ONLY.json` 과 비교하면 line angle 이 2.31 vs 5.83 으로 벌어져
"line 이 깨졌다" 는 잘못된 결론이 나온다. 3k arm 의 line 기준선은 항상
`stopgrad_E0_CONTINUE_LINE_seed{n}.json` 이다.
