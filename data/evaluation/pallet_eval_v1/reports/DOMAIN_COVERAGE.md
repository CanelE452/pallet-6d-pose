# Acquisition domain coverage

`acquisition_domain` 은 환경 factor 가 아니라 **capture provenance** 다.
indoor/outdoor x day/night 2x2 factorial 설계는 폐기했다 — 두 축을
독립으로 가를 provenance 가 데이터에 없다.


```text
Domain      Obj      Role                      N  Sess  Min  Pref  MinSess   Status
--------------------------------------------------------------------------------------------
outside     plastic  MAIN_REQUIRED            70     3   50    60        2   PREFERRED_READY
night       plastic  MAIN_REQUIRED            28     2   50    60        2   FRAME_DEFICIT
noapril     plastic  CONDITIONAL              12     1   40    50        2   DEFICIT
cad         any      APPENDIX_STRESS_ONLY     18     1    0     0        0   APPENDIX_ONLY
```

## Frames per acquisition domain (전체 PAPER_EVAL positive)

```text
outside         70
night           28
noapril         12
cad             18
unknown         45
TOTAL          173
```

## Status

```text
PREFERRED_READY    frames >= preferred AND sessions >= minimum_sessions
MINIMUM_READY      frames >= minimum   AND sessions >= minimum_sessions
SESSION_DEFICIT    frames 충족 · 독립 세션 부족  <- 인접 프레임 몰아찍기 방지
FRAME_DEFICIT      세션 충족 · frames 부족
DEFICIT            둘 다 부족
APPENDIX_ONLY      MAIN readiness 계산에 넣지 않는다
```

## Deficit

```text
Domain      Obj        have  to min  to preferred
-------------------------------------------------
noapril     plastic      12      28            38
night       plastic      28      22            32
```

## M2 dataset gate

```text
MAIN_DOMAINS_READY   false
```

이 게이트가 참이 아니면 `_docs/paper/EXPERIMENTS.md` 의 M2 는 성립하지 않는다.
도메인 배정 근거는 `ACQUISITION_DOMAIN_MAP.json` 에 세션별로 적혀 있다.

