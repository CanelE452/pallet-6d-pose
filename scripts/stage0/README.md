# scripts/stage0

실험 스크립트가 쌓이는 곳. 2026-08-14 에 196개까지 불어나 계열별로 나눴다.

```
루트            124   서로 import 로 얽혀 있어 나눌 수 없는 것들 + 아직 분류 안 한 것
line/            17   구조선 계열 — direct_hough · structural_line · instance_edge
                      role_* · supporting_line · edge_mandatory
paper_s2/        19   PAPER_S2 스크린 중 독립 실행되는 것
stage_screens/    7   stage15~25 스크린
diffpnp3d/        3   DiffPnP3D 계열
_archive/        26   저장소 어디에서도 참조되지 않는 것 (README 참조)
```

## 왜 절반 이상이 루트에 남았나

import 의존을 그래프로 보면 **88개가 하나의 연결 덩어리**다. `sys.path` 에는
stage0 루트만 들어가므로, 이 덩어리를 흩으면 서로를 찾지 못한다. 나눌 수 있는 것은
그래프상 고립된 파일뿐이고, 그중 접두어가 뚜렷한 46개만 옮겼다.

## 여기 파일을 옮길 때 확인할 것

한 단계 깊어지면 **자기 위치로 계산한 경로가 전부 한 칸 밀린다.** 이동과 보정은
같은 작업으로 해야 하고, 아래를 전부 봐야 한다. 2026-08-14 재편에서 이 목록은
하나씩 사고를 내며 늘어났다.

```
ROOT 계산
  parents[N]                                  → N+1
  os.path.join(dirname(__file__), "..", "..") → ".." 하나 추가
     ★ 중첩 괄호라 단순 정규식이 첫 ')' 에서 멈춘다. 7개를 놓쳤다가 검산에서 잡음
  sh: cd "$(dirname "$0")/../.."              → 4개가 여기 걸려 이동에서 제외

이 파일을 가리키는 참조 — 형태가 네 가지다
  "scripts/stage0/x.py"           문자열
  ROOT / "scripts" / "stage0" / "x.py"    Path 조립
  STAGE0 / "x.py"                 변수 경유   ★ 이걸 놓쳐 20 failed / 50 errors
  _load("이름", STAGE0 / "x.py")  동적 로드   ★ import 문만 보면 안 보인다
```

`import` 문만 훑어서는 의존을 다 못 본다. 위 네 형태를 모두 grep 한 뒤에 옮길 것.

## 보정이 맞았는지 확인하는 법

학습·평가 스크립트라 실행해서 검증할 수 없다. 대신 파일을 실행하지 않고 **ROOT
대입식만 평가**해 저장소 루트가 나오는지 본다. 재편 때 이 검산이 미보정 7건을
잡았다.

```python
tree = ast.parse(path.read_text())
ns = {"__file__": str(path), "os": os, "pathlib": pathlib, "sys": sys, "Path": Path}
# ROOT 앞의 단순 대입(HERE 등)을 순차 평가한 뒤 ROOT 를 평가해 비교
```

마지막 관문은 `pytest challenge/tests/` 943 passed 다.
