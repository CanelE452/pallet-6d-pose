# scripts/stage0

실험 스크립트가 쌓이는 곳. 2026-08-14 에 196개까지 불어나 계열별로 나눴다.

```
루트             19   ★해시에 묶인 3개 + 계열이 정말 없는 것
paper_s2/        34   PAPER_S2 스크린
stage_screens/   28   stage5~25 · offset · padding · v3
_archive/        26   저장소 어디에서도 참조되지 않는 것
line/            21   구조선 — direct_hough · structural_line · instance_edge
                      role_* · supporting_line · partial_edge · degeneracy · r1c
diag/            13   diag1~8 진단
_run/            11   셸 실행 래퍼
selftrain/       10   s1_ · s2_ · s16_ (self-training / 체크포인트 선정)
ralph/            8   ralph 계열
filter_pl/        7   tau_calibrate · four_arm_pl_compare · pl_* · orthogonal_filters
diffpnp3d/        5
adaptation/       4   late_a1 · regularized_late_a1
eval_harness/     4
wood/             4
paper_s1/         2
```

## 폴더가 나뉘어도 서로를 찾는 방법

계열 경계를 넘는 import 가 119개다. `sys.path` 에 stage0 루트만 넣으면 나누는 순간
전부 끊긴다. 그래서 하위 폴더의 모든 `.py` 최상단에 이 블록이 있다.

```python
import os as _os, sys as _sys
_S0 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_S0] + [_os.path.join(_S0, _d) for _d in sorted(_os.listdir(_S0))
                         if _os.path.isdir(_os.path.join(_S0, _d)) and not _d.startswith(".")]
```

**docstring 바로 다음**에 둔다. 형제를 import 하는 줄보다 먼저 실행돼야 하는데,
"마지막 import 뒤"에 넣으면 형제 import 도 import 문이라 이미 늦는다. 실제로 그렇게
넣었다가 `ModuleNotFoundError` 를 봤다.

테스트 쪽은 `challenge/tests/conftest.py` 가 같은 일을 한다. 전에는 어느 스크립트가
먼저 로드되며 `sys.path` 를 채워 주는 부수효과에 얹혀 있었다 — 즉 실행 순서에
의존했고, 폴더를 나누자 그 우연이 깨졌다.

## ★ 루트에 남겨야 하는 파일

**자기 파일의 sha256 을 기록하는 스크립트는 옮기면 안 된다.** 캐시·감사 무결성이
그 해시에 걸려 있어서, 한 글자만 바뀌어도 저장된 결과가 무효가 된다.

```
paper_s2_frozen_diagnostic.py      cache_key 의 frozen_helper_sha256
paper_s2_mechanism_diagnostic.py   cache_key 의 script_sha256 (자기 자신)
paper_s2_frozen_post_analysis.py   같은 계열
```

찾는 법: `grep -rln "sha256_file(.*__file__\|script_sha256" scripts/stage0/`

## 여기 파일을 옮길 때

한 단계 깊어지면 자기 위치로 계산한 경로가 한 칸 밀린다. 이동과 보정은 한 작업이다.

```
ROOT 계산
  parents[N]                                   → N+1
  os.path.join(HERE, "..", "..")               → ".." 하나 추가
  os.path.join(os.path.dirname(__file__), …)   중첩 괄호라 순진한 정규식이 첫 ')' 에서 멈춘다
  sh: cd "$(dirname "$0")/../.."               py 만 검사하면 놓친다

이 파일을 가리키는 참조 — 다섯 가지다
  "scripts/stage0/x.py"                  문자열
  ROOT / "scripts" / "stage0" / "x.py"   Path 조립
  STAGE0 / "x.py"                        변수 경유
  _load("이름", STAGE0 / "x.py")         동적 로드 (import 문에 안 보인다)
  from scripts.stage0 import x           패키지 import
```

`import` 문만 훑으면 의존을 다 못 본다. 다섯 형태를 모두 grep 한 뒤 옮길 것.

## 보정이 맞았는지 확인하는 법

학습·평가 스크립트라 실행 검증이 불가능하다. 파일을 실행하지 않고 **ROOT 대입식만
평가**해 저장소 루트가 나오는지 본다 (`ROOT` 앞의 `HERE` 같은 대입을 순차 평가한 뒤
`ROOT` 를 평가). 재편 때 이 검산이 미보정 9건을 잡았다.

마지막 관문은 `pytest challenge/tests/` 943 passed 다.

셸 래퍼는 부트스트랩이 없으니 `cd "$(dirname "$0")/../../.."` 가 정말 저장소 루트를
내는지 직접 확인한다. `bash -c 'cd "$(dirname "<경로>")/../../.." && pwd'`.

## flaky 테스트

`test_the_adapted_arm_carries_gradient` 와
`test_h2_is_nearly_invariant_to_a_uniform_probability_floor` 는 전체 실행에서
간헐적으로 떨어진다 (4회 돌려 2/1/1/0). 단독으로는 통과하고 동명 모듈 충돌도 없다.
memory 의 "grid_sample input-grad 는 비결정" 과 같은 계열로 보인다 — 재편 때문이
아니라 원래 그런 것으로 판단했으나 확증은 못 했다. 실패하면 한 번 더 돌려 볼 것.
