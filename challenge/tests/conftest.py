"""stage0 모듈을 이름으로 import 할 수 있게 sys.path 를 준비한다.

여러 테스트가 `import paper_s2_frozen_diagnostic` 처럼 stage0 스크립트를 모듈
이름으로 부른다. 예전에는 stage0 이 평평한 폴더라 어느 스크립트든 먼저 로드되면서
sys.path 를 채워 줬고, 그 부수효과에 얹혀 동작했다 — 즉 테스트 실행 순서에 의존하고
있었다.

2026-08-14 에 stage0 을 계열 폴더로 나누면서 그 우연이 깨졌다. pytest 는 conftest 를
가장 먼저 읽으므로, 여기서 stage0 과 그 하위 폴더를 전부 넣어 순서 의존을 없앤다.
"""
import pathlib
import sys

_STAGE0 = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "stage0"

if _STAGE0.is_dir():
    _dirs = [_STAGE0] + sorted(p for p in _STAGE0.iterdir()
                               if p.is_dir() and not p.name.startswith("."))
    for _d in _dirs:
        _s = str(_d)
        if _s not in sys.path:
            sys.path.insert(0, _s)

# challenge/scripts 도 2026-08-14 에 계열 폴더로 나뉘었다. annotate_pnp 처럼 이름으로
# 부르는 모듈이 여러 테스트에 있어 같은 처리를 한다.
_CS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if _CS.is_dir():
    for _d in [_CS] + sorted(p for p in _CS.iterdir()
                             if p.is_dir() and not p.name.startswith((".", "_"))):
        _s = str(_d)
        if _s not in sys.path:
            sys.path.insert(0, _s)
