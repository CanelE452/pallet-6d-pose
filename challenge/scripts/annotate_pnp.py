"""annotate_pnp 재수출 shim — 실제 코드는 scripts/annotate/annotate_pnp.py 에 있다.

2026-08-15 에 어노테이션 툴 전체를 `challenge/scripts/annotate/` 에서
`scripts/annotate/` 로 옮겼다. 그런데 `scripts/stage0/paper_s2_frozen_diagnostic.py`
가 `challenge/scripts` 를 sys.path 에 넣고 `import annotate_pnp` 를 한다. 그 파일은
자기 소스 해시로 cache_key 를 만들기 때문에, import 경로를 고치려고 한 글자라도
건드리면 저장된 진단 결과가 전부 무효가 된다.

그래서 모듈을 옮기되 이 자리에 이름만 남겼다. 여기서 새 위치를 sys.path 에 넣고
그대로 재수출하므로, 옛 경로로 부르는 코드도 수정 없이 같은 객체를 얻는다.

★ 이 파일에 로직을 추가하지 말 것. 실제 구현은 scripts/annotate/annotate_pnp.py 다.
"""
import os as _os
import sys as _sys

_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_NEW = _os.path.join(_REPO, "scripts", "annotate")
if _NEW not in _sys.path:
    _sys.path.insert(0, _NEW)

# 같은 이름의 이 shim 이 아니라 새 위치의 모듈을 확실히 집도록 파일 경로로 로드한다.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_annotate_pnp_impl",
                                     _os.path.join(_NEW, "annotate_pnp.py"))
_impl = _ilu.module_from_spec(_spec)
_sys.modules["_annotate_pnp_impl"] = _impl
_spec.loader.exec_module(_impl)

# 이름을 실제 모듈 객체로 바꿔치기한다. 값을 globals() 로 복사하면 사본이 생겨서,
# mock.patch.object(annotate_pnp, "PALLET_DIMS", ...) 같은 패치가 원본에만 적용되고
# 이쪽 사본은 그대로 남는다(2026-08-15 테스트 3건이 실제로 이렇게 깨졌다).
# 모듈 자체를 넘기면 옛 경로로 import 해도 새 위치와 같은 객체를 공유한다.
_sys.modules[__name__] = _impl
