"""기존 evaluator 를 그대로 실행하되 inference-path patch 를 먼저 건다.

evaluator 정의는 한 줄도 바꾸지 않는다 — runpy 로 원본 스크립트를 실행한다.
"""
import os, runpy, sys

NS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NS)
import p26_paths                                                    # noqa: E402

mode = sys.argv[1]
script = sys.argv[2]
p26_paths.install(mode)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
print(f"[p26] mode={mode} calls={p26_paths.CALLS}", file=sys.stderr)
