# stage0/_archive

2026-08-14. `scripts/stage0` 가 196개까지 불어나 활성 스크립트가 묻혔다. 여기 있는
26개는 **저장소 어디에서도 참조되지 않는** 것들이다 — 테스트가 경로로 열지도, 다른
스크립트가 import 하지도, 셸/설정에서 부르지도 않는다. 지운 게 아니라 옮겼을 뿐이고,
`python scripts/stage0/_archive/<name>.py` 로 그대로 실행된다.

## 옮겨도 되는지 어떻게 판단했나

한 단계 깊어지므로, **자기 위치를 기준으로 무언가를 계산하는 스크립트는 옮기면
조용히 다른 곳을 가리킨다.** 그래서 아래를 전부 확인하고, 하나라도 걸리면 제외했다.

```
py   parents[N]                     한 칸 밀려 ROOT 가 scripts/ 를 가리키게 된다
py   HERE = dirname(__file__) + "/../.."      같은 이유
py   sys.path.insert(0, HERE) 후 형제 import  HERE 가 _archive 라 형제를 못 찾는다
sh   cd "$(dirname "$0")/../.."     ★ py 만 보다가 놓칠 뻔했다. 4개가 여기 걸려 제외
```

절대 경로(`ROOT = "/home/minjae/..."`)나 repo 루트 기준 상대 경로
(`scripts/stage0/x.py`)만 쓰는 것은 위치와 무관하므로 안전하다.

## 제외한 것 (stage0 루트에 그대로 있다)

```
보정 없이는 못 옮김
  diag7_partial_pnp.py                 HERE/../.. + 형제 import
  extract_gt_candidates.py             HERE/../..
  paper_s2_regression_3to8_montage.py  형제 import
  paper_s2_stagewise_bias_report.py    parents[N]
  paper_s2_testset17_overlay.py        형제 import
  role_target_aliasing_audit.py        parents[N]
  stage20_cutpaste_eval.py             HERE/../.. + 형제 import
  stage20_cutpaste_train.py            HERE/../.. + 형제 import
  run_stage24_{evalall,quickscreen,seeneval}.sh · run_stage7_sparse.sh
                                       cd "$(dirname "$0")/../.."
```

## 검증

26개 전부 `py_compile` / `bash -n` 통과, 남은 코드에서 이들을 가리키는 참조 0건,
`pytest challenge/tests/` 943 passed. 되돌리려면 파일을 한 단계 위로 옮기면 된다.
