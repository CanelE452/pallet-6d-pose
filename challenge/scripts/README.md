# challenge/scripts

2026-08-14 에 평평한 36개를 역할별로 나눴다.

```
annotate/     8   어노테이션 툴과 그 모듈 (annotate · _draw · _io · _wood),
                  GT 후처리 (_repnp_with_new_dims · fix_manual_swap · restore_orig ·
                  convert_to_camera_facing_v4)
dataset/      7   데이터셋 생성·증강 (augment · merge · truncation crops · pseudo/apriltag GT)
visualize/    6   belief · 변환 GT · v4 preview · 시퀀스 뷰어 · loss 곡선
live/         5   실시간 추론 (run_live 와 그 모듈) + 지연 측정
evaluate/     4   A/B 비교 · manual vs inference · 시퀀스 통계
infer/        3   mp4 추론 (DOPE · DOPE+padding · YOLO)
_legacy/     37   손대지 않는 옛 코드
루트          4   아래 참조
```

## 폴더가 나뉘어도 서로를 찾는 방법

각 파일 최상단에 형제 탐색 블록이 있다. `scripts/stage0` 과 같은 방식이고, 이유도 같다 —
`sys.path` 에 `challenge/scripts` 만 들어가면 나누는 순간 서로를 못 찾는다.

```python
_CS = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_CS] + [_os.path.join(_CS, _d) for _d in sorted(_os.listdir(_CS)) ...]
```

저장소 바깥에서 이 모듈들을 이름으로 부르는 곳(39개 파일, 대부분 `annotate_pnp`)은
`sys.path` 에 하위 폴더를 함께 넣도록 고쳤다. 테스트는 `challenge/tests/conftest.py`
가 처리한다.

## ★ 루트에 남는 것

```
annotate_pnp.py    ★옮길 수 없다. 해시가 고정된
                   scripts/stage0/paper_s2_frozen_diagnostic.py 가
                   "challenge/scripts" 경로로 이 모듈을 import 하는데, 그 파일은
                   한 글자만 바뀌어도 cache_key 가 달라져 저장된 결과가 무효가 된다.
                   대신 이 파일에 형제 탐색 블록을 넣어 annotate/ 안의 모듈을 찾게 했다.
finetune.sh · run_pipeline.sh · annotate_forklift.bat    진입점
```

## 여기 파일을 옮길 때

`scripts/stage0/README.md` 의 목록이 그대로 적용된다. 여기서 추가로 물린 것:

```
challenge/data_paths 보일러플레이트
  _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
  → 한 단계 깊어지면 parents[3]. "ROOT =" 로 시작하지 않아 depth 보정에서 놓쳤다.

해시 고정 파일의 의존
  옮길 수 없는 파일이 import 하는 모듈도 옮길 수 없다. 의존을 한 단계만 보지 말 것 —
  annotate_pnp → convert_to_camera_facing_v4 처럼 연쇄한다.

로드 순서 의존
  "다른 모듈이 sys.path 를 채워 준다" 는 주석이 붙은 파일이 여럿 있었다.
  그 우연은 폴더를 나누는 순간 깨진다.
```

검증은 `pytest challenge/tests/` 943 passed 와, 서브프로세스로 실행되는 스크립트를
직접 한 번 돌려 보는 것까지다.
