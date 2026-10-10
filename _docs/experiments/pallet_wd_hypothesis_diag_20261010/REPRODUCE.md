[확인] 아래 명령은 기존 학습·코너 보정 결과를 입력으로 1단계 계산만 재현합니다. 새 학습·데이터 생성은 없습니다.

# 재현

게시 저장소 checkout에서 기존 pose 환경을 사용합니다. 원래 private 입력은 `PALLET_SOURCE_ROOT`로 지정하며 공개 보고서에는 개인 경로를 쓰지 않습니다. `PALLET_WD_OUTPUT`은 존재하지 않는 새 디렉터리로 지정합니다. 완료한 산출물 위에 preflight/stage1을 다시 실행하면 보존 검사로 중단합니다.

```bash
export PALLET_PYTHON=/path/to/existing/pallet-pose/bin/python
export PALLET_SOURCE_ROOT=/path/to/original/private/source
export PALLET_WD_OUTPUT=/path/to/new/nonexistent/output
export PYTHONDONTWRITEBYTECODE=1
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.fresh_stage1
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.summarize
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.report --doc "$PALLET_WD_OUTPUT"
```

공개 수치만으로 표·그림을 다시 만들 때는 다음을 사용합니다. private RGB·checkpoint가 필요하지 않습니다. `--documents-only`는 그림/index를 보존하며 Markdown만 갱신합니다.

```bash
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.report --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.report --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010 --documents-only
```

Depth 준비는 다음처럼 새 private 디렉터리에 pinned 공식 source와 Small checkpoint만 받습니다. 기존 Python 환경은 바꾸지 않고 private target에 부족한 세 패키지만 설치합니다. 설치 도중 build isolation에서 pkg_resources가 없는 문제를 피하기 위해 기존 setuptools를 사용하는 --no-build-isolation을 고정합니다. fetch와 prepare는 모델 생성·forward가 0입니다.

```bash
export PALLET_DEPTH_PRIVATE=/path/to/new/private/metric3d
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.depth fetch --private-root "$PALLET_DEPTH_PRIVATE"
"$PALLET_PYTHON" -B -m pip install --no-deps --no-compile --no-cache-dir --no-build-isolation \
  --target "$PALLET_DEPTH_PRIVATE/deps" mmcv==1.7.2 addict==2.4.0 yapf==0.40.1
"$PALLET_PYTHON" -B -m scripts.research.pallet_wd_hypothesis_diag_20261010.depth prepare --private-root "$PALLET_DEPTH_PRIVATE"
```

실제 depth cache는 Stage2 보고 완료 뒤의 Stage3 entry point입니다. private JSONL에는 id/population/image_path/K/raw_hw/crop_lrtb만 두고 GT·오류·final-test label을 입력하지 않습니다. K는 crop 후 native 좌표에 맞아야 합니다. 원본 RGB와 전체 깊이 map은 공개하지 않습니다. 아직 Stage3 정확도 gate를 평가하지 않은 1단계에서 cache 명령을 실행하거나 S4 결과를 주장하지 않습니다.
