# 재현 및 환경

Ubuntu22.04.5 LTS, minjae-MS-7D17, i9-11900(16 logical CPU),31GiB RAM. 호스트 RTX3080 10GiB/driver580.178.04/CUDA13.0 확인. 확인 당시 GPU≈992MiB/7%는 데스크톱·rustdesk 사용이며 학습 작업 점유는 보이지 않았다. sandbox ps/GPU는 격리되므로 호스트 읽기 전용 확인을 따로 수행했다. 작업 종료·GPU 선점 없음.

Python `3.10.20`, NumPy `1.26.4`. 정확한 dependency 버전은 ENVIRONMENT.json. 모든 계산CPU1, BLAS/OMP1, GPU0. 분석CLI wall=19.594951s이며 전체 작업/시각 검토 시간이나 영상추론 latency/FPS가 아니다.

기준 저장소 remote `https://github.com/CanelE452/pallet-6d-pose.git`, 공개 저장소, 원격 main 확인SHA `93c24d143fa3901de1fa127b3156e8250d81dba8`. 사용자 원 checkout은 다른 오래된 main과 dirty 변경을 보존했다. 최신 worktree는 위SHA와 기존 untracked draft를 보존했다. 현재 작업의 새 파일만 별도 게시 checkout에서 commit한다.

아래 DATA_ROOT는 기존 local 연구 캐시 디렉터리다. 원 RGB·참조·체크포인트·후보은행은 공개 패키지에 포함하지 않으며 없는 입력을 재구성/다운로드하지 않는다. 관련9개 보고서는 source repo `_docs/experiments`에서 모두 찾았다. 다른 모집단으로 대체하지 않았다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/pallet-mean-gap-mpl
PY=/home/minjae/anaconda3/envs/pallet-pose/bin/python
DATA_ROOT=/home/minjae/Documents/pallet-research
SOURCE_REPO=/tmp/pallet-bi-full-execution-20261019-v1
NEW_OUT=/tmp/pallet-mean-gap-independent-reproduction
$PY -B analyze.py --data-root "$DATA_ROOT" --source-repo "$SOURCE_REPO" --out "$NEW_OUT"
$PY -B visualize_results.py --data-root "$DATA_ROOT" --analysis "$NEW_OUT" --out /tmp/pallet-mean-gap-new-figures
$PY -B verify.py --public . --data-root "$DATA_ROOT" --output /tmp/pallet-mean-gap-check.json
```

analyze.py/visualize_results.py는 새 출력만 생성하고 기존 출력 overwrite를 거부한다. make_report.py는 실제 local 실행 폴더(results_attempt2, figures, PREFLIGHT.json)를 받아 문서·공개 subset을 조립하는 packaging 전용 도구다. 표는 원정밀 CSV, 문서는 반올림 요약이다. 공식 연속 지표는 available-only; paired 공통 산출은 별도 표다. 실패/no-pose가0오차로 대입되지 않는다. median/linear P90/P95·sample variance(ddof1),5cm/5° 공동성공·엄격한 큰오차 횟수를 사용한다. Δ는 method−comparator이다.

입력 체인: 원 D1 real238 upstream/HEAD bank/봉인 predictions → P0/S1/S2/S3S4/F와 기존risk/observation evaluation scalar. 고정 reference seal SHA를 검증한 후 saved pose를 재채점한다. 238RGB hash·K·치수·partition·session·recording을 조인하고 원 saved returned pose5경로와 bitwise 일치를 확인한다. 그림은 같은 reference와 saved observation·pose만 표시한다. SOURCE_MANIFEST는 상대 alias/기존 basename/bytes/SHA를 보존하고 local 절대경로 mapping은 비공개로 남긴다.

기술 실패1회: 처음 reader는 모든 seal에 complete=true가 있다고 가정해 과거 risk/observation의 status=COMPLETE 형식에서 조기중단했다. 그 초기 소스·빈 출력은 local 보존했다. 실제7개 seal과 scored hash를 확인하고 exact COMPLETE 형식도 허용하도록 reader만 수정했다. 과학 설정·원 데이터 변경은0이다. 최종 분석은 results_attempt2 한 번 완료했다.
