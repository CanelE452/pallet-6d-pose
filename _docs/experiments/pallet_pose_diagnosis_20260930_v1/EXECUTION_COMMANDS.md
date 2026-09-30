# 실행 명령

환경: `MPLCONFIGDIR=/tmp/pallet-mpl /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_pose_diagnosis_20260930_v1.` 뒤에 아래 모듈·인자를 붙였다. 기존 결과를 덮어쓰지 않으므로 재실행은 새 namespace를 사용한다.

```text
run prepare
run e1
inference prepare
inference r0
inference r0_cpu
inference PRIOR1
inference FULL125
inference e6
scoring detection
scoring visibility
scoring realft
scoring e3
transfer
stress
scoring stress
provenance
close
```

`inference r0`는 CPU/GPU 캐시 재사용 smoke 불일치로 종료했다. `r0_cpu`가 동일 CPU 조건으로 clean/가림을 구성했고 최초 smoke 출력은 재사용했다. 초기 `pallet-pose` 환경의 import는 오래된 ultralytics 때문에 실패했고, 설치된 `pallet-yolo26` 환경으로 옮겼다. 학습/패키지 설치/드라이버 변경은 없었다.
