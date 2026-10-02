# 실행 명령

실행 checkout은 `/tmp/pallet-pose-n3-v3` (a7fb680 기반)이다. 원시 자료 소유 checkout은 `/home/minjae/Documents/github/pallet-pose`이며 그main의기존변경은보존했다. 이작업은새namespace만쓴다. 기존통합all/verify는리프터를포함하므로호출하지않는다.

```bash
cd /tmp/pallet-pose-n3-v3
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.compute
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.paired
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.audits
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.pose_trace
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.uncertainty_extra
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.subgroups
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.paper
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.figures
/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.finish
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_n3_static_closeout_v1.runtime_yolo
```

실제실행은compute→paired/audits→runtime_yolo→pose_trace→uncertainty_extra→subgroups→paper→figures→finish순이었다. 파일이완료되어있으면재실행할필요가없다. compute와runtime_yolo는완료결과를재사용한다. 그밖의감사/내보내기는결과재생성명령이므로수정이나검증필요가있을때만실행한다. 신규학습/optimizer호출은없다.

검증: 기존test_metrics/test_evaluation/test_reuse/test_square/test_square_yolo와test_adapters::ConstantCheckpointTests를실행했다. 최초로그와경로복원뒤실패3개재실행로그를함께저장했다. GPU는승인된기존장비에서단일프로세스로계측했으며의존성을설치/업그레이드하지않았다.
