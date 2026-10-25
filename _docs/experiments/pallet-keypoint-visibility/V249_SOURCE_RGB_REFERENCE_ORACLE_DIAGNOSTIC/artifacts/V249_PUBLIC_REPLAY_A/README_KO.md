V249 공개 SOURCE full-CPU replay

이 wrapper는 실행 준비 CODE이며 아직 실행하지 않았습니다. NumPy가 설치된 Python과 기존 공개 저장소 checkout만 필요합니다. 모형/가중치/RGB/원 GT/optimizer를 사용하지 않습니다.

패키지의 artifacts 하위 sibling 구조를 유지합니다. V249_PUBLIC_REPLAY_A/replay_public.py와 bundled dependency_map.json, 옆 V249_RGB_REFERENCE_DIAGNOSTIC_B/diagnose_source_rgb_references_v2.py가 필요합니다. map SHA95a379fb…와 V2 CODE SHA01f0f81d…는 코드에 고정되어 있습니다.

```sh
python -B /absolute/package/artifacts/V249_PUBLIC_REPLAY_A/replay_public.py \
  --public-repo /absolute/pallet-visibility-publication-20261024 \
  --output-dir /absolute/fresh/V249_CPU_REPLAY
```

기존 공개 SOURCE3파일의 repository_relative_path/SHA/size만 사용합니다. original_local_path는 provenance이고 절대로 fallback하지 않습니다. Git 실행/네트워크도 필요하지 않습니다. 세 파일과 진단 코드를 먼저 모두 검산한 뒤 NPY feature/NPZ5members/CSV선택14fields를 읽고 정확한 run_diagnostic 및 write_results를 호출합니다. CSV transport 중 다른 문자열 열은 버리고 숫자 해석·진단에 사용하지 않습니다.

출력은 B pure API의6개 산출물과 PUBLIC_REPLAY_PROVENANCE.json/COMPLETE.json입니다. teacher object ID로 선택한 정답 prototype은 oracle 진단이지 배포 때 쓸 수 있는 reference retriever가 아닙니다. 원 SOURCEDEV는 이미 노출·재사용된 내부 개발 자료이고 native gold/실사 성공/인과 증명이 아닙니다. 새로운 best-view/threshold/모형·data 선택이나 학습은 하지 않습니다. 새 output만 허용하며 overwrite/automatic retry는 없습니다.

원 absolute source paths, current/held·평가 입력과 v112/cache/pyc·symlink는 허용하지 않습니다. 실제 replay는 Root의 primary actual 진단 후 별도 승인으로 수행합니다.
