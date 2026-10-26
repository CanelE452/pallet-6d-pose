# T1(b) 미실행 근거

[확인] T1(b)는 실행하지 않았다. 요청한 **400×400 squash + 7×7 soft-argmax(T=0.1)** 조합이 R0 학습 당시 추론 경로와 동일하다는 근거를 확인하지 못했다. 요청서 30–31행의 조건에 따라 유사 경로를 임의로 구현하지 않았다.

[확인] 기존 코드는 존재한다. 400×400 비등방 squash의 학습·평가 전처리는 기록과 현재 Q1 코드가 뒷받침한다. 그러나 INVENTORY의 7×7·T=0.1은 DiffPnP3D **학습 보조 손실** 항목이고, D0/P0는 최근 진단 결과의 parity 경로라는 설명이다. 두 사실만으로 원래 추론 경로의 동일성을 인증할 수 없다.

- **요청의 미실행 조건**: `/home/minjae/.codex/attachments/26cfa37e-bef2-4ab6-9ebe-548345126331/goal-objective.md:30–31` — 학습 당시와 같은 추론 경로라는 근거가 없으면 T1(b)를 실행하지 말고 유사 구현을 만들지 말라는 명시 조건이다.
- **원 학습 기록**: `/home/minjae/Documents/github/pallet-pose/data/pallet/results/paper_s2_scratch_diffpnp/diagnostic_audit/INVENTORY.md:16,63–88,109–111` — 400×400 squash와 R0 계보는 기록되어 있으나, 7×7·T=0.1은 DiffPnP3D 학습 보조 손실 항목이다. 원래 추론 디코더라는 인증 근거가 아니다. 일부 학습 코드가 미커밋이라는 경고도 있다.
- **현재 D0/P0 의미**: `/home/minjae/Documents/github/pallet-pose/scripts/stage0/decoder_paths.py:3–14,35–41,63–65` — D0는 최근 eval56/wood 진단에 쓰인 mechanism decoder이고 P0는 그 진단 결과의 parity 경로이다. 해당 설명만으로 R0 학습 당시 공식 추론 경로와 동일하다고 확정할 수 없다.
- **디코더 비교 문서**: `/home/minjae/Documents/github/pallet-pose/_docs/audits/eval56_summary/decoder_reconciliation/DECODER_COORDINATE_COMPARISON.md:27–30` — D0가 7×7 soft-argmax(T=0.1)를 사용함을 뒷받침하지만 원 학습 당시 추론 경로의 인증은 하지 않는다.
- **실제 현재 Q1 평가 경로**: `/home/minjae/Documents/github/pallet-pose/scripts/stage0/diffpnp3d/diffpnp3d_q1_eval.py:4–12,51,76–81,137–140` — 400×400 squash+ImageNet 정규화의 기존 train/infer parity 평가 코드는 존재한다. 좌표 추출은 filter_pr_camfacing.extract_keypoints_from_belief를 호출하며 이 호출부가 7×7 soft-argmax를 고정하는 근거는 아니다.
- **기존 선택기**: `/home/minjae/Documents/github/pallet-pose/scripts/stage0/diffpnp3d/diffpnp3d_val_select.py:4–9,99–104` — 원 계보의 선택기는 Q1E.eval_frame을 재사용한다고 명시한다.
- **기존 실행 스크립트**: `/home/minjae/Documents/github/pallet-pose/scripts/stage0/_run/diffpnp3d_full_run.sh:9–10,81–89,99–104` — 학습 imagesize400 및 DiffPnP 온도 0.1과 실제 nested selector 경로를 구분할 수 있다.

[확인] INVENTORY 109행의 오래된 호출 경로 `/home/minjae/Documents/github/pallet-pose/scripts/stage0/diffpnp3d_q1_eval.py`는 현재 존재하지 않는다. 실제 현재 코드는 `/home/minjae/Documents/github/pallet-pose/scripts/stage0/diffpnp3d/diffpnp3d_q1_eval.py`에 있다. 경로 이동 자체가 원래 디코더의 동일성 근거가 되지는 않는다.

[확인] 이 판정은 텍스트 코드와 메타 문서만으로 수행하였다. 모델 forward·학습·RGB·정답 본문 열람은 모두 0회이다. 파일 SHA와 판단은 T1B_SKIPPED.json에 보존하였다.
