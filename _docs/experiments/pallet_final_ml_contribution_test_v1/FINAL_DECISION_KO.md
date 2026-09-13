# 최종 의사결정

시작 commit: `a9640cbf9a1ca2a7aab323258577a49b15bbf3d0` (`main`). 기존 결과를 보존한 별도 최종 감사/대조 실험이다.

## A: task-risk failure 감사

`TASK_RISK_AL_NO_SIGNAL` 유지. 기존 QA flag가 평가145장 전부에 있어 QA_CLEAN-only S1은 NOT_ESTIMABLE이다. QA flag를 근거로 문제 프레임 하나만 제거해 성공으로 바꾸지 않는다. 감사 판정은 `TASK_RISK_RESULT_QA_SENSITIVE_REQUIRES_CAUTION`이다.
문제 프레임은 원본640×480 밖 x≈698–740의 padding-region 검출이 세 seed 모두 최고 score로 선택되었다. 2D 점 오차는 약286px이고 두 기존 치수 가설 모두 약27–32m 위치 오류다. 선택된 위치의 오류가 이미 존재하고 PnP에서 증폭되며, C2 순서·축 선택만으로 회복되지 않는다. 후보 교체/selector/padding 필터는 적용하지 않았다. Reference provenance는 제한사항이나 제안 모델만의 실패가 GT 오류 때문이라는 근거는 아니다.

## B: 마지막 controlled ML architecture 비교

판정: `LOCAL_REFINEMENT_ONLY_SIGNAL`. Primary L−P=+0.165021px, 13-session95% CI [-0.009842, +0.381123]. Gates: {'G1': False, 'G2': True, 'G3': False, 'G4': True, 'G5': False}.
R0/L을 재학습하지 않았고 P만 seeds1/2/3×6000steps 학습했다. Synthetic 선택 후 실제 DEV319+negative2689 추론, 기존 evaluator, paired10k bootstrap, 기전 및 runtime 감사를 수행했다. 상세: `B_line_vs_point/REPORT_KO.md`.

다음 행동은 현재 line-specific architecture contribution 주장을 종료하는 것 하나다. 추가 DHT/Hough/selector/active score/self-training/module 탐색은 허용되지 않는다. Generic P의 novelty도 별도 입증 없이 주장하지 않는다.

Git은 main-only로 새 code/docs만 commit/push한다. 최종 commit의 자기참조 SHA를 보고서에 억지로 기록하지 않고, push 후 raw `GIT_PUSH_RECEIPT.json`과 CLI에서 local/origin/main 정확 일치를 확인한다. 기존 capacity_screen 미추적 폴더는 본 커밋에서 제외한다.

Local refinement is useful, but the current evidence does not establish a line-specific inductive-bias contribution.
