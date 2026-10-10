# Feature-Guided Joint Keypoint Refinement (2026-10-10)

실행 상태 **COMPLETE**, 과학적 판정 **TRADEOFF**. 새로운 공동 위치 계산을 기존 N3→SubPix와 비교했다. 동일한 DEV 319장과 3 seed를 사용했으며 추가 학습은 0회다. 직렬 대비 위치 오차 평균과 중앙값, ADDsym-AUC가 악화했고 회전 평균이 줄었다. 두 평균 차이의 95% CI가 모두 0을 포함하므로 확정 개선으로 선언하지 않는다.

[한국어결과](RESULT_KO.md) · [정확한수식/계산](METHOD_KO.md) · [재현](REPRODUCE.md) · [수치CSV](METRICS.csv) · [JSON](METRICS.json) · [짝95%CI](PAIRED.json) · [실패/손상ID](FAILURES.json)

[원행](PREDICTIONS.jsonl.gz) · [실제222후보capture](POSTERIOR_CAPTURE.jsonl.gz) · [참조전좌표봉인](NEW_COORDINATES_SEALED.jsonl.gz) · [원입력lock](INPUT_LOCK.json) · [방법lock](FUSION_METHOD_LOCK.json) · [코드lock](CONFIG_LOCK.json) · [실행ledger](EXECUTION_LEDGER.json)

[단위검사](UNIT_TESTS.json) · [N3 parity](POSTERIOR_PARITY.json) · [독립 posterior 검산](POSTERIOR_NUMERIC_VERIFICATION.json) · [R/t 검산](POSE_NUMERIC_VERIFICATION.json) · [보조 자료 검산](PUBLIC_AUXILIARY_VERIFICATION.json) · [새 계산 증거](MECHANISM_PROOF.json) · [통계/그림 검산](VERIFICATION.json) · [그림 manifest](FIGURE_INDEX.json) · [전체 SHA256 manifest](SHA256_MANIFEST.json)

원본 RGB의 유통 권한이 확인되지 않아 좌표와 집계 그림을 공개했다. 좋지 않은 seed, 큰 오류, 좋은 코너의 손상, CI의 0 포함을 보고서에 유지했다. 논문, LaTeX, PDF는 수정하지 않았다.

| PNG | 내용 |
| --- | --- |
| [figures/01_joint_method.png](figures/01_joint_method.png) | One joint posterior/image equation versus existing sequential route |
| [figures/02_seed_pose_comparison.png](figures/02_seed_pose_comparison.png) | All6 methods, three seeds and per-image seed mean |
| [figures/03_paired_ci.png](figures/03_paired_ci.png) | Joint minus sequential / single methods / isotropic: session95%CI |
| [figures/04_per_grade_damage.png](figures/04_per_grade_damage.png) | Every difficulty grade, corner harm and fallback |
| [figures/05_qualitative.png](figures/05_qualitative.png) | Rule-selected coordinate-only improvement/worsening/nearzero/gross cases |
| [figures/06_covariance_vs_gradient.png](figures/06_covariance_vs_gradient.png) | Actual prior/gradient axes and output movements |

[새코드](../../../scripts/research/pallet_feature_gradient_joint_20261010/) · [실제joint](../../../scripts/research/pallet_feature_gradient_joint_20261010/fusion.py) · [단위검사](../../../scripts/research/pallet_feature_gradient_joint_20261010/test_fusion.py) · [그림생성](../../../scripts/research/pallet_feature_gradient_joint_20261010/figures.py) · [보고서생성](../../../scripts/research/pallet_feature_gradient_joint_20261010/report.py)
