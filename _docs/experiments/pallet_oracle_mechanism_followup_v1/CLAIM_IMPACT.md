# 기존 논문에 대한 영향 — 수정 제안, main 미수정

[확인] 기존 manuscript·PDF·주표·교사·checkpoint·GT의 입력 hash를 유지한다. 이 문서는 사후 연구의 원고 반영안이며 실제 main을 대체하지 않는다.

## 유지 가능한 주장

- 반복 DEV의 Plastic에서 matched corrected-target self-training은 raw-target student와 synthetic-only R0보다 PCK10 및 기존 D9 ADDsym AUC가 높았다.
- Wood에서는 corrected−raw의 기존 작은 양의 효과가 R0를 넘는 실용적 개선을 보장하지 않았다. 재료별 표를 유지한다.
- 수동9장·38점이 교사에 사용됐음을 공개한다. C3에서 같은 점을 학생에게 직접 제공하는 경로는 감독 예산의 **대안적 사용**이지 추가 어노테이션이 없는 완전 무감독 실험이 아니다.

## 사후 진단에서 추가 가능한 설명

> Post-hoc analyses on reused development data revealed complementary information among frozen teacher and student outputs. For the corrected Plastic student, selecting the best existing width/depth pose candidate using the reference would increase normalized ADDsym AUC from 0.3590 to 0.4501. This diagnostic headroom is not deployable performance. Replacing the residual score with a fixed Huber score did not change candidate choices, and matched real-only affine-removal training did not improve the previous corrected student on either material's primary metrics.

> Native training-image residuals indicated partial imitation of corrected targets rather than complete failure of signal transfer. However, target imitation does not establish physical coordinate accuracy. The largest augmentation-probe residual involved an instance-selection switch, limiting its interpretation as evidence of a general coordinate-regression defect.

> A separate support-matched control directly reused the teacher's existing nine manually supervised training images and 38 clicked corners. Direct-coordinate training improved in-sample PCK10 from 32/38 to 35/38 relative to the raw-coordinate control, while three large Wood errors remained. Development-set effects were material-dependent: +5/985 correct Plastic corners but −22/346 Wood corners at 10 pixels. Both results remained below the respective previous corrected student. This finite, mixed-material training control neither bounds full-data learning nor establishes physical pose accuracy.

## 반드시 유지할 한계

독립 확인 없음; 반복 DEV; geometry-derived pose reference; Wood 직접 클릭 평가점 출처 미확인; small/clustered recording 수; 단일 randomness 설정의 작은 변화; finite TRAIN capability가 전체실사 정답학습 상한이 아님. C3 결과는 별도 사후 대조로만 인용하고 주표를 치환하지 않는다.

## 사용하면 안 되는 표현

`oracle achievable accuracy`, `all material generalization`, `all augmentation is harmful`, `teacher confidence measures corrected-coordinate correctness`, `Wood contains no useful information`, `representation capacity proved sufficient/insufficient`, `verified66 +1 means resolved`.

기존 원고를 실제 수정하지 않았으므로 새 PDF 빌드도 필요하지 않다. 종전 TeX/PDF hash 보존과 새 Markdown의 링크·숫자·그림 감사를 수행한다.
