# 저장 SOURCE branch 산술 검산

표준 Python만 사용합니다. 새 모델 forward·학습은 없습니다.

```bash
python -B verify_saved_branches.py --saved-dir <복사된_actual_diagnosis_v1>
```

검산기는 그 폴더의 COMPLETE.json 및 고정된 5 output basename만 읽습니다. COMPLETE 내부의 checkpoint/feature/원본 absolute 경로는 따라가지 않습니다. 원문 SHA/size 확인 뒤 7617 CSV행의 FP32 additive sum, 두 binary family logit 중 해당 family의 0 문턱, signed residual, 회복/악화, CM/P/R 및 분포/CSV strata를 재계산하고 stdout JSON 하나를 냅니다. `--output-directory <새폴더>`를 쓰면 REVIEW.json을 별도로 저장합니다. Root는 동일 tree를 package에 보존하거나 이 CLI를 얇게 감싸 재사용할 수 있습니다.

saved TOTAL의 전체15234 logit CPU 재생은 upstream QA receipt를 인용합니다. 이 검산기는 각 행의 두 own-branch 합과 selected saved TOTAL delta/sign을 독립 확인하며 실제 checkpoint/특징/원 샘플을 다시 검증한 것으로 주장하지 않습니다. 수치 통계의 비교 오차 1e-10은 저장 scalar 통계 산술용이고, frozen feature byte 공유 및 기존 CPU replay 기준을 바꾸지 않습니다.
