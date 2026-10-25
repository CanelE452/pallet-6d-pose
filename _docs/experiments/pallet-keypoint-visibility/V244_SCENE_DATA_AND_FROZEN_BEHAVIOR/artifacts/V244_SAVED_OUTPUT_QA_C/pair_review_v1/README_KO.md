# V235 원 FIT pair 저장 산술 검산

```bash
python -B verify_saved_pairs.py --saved-dir <복사된_v235_pair_replay_v1>
```

표준 Python만 필요합니다. COMPLETE.json과 ALL5406_OLD_PAIR_BEHAVIOR.csv, SUMMARY.json의 고정 local basename 3개만 읽고 원 SHA/size를 확인합니다. 원문 내부의 feature/encoder/checkpoint/샘플/GT 절대 경로는 따라가지 않습니다. stdout JSON 하나를 내며 `--output-directory <새폴더>`를 지정하면 REVIEW.json을 별도로 저장합니다. Root wrapper에서 own-branch verifier와 각각의 saved-dir로 호출할 수 있습니다.

검산 대상은 전수 행 identity/family/FIT, 실패39의 null/U/target−1, 성공5367의 두 family FP32 logit와 해당 family scalar의 고정0 문턱, COVER−SHAM 및 원 natural D 대비 delta, family metrics/중앙값/방향/보존율입니다. 실제 encoder/head replay나 원 teacher 정당성은 새로 인증하지 않습니다.
