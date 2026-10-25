# V252 공개 저장 direct-reference 추론 검산

NumPy와 PyTorch가 필요합니다. 최종 package에 담긴 wrapper를 실행합니다.

```bash
python -B /absolute/package/artifacts/V252_PUBLIC_REPLAY_A/verify_saved_direct.py \
  --public-repo /absolute/public/repository \
  --package /absolute/final/package \
  --entry /absolute/late/public-replay-entry.json \
  --output-dir /absolute/fresh/replay-output
```

Root가 SOURCE 두 최종1000 checkpoint·로그·bank 봉인과 독립 QA 후 `ENTRY_FALSE_TEMPLATE.json`의 별도 true 사본에 실제 package 상대 경로/SHA/size를 채웁니다. false entry는 payload 전에 거절합니다. 기존 SOURCE 입력9개는 `runtime_dependencies.json`의 공개 상대 경로만 사용하며, 원 local 절대경로는 따라가지 않습니다. SHA/size와 both-final/DEV barrier를 검증한 뒤에만 숫자·checkpoint를 읽습니다.

B의 동결 CODE에서 direct1410 head/입력조립/저장 generic bank 기반 cosine helper만 AST로 골라 types.ModuleType+exec(compile)합니다. CONTROL은 저장 generic prototype으로 reference9를 계산하고 TREAT는 동결 V251 target reference9 cache를 그대로 사용합니다. 두 모델 모두 원 condition32를 붙인41차원만 받습니다. generic bank를 새로 적합하거나 optimizer/학습을 실행하지 않습니다. 각 checkpoint의1410 파라미터·공유 초기값·초기 final-layer0·최종 state SHA를 검증합니다.

원7617행 두 logit은 사전 허용오차 atol=rtol=1e-5, 선택 family prediction은 모두 정확히 같아야 합니다. BASE는 이미 공개된 CSV의 원 row join만 확인합니다. 기존 core CP/모듈/hidden/logit forward는 없습니다. 실패와 부분 output은 보존하고 자동 재시도하지 않습니다.

이는 노출된 SOURCE DEV에 대한 저장 추론 재현이며 물리적 native gold·실사 개선·독립 일반화 증거가 아닙니다. SYN intended-native-address와 YCB associated-r AUX/nativeUNKNOWN 권한은 유지합니다. generic cosine3의 두 번 사용은 의도된 입력이며 잘못된 물체/negative를 인증하지 않습니다. 이번 CODE 준비에서는 실제 배열·CSV 본문·CP·모델을 읽거나 실행하지 않았습니다. 최종 package 실제 replay는 Root가 한 번만 수행합니다.
