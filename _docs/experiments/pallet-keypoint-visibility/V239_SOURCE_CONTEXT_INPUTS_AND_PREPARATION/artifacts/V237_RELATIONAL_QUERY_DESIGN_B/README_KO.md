# V237: 원 query 관계 읽기의 최소 CODE 안

Root가 다음 V238 SOURCE 준비안으로 선택한 순수 모듈입니다. **실제 feature·영상·가중치·학습은 실행하지 않았습니다.**

| 경로 | 이미/새로 명시되는 영상 증거 |
|---|---|
| V233/V235 | 원 active RAW96 fullCLS·centre32CLS·중앙 fixedQ + support/condition32. crop 밖 RGB는 없음. |
| 기존 C11 | q8 extent의 targetCLS·active contextQ·local full/centre CLS·localQ 및 cosine/support. fullcontext와 active/context 비교는 이미 했음. |
| 이번 안 | 기존1186 base를 그대로 두고, 같은 actual RGB에서 원 qj/Lj로 얻은 **다른7 local fullCLS**를 active CLS와 명시 비교 + targetCLS residual. |

새 branch는 shared384→16 projection과 원 상대XY/L의2→1 bias로 active-vs-other7 점수를 만들고, 원 input availability와 pixel support만으로 masked softmax합니다. active/message/difference/product/context + 기존condition32 +4supportflags를116→32→2로 읽습니다. 마지막2logit layer는0으로 초기화하므로 시작 때 기존 base 출력과 정확히 같습니다. 파라미터는 base76,098 + residual9,956 = **86,054**, tensor10개입니다. encoder/LoRA는 추가하지 않습니다.

active CLS는 기존1186의 앞384개를 사용합니다. 새 q8 node의 active CLS는 별도 정합 함수로 비교하며 불일치를 덮어쓰지 않습니다. 기본 tolerance1e-6은 실제 추출 전에 Root 프로토콜로 고정해야 합니다. Base는 원 V233 **initial**4tensor를 strict-load해야 하며 기존 final head에서 미세조정하는 안이 아닙니다. Zero-init 때문에 첫 backward의 초기 projection/attention gradient가0인 것은 정상입니다.

C11 q8 extent의 endpoint는 context patchgrid 경계에서 border read되며 이는 선언된 embedding 연산입니다. 이 안은 각 qj를 자기 Lj RAW96 중앙으로 읽으므로 다른 query의 증거를 그 endpoint 한 토큰에 맡기지 않습니다. 그래도 image-edge padding, q8 extent 밖 정보, 잘못된 target/초기점 대응을 해결했다고 주장하지 않습니다. 기존 context의 반복·추가 capacity·새 read 경로가 함께 바뀌므로 gain이 나도 relational 계산만의 인과 효과로 분리되지는 않습니다.

Mask에 다른 점의 D/E/S/U·owner·GT instance/target ID를 쓰지 않습니다. q8/relativeXY/native/availability는 원 입력 권한이며 GT ref·pose·depth·mask·part annotation은 forward0입니다. SYN actualimage와 anchor는 구분해야 하고, q는 anchor에서 왔더라도 RGB는 해당 실제 관측 scene이어야 합니다. YCB q8은 SOURCE GT-jitter emulation이며 native UNKNOWN/aux는 논리 role입니다. 원U는 fixed0 decoder에서 그대로 U이고 새로운 confidence/semantic label을 만들지 않습니다.

Root의 예정 첫 trial은 natural SOURCE/원1000 PCG17draw/기존 family loss·고정encoder를 유지하고, 원 CONTROL1000 bytes를 재사용하며 TREAT만 원 initial base+residual로1000 update하는 안입니다. 새 V235 CF는 첫 trial에 넣지 않습니다. 이것은 Root 결정의 기록이며 이 CODE 산출물은 feature/fit 실행 권한이 아닙니다.

[모듈](relational_query_head.py) · [API/권한 계약](MODEL_AND_INPUT_CONTRACT.json) · [가상 CPU 검사](INVENTED_CPU_CHECKS.json)
