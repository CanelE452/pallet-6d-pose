# V231 기존 head·라벨 소비 경로 점검

새 SOURCE 학습·설계 없이 기존 13개 코드와 SOURCE 실행/설계 metadata만 대조했습니다. 핵심 사실은 이미 V230의 독립 consumer 점검에서 확인된 내용이며, 새 YCB 감독 발견이 아닙니다.

| 방법 | 실제 목표·SOURCE | 입력·head | loss·실제 실행 근거 |
|---|---|---|---|
| C10/V110 | YCB associated-r D/E; logical slot의 pallet native는 UNKNOWN. 기존 old/synth도 결합 | frozen ResNet18의 q8-context/local96/center32, 1541→관계항2565→64→1 | YCB D8/E12 포함 balanced64 BCE + old/synth COVER−SHAM margin2·.25. completed MODEL_FREEZE와 기존 C의 2000-update closure |
| C11/V111 | C10의 같은 라벨·pool·draw | frozen DINO T/Q/F/C/P. 원 q의 context patch bilinear read; 1928→관계항3464→64→1 | 동일 BCE+hinge 및 draw. completed MODEL_FREEZE와 기존 C의 2000-update closure |
| C65/V219 | SYN own-image point6096(D5853/E243) + public expected finite-part weak FIT45(D39/E6), DEV12(D9/E3). natural YCB pool 없음 | scratch RAW96 CNN118359. native/mask/q8는 pool/shared64 뒤 residual로 결합. crop밖 RGB 없음 | old5 AUX 평균 + point BCE32 + .25 weak BCE16. 실제4000×2 완료. V219에 새 finite-part CF pair hinge는 없음 |
| V227 fixedQ/attention | 원 MR6D query의 기존 role-weak368. FIT65/DEV303 고정, YCB 없음 | frozen DINO local/read/CLS/cond32→1184 linear. 고정 Q vs 조건·상대좌표 attention; 각4561 | fullFIT balanced BCE+.001 L2, 실제300×2 완료. paired CF/ROI auxiliary 없음 |

YCB는 TRAIN1000 instance·700frame·8000slot이며 declared D1181/E340/S1917/U4562입니다. q는 r에 기존 ≤4px jitter를 더한 emulation입니다. C10/C11은 알려진 D/E와 available·TRAIN의 교집합만 BCE에 쓰며 S/U를 E로 바꾸지 않습니다. 자연 YCB 자체를 matched COVER/SHAM triplet이라고 선언하지도 않습니다.

C10/C11은 ImageNet RGB 전처리와 descriptor L2를 사용하고 fitted normalizer가 없습니다. C11은 raw patch를 bilinear sample한 뒤 Q를 L2 정규화합니다. V227은 token별 L2 후 sample하고 다시 L2합니다. C65는 RGB/255와 GroupNorm이며, owner class 가중치는 TRAIN·nominal·nonDEV·lineage mask 뒤에만 재정규화합니다.

C10/C11은 코드상 기존 VAL cohort로 best checkpoint를 선택했습니다. 그 정책만 확인했고 VAL 값은 열지 않았습니다. C65와 V227은 각자 고정된 양 final 뒤에 DEV를 산출했습니다. V227 DEV는 재사용·노출된 SOURCE로 fresh blind가 아닙니다.

따라서 **natural YCB D/E 직접학습, frozen RGB context, 원 q patch read, COVER/SHAM hinge, fixedQ 대 attention을 처음 시도한다는 주장은 성립하지 않습니다.** 반면 이 방법들의 데이터·target 권위·head·normalization·split이 달라 완전히 같은 실험이라고 할 수도 없습니다. 같은 scalar에 서로 다른 목표 정의가 들어간 구조는 실패 원인 증명이 아닙니다. 새 head 제안이나 성능 기반 후보 선택은 하지 않았습니다.

정확한 13개 코드 SHA·라인, 읽은 metadata와 실제 학습 근거의 직접/기존검토 구분은 REVIEW.json에 있습니다. 개별 draw·gradient·feature/weight/score는 재생하지 않았고, 새 array/이미지/GT/모델/학습/PnP 및 보호 score 접근은 0입니다. YCB native UNKNOWN이나 약한 finite-part 권한을 pallet native gold로 올리지 않으며, 완벽한 physical gold를 모든 AUX의 필수 조건으로 새로 추가하지 않습니다.
