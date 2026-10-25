# V233 common-local-crop paired adaptation 준비

원 C11 전체 실험 재현이 아닙니다. full96 CLS384 + 같은 full96 rawpatch의 고정 centre-Q384 + centre32 CLS384 + validfractions2 + condition32 =1186을 사용합니다. Q는 central4로만0 gate하고, 두 CLS는 각 fraction>0으로 gate합니다. available·lossmask를 추가 삭제하지 않습니다. SYN의 기존 condition32를 유지하고 YCB는 native UNKNOWN이므로 전부0입니다.

동일 `1186→64 ReLU→2` head(76098개, final0)를 양팔에 복제합니다. output0은 SYN intended-native witness0, output1은 YCB associated-r surface로 두고, 각 family 자기 출력에만 BCE를 적용합니다. loss는 두 family meanBCE의 평균입니다. Control base는 eval/frozen이며 treatment만 마지막 fused qkv rank4 LoRA6144개를 추가 학습합니다.

PCG64seed17로1000×32 draw를 미리 고정합니다(각family8D+8E). AdamW headLR.001/adapterLR.0001/wd.0001, FP32 mathSDPA·TF32off,1000updates씩입니다. FIT known+available sample이 참조한 prefix row만 먼저 fetch합니다. 전체prefix SHA는 opaque stream으로 검증하되 수치 body는 memmap 선택행만 해석합니다. 양 final1000 checkpoint와 freeze를 저장하기 전 DEV forward는 없습니다. 이후 SOURCE CM/P/R/U는 family·FIT/DEV별로 구분합니다.

`load_backbone(config,device,treatment=False)`가 Root prefixbuilder에도 제공됩니다. 공식 state strict-load 후 FP32→eval/freeze→선택적 seed17 LoRAwrap 순서입니다. 실제 config의 asset/code/payload path·SHA와 vendor revision을 Root가 주입해야 합니다. trainer는 원 rawRGB/원GT/oldSOURCE NPZ를 읽지 않습니다. Foundation base는 새 PT에 복사하지 않고 공식 binding과 base hash를 기록합니다.

가짜 prefix·metadata·fake backbone 통합 smoke가 PASS/exit0입니다. 1186 동등성, loss routing, descriptor gate, paired draws, DEV fetch 차단, frozenbase/LR groups를 확인했습니다. 실제 DINO·state·SOURCE/cache 접근, optimizer step·실학습은0입니다. 실제 모델 loader·1000step loop·asset spotcheck는 실행하지 않았습니다. 데이터 admission과 실제 실행은 Root의 별도 trueentry가 필요합니다.
