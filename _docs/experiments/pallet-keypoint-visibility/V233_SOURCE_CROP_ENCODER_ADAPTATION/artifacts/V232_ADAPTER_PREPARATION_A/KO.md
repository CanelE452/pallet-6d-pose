# V232 module-only 준비

pinned DINO ViT-S/14 reg4는 dim384·12block·6heads이며 마지막 `blocks[11].attn.qkv`는 fused Linear384→1152입니다. 이 한 Linear에 rank4 `B(A(x))`를 더하는 torch-only wrapper를 준비했습니다. B=0이므로 초기 출력은 원 Linear와 같고, 추가 trainable parameter는6144개입니다. 실제 vendor model/weight는 생성·로드·forward하지 않았습니다.

C11의 **raw context tokens → 원 q bilinear(border, align_corners=False) → L2**를 보존했습니다. 1928 특징 조립은 고정된 T/Q/F/C/P+8scalar이고, scalar `3464→64→1` class는 원 C11 AST와 같습니다. tokenwise pre-L2, learned attention, q 보정, fitted normalizer를 추가하지 않았습니다.

Root 지시에 따라 dual-target head/타깃명명/공유방식/trainer는 보류했습니다. 원 q·instance·reference의 감독 의미와 실제 SOURCE 계약을 이 module이 결정하지 않습니다.

random tensor·fake Linear12block만의 smoke가 exit0/PASS입니다. 초기 identity, A/B gradient, frozen base 제외, fixed-Q/1928 assembly의 pinned C11 결과 일치, scalar AST·초기0출력을 검증했습니다. B=0에서는 A gradient가0이며, scalar 마지막층도0이면 최초 분류 backward에서 adapter gradient 역시0일 수 있다는 단계적 초기화 한계를 기록했습니다. 가짜 B수동변경 검사는 gradient 경로만 확인한 것이며 optimizer step·학습은0입니다.

실제 pretrained state는 미래 Root entry에서 wrapper 삽입 **전** strict-load해야 합니다. base는 eval/frozen을 유지하되 adapter까지 gradient가 필요한 future forward를 inference_mode/no_grad로 감싸면 안 됩니다. 이 준비는 실제 설계 승인·데이터 접근·학습 입장이 아닙니다. 정확 API·code SHA·source lines·제한은 MODULE_CONTRACT.json에 있습니다.
