# 기대 부품·구조 단서 감독의 최소 계약 — V93 V2 초안

**상태: 문서 초안. 새 라벨·pixel·render·모델·fit·private 접근0.** 목표는 초기 q 중심 crop에서 **대상 팔레트의 기대 reference/구조 단서**가 보이는지 판별하는 것이다. geometry 부품 분리나 보조과제 통과만으로 native8 목표 해결을 선언하지 않는다. part152의 실제 물리적 의미와 asset의 팔레트 여부도 후속 render 검토 전에는 UNKNOWN이다.

1. **query-level 정의와 같은 연산자.** 원래 `query_id,q,L,target,expected_reference`를 보존한다. `expected_reference`는 검증된 native convention 또는 GT-free token이며 실제 GT r좌표를 모델 입력에 주입한다는 뜻이 아니다. 출력은 이 기대 단서에 대한 visible support와 typed U이며, foreign/self 원인은 해당 증거가 있을 때만 별도 제공한다. 학습·VAL·실사용은 같은 q 중심 crop·같은 scalar를 읽어야 한다. `GT r/uncertainty 점수 → mosaic 점수 → q-centre 점수`를 서로 바꾸지 않는다. r은 loss/평가의 reference일 뿐 crop·출력 위치를 정하는 입력이 아니다. q OOB·입력 무효는 입력 availability 사유다. reference MISS/PARTIAL은 r/기하가 있는 source·eval에서만 산정하는 GT-only availability이며 원reference 상태를 E로 바꾸지 않는다. 실사용에서는 GT r을 읽어 U를 정할 수 없고, 입력 유효성·학습된 confidence·검증된 GT-free evidence로만 abstain한다.

2. **정체와 ownership이 먼저다.** `target instance × expected cue/reference`의 정의가 필요하다. 이를 원래 slot convention으로 검증했거나 GT-free target 정보로 식별할 수 있어야 한다. 여러 팔레트·반복된 발 중 어느 것을 기대하는지 입력으로 구별할 수 없다면 원리적으로 U다. 동일 crop에 서로 다른 기대 부품의 상반된 정답을 주지 않는다. 새 asset-local partID와 native8 연결은 검증 전 UNKNOWN이며, 가까운 mesh/보이는 모서리로 r을 snap하거나 nativeindex를 재명명하지 않는다. 연결이 없는 자료는 **별도 physical-part auxiliary**만 감독할 수 있다.

3. **wholeboard와 cue를 분리한다.** Topboard 전체에서 일부 pixel이 보인다는 사실은 특정 끝점·발/구멍 접합선이 보인다는 뜻이 아니다. 감독할 cue의 구조·support 범위·가림 판독 규칙을 모델 결과 전에 선언해야 한다. 예컨대 뒤쪽 아래 reference가 위쪽 deck 위로 투영되면 목재는 보이지만 기대 아래 reference가 보이는지는 모른다. 옆 팔레트의 판재가 보이거나 q가 기대 발 옆 바닥에 놓여도 D로 바꿀 수 없다. 물리 cue가 근처에 있다는 사실과 exact pixel membership도 구분하며, virtual reference를 반드시 mesh 표면점으로 제한하지 않는다.

| 증거 | 만들 수 있는 감독 | 자동 승격할 수 없는 것 |
|---|---|---|
| instance·part ID가 고정된 `P_A`(part 단독), `P_S`(target 전체), `P_V`(foreign 포함) | part별 projected/self-hidden/foreign-hidden mask 및 연속 fraction | human/native D/E/S, 특정 corner visibility |
| 공개 before에서 동일 target·cue의 고정 r/uncertainty를 확인하고 foreign after가 그 cue를 실제 덮음 | 근거 tier를 붙인 조건부 physical-cue external 보조라벨 | 자연 E와 controlled paste의 혼합, native8 대응 |
| whole-pallet M4 양성·목재재질·foreign surface·어두운 구멍만 관찰 | pixel/appearance 진단, 모호함 U | D 또는 E/S 정답 |

4. **합성 proxy의 한계와 부품별 구별.** face 소유자는 `(asset_instance_id,asset_local_part_id)`로 관리한다. `P_A`가 **certain selected part**이고 `P_S`가 **certain other target part**일 때만 SELF proxy다. `P_S`가 **certain selected part**이고 `P_V`가 **certain foreign owner**일 때만 FOREIGN proxy다. `P_V`가 **certain selected part**이면 VISIBLE이며, 그 외 background·missing·경계/owner ambiguity는 U다. 단순 `P_A & ~P_S` / `P_S & ~P_V` 차집합은 이 조건을 대신하지 못한다. 다른 target 부위의 ID는 기대 part의 visible positive가 될 수 없다. 다만 이는 고정 opacity·depth·camera 조건의 합성 proxy이며 human S/E가 아니다. zero-area·불명 opacity·missing payload·ambiguous cue는 U로 보존한다. fraction의 임계값을 D/E 의미로 사후 선택하지 않는다. 기존 convex-self/ray proxy와 같은 이름의 gold로 합치지 않는다.

5. **공개실사–합성 bridge의 최소 훈련 조건.** 실사와 합성에서 같은 target/cue 정의·q/L 분포·입력 연산자·U 규칙을 사용한다. 공개 before의 ownership·cue 대응을 독립적으로 고정하고, 동일 q/L의 opaque cover와 cue를 보존한 sham을 구분한다. after RGB와 특징을 함께 다시 만들고 일부 부위만 가렸다고 모든 query를 E로 만들지 않는다. 실사 자연가림·controlled foreign·합성 self를 다른 evidence tier로 저장하며, 기존 GT를 봤던 노출과 GT-jitter q라면 emulation임을 명시한다. cue·instance·capture family·before/after·donor를 같은 partition에 묶고 source-heldout을 둔다.

6. **평가의 최소 분모와 진입 조건.** 원래 모든 query와 U·wrongtarget·MISS를 유지하고, known 기대 단서의 성능과 전체 availability/보류율을 따로 낸다. instance/cue별 중복과 before/cover/sham·자연/합성·self/foreign을 구분해 같은 연산자로 채점한다. 라벨·split·selector·문턱은 새 forward 전에 고정하고 현재223을 fit·문턱·모델 선택에 쓰지 않는다. source proxy 통과 뒤에도 native reference 대응과 자연가림 근거가 없으면 해당 목표는 여전히 미해결이다. 현 단계는 geometry 의미·관측 가능성·라벨 일관성 검토가 먼저이며 새 fit을 시작하지 않는다.
