# V238 SOURCE context residual trainer

원 V233 CONTROL_LOCAL_CROP final1000 PT/log/CSV를 바이트 재사용하며 새 CONTROL 학습은0이다. TREAT_CONTEXT_RESIDUAL은 원 initialbase76098+zero-final residual9956(총86054)을 seed17, 원 PCG17 1000×32 draw, AdamW .001/wd.0001, family별8D8E의 평균 BCE로1000회 학습하는 코드다. 새 V235 CF는 쓰지 않는다.

- 실행은 Root가 `ENTRY_FALSE_TEMPLATE.json`의 실제 scope·code/metadata/payload·officialasset whitelist를 채우고 승인한 뒤 `python -B train_context_residual.py --entry ROOT_TRUE_ENTRY.json`로1회만 한다. 현재 template은 false이며 데이터·가중치·feature·실제 DINO 접근0이다.
- V235 `load_sources/load_preprocess`와 V233 `features_for`/strict officialfactory를 재사용한다. old `run`·wholefitentry·혼합dataset producer를 호출하지 않는다. V108 `preprocess_target`은 고정 함수 AST만 재사용한다.
- geometry.node_available=원available & central4 & anyvalid를 실제valid에서 검증해 nodeCLS추출·residual mask에 같이 쓴다. S/U 라벨·owner·reference는 이를 결정하지 않는다. queryrelative는 원q/L F64→clip±8→F32이며 SYN원condition과 비교한다.
- FIT5876 descriptor(1e-6/rtol0/fullchecked) 통과 뒤 optimizer를 시작한다. 최종1000 PT/log/ledger와freeze 이후에만 DEV1741 정합검사 및 TREAT DEV output을 한다. CONTROL두logit7617 byteexact replay도 그 뒤 수행하며 실패시 자동재시작·문턱/허용오차 변경이 없다.
- 원 RGB/valid exact rebuild는 Root input admission/independentQA에 연결한다. 이 prep에서 새 pixel을 읽거나 재인증하지 않았다. DINO는 공식state strictload·frozen/math FP32, noLoRA다.
- 공개용 frozen feature3개: NODE_CLS.npy, CONTEXT_CLS.npy, postfreeze ORIGINAL_ALL_FEATURES.npy. 원 FIT featurecache는별도 ORIGINAL_FIT_FEATURES.npy이고 main에서 제외 가능하다. sample-group/node/relative maps도 보존한다. head replay에는 원availability/node_support/context_fraction을 가진 entry-bound INPUT_GEOMETRY.npz도 필요하다.
- SOURCE gate: family DEV balanced recall평균+3pp 이상, 모든D/E recall하락2pp 이하, YCB DEV Erecall>=50%. gate와무관하게 current_admission=false/newcurrent0이며 자동모델선택을 하지 않는다. SYNwitness0/YCBassociated-r(nativeUNKNOWN)는 별도출력이다.

산출물은 COMPLETE.json이 checks/checkpoints/logs/features/CSV/summary/freeze를 정확SHA로묶는다. CSV키는 sample_index,row_id,family,role,available,target,logit0,logit1,selected_family_logit,prediction다.

인공 검사7그룹 PASS (`9bc12b exit0`): initialbaseexact/zero residual, 86054/10tensor, 원PCG17 quota, qL/role/sample join 거부, F64relative/self/U0, 실제지원branch를 parameter-free mock으로통과, onebackward finite10grad, gatefloor/forbiddenlexical. 실제 optimizerstep0·DINOinstantiate/forward0·sourcepayload0이다. 저장 hook 추가 전 토이결과도 별도보존했다.

실제 실패는 freshoutput의 FAILED_EXECUTION.json과 Root terminal에 보존한다. output 생성 전 entry/metadata guard 실패는 Root가 terminal을 보존해야 한다. 원 prior code/phase는수정하지않았다.
