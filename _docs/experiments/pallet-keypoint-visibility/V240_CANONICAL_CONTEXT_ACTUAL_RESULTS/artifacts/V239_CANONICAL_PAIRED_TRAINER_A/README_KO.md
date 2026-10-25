# V239 canonical frozen-vector paired head preparation

준비 완료·미실행(CODE + invented toy only). 원V238은 descriptor검사에서optimizer0/DEV0으로종료했다. 그실패를학습결과로쓰거나1e-6를완화하지않는다. V239는동일producer의한commoncache에서새control/treatment두head를각1000회학습하는별도계획이다.

- CONTROL_CANONICAL_LOCAL:76098, 새1000, optimizer4state. TREAT_CANONICAL_CONTEXT:86054, 새1000, optimizer10state. 원seed17 baseSHA d1b8da97… exact, TREATresidualfinal0. oldCONTROL재사용0.
- 같은PCG17 1000×32 ledgerSHA20a1fc74…, family각8D8E, .5 familymeanBCE, AdamW.001/wd.0001, FP32deterministic/문턱0. 새CF/encoder/normalizer/earlystop선택없음.
- trainer는B canonical6featureNPY와원samples/geometry만읽는다. DINO/weights/RGB/oldprefix/oldCONTROL checkpoint loader는없다. NODE_PIXEL_FEATURES1154는producer감사artifact이며trainer는공통BASE/NODECLS를byte검증한다.
- 전7617행basefirst384/condition/fractions/index/availability를byteexact검산하고공통QA를봉인한뒤FIT5876만optimizer에넣는다. 양head의final1000PT+log가모두freeze된뒤에만DEV포함7617logits를계산한다.
- SOURCE15040/7423masked는producer계보에보존,known7617분모만별도CSV로유지한다. SYNwitness0와YCBassociated-r/nativeUNKNOWN는출력별로분리한다. current/held/newGT/PnP/좌표수정은0이다.
- SOURCEgate는meanfamilyDEVbalancedrecall+3pp/모든D/E하락2pp이하/YCB E>=50%/SYN D와E>=90%다. 노출된internalDEV의최소admission계획이며실사성공이나자동current채택이아니다.

`ENTRY_FALSE_TEMPLATE.json`은false이며실제cacheCOMPLETE/featurebindings가null이다. Root의추후별도actualproducer완료·QA·exactentry가없으면실행할수없다. 사용CLI는 `python -B train_canonical_pair.py --entry ROOT_TRUE_ENTRY.json`이다. 이번mainpush범위는준비물공개이며새학습은없다.

토이6그룹 b7de11 exit0 PASS: base초기SHAexact/76098·86054, 공유byte검산과1ULP·signedzero차이거절, PCG17 quota, 동일초기logits·twofamilyloss, randomhead2backwardfinite/optimizer0, productiontrain함수의DEV입력차단, SYN/YCBgate경계. 실제SOURCEvectors/geometry/weights/pixels읽기0이다. 독립CODEreview는pending이며이를실제학습완료로표시하지않는다.
