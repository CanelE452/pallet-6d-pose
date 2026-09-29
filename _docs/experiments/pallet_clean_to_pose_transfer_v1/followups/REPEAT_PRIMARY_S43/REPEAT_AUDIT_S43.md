# 같은OCC recipe seed43 반복 독립감사

상태: **PASS**. 고유입력해시 1390개검증. 새학습/optimizer/GPU감사0회.

- RAW/REF 각각320update·5epoch, 같은R0·78실사·512source·augmentation·mask·box·실제입력계약을확인했다.
- 각checkpoint879개state중보호된747개를R0와CPU bit-exact비교했다. 나머지변경은허용132parameter범위안이다.
- 실제320batch RAW/REF 순서/RGB/support/box/가림계획을독립비교했다. 좌표값은달라야한다.
- seed42와43의실제순서·RGB·가림계획·최종가중치가달랐다. 같은이미지첫occurrence도기본증강/RNG변경을확인했다.
- 첫8개실제학습입력이CPU사전검사와일치했다. 사전검사의uint8 RGB batch와실제/255 float batch hash를같다고주장하지않고image별before/after hash를비교했다.
- 전역원장6fit·1920update·selector fit0을검산했다. 사전원장4fit의불변사본과현재변하는전역원장을구분했다.
- 동일128/99및985코너분모,그룹별T/R/yaw median/P90·PCK/AUC·paired9분류를저장per-frame오차에서다시계산했다.
- 예측잠금선행은고정코드의읽기순서·hash·산출물mtime에근거한다. 별도SCORING_START이벤트나파일접근로그가있는것으로꾸미지않았다.
- 초기879개tensor동일성은runtime torch.equalassertion과R0binding에근거한다. 저장되지않은step0snapshot을사후복원했다고쓰지않는다.

[상세 JSON](REPEAT_AUDIT_S43.json)
