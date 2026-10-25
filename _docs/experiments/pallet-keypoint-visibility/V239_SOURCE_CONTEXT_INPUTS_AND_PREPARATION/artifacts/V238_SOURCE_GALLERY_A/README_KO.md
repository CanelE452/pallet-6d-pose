# V238 fixed SOURCE context gallery (CODE-only)

`ENTRY_FALSE_TEMPLATE.json`은 false다. Root의 별도 completedfit/최종freeze/전체QA/CSV/선정pixel scope 없이 실행하지 않는다. CODE와 사전 V235정책만 읽었고 새 CSV·SOURCE cache·weight·feature·GT는0이다.

실행예: `python -B build_gallery.py --entry ROOT_TRUE_VISUAL_ENTRY.json`

같은 사전12bucket에서 최소DEV sample_index를 고르고 CASE_IDS_BEFORE_PIXELS를먼저쓴다. 빈bucket 대체0, 중복사례는그대로두고고유수를별도로적는다. 그후선정group의 원cache8nodes/context만읽는다. 새원RGB 재crop·모델forward·좌표보정은없다.

최종예정: `gallery_v1/COMPLETE.json`, `PORTABLE_GALLERY_KO.html`, `FIXED12_SOURCE_DEV.png`, `CASES.csv`, `CASE_DATA.json`, `CASE_IDS_BEFORE_PIXELS.json`, `CASE_XX_SAMPLE_XXXX/CARD.png`. 각card는원active96/valid/q8context224/다른7crop과CONTROL/TREAT 저장점수·회복악화를보인다. Attention은재생하지않으며 visibilitygold로표시하지않는다.

원node CHW→HWC PNG복원SHA를검증한다. F32context는display만half-up양자화하고원F32SHA와표시SHA를별도로남긴다. SYN witness0와YCB associated-r/nativeUNKNOWN 교사를구분하고literalqowner·부재→E를만들지않는다. 모든자료는재사용 SOURCEinternalDEV이며blind/current평가가아니다.

V1의실행전순서수리와V2실제descriptor검사종료(optimizer0)는구분한다. 아직후속완료학습이없다면실제gallery는생성할수없다.

토이5그룹82954d exit0 PASS. 인공CSV/배열로minimum/빈bucket/FIT/U/중복배제·보존과CHWPNG·F32display를확인했고 `INVENTED_TOY_CARD.png`만실제toolview했다. 이것을실제SOURCE시각검사로세지않는다.
