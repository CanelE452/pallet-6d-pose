# V245 SOURCE YCB q–r·object/aux join CODE 준비

Root가 finalfit/freeze/독립fitQA와두자연SOURCECSV를승인한뒤에만실행합니다. 지금 metadataCSV·새CSV·이미지·배열·PT·GT·모델읽기0입니다.

```bash
python -B join_query_error.py --entry /absolute/ROOT_TRUE_ENTRY.json
```

기존1521 YCB rowid/sampleindex/role/associated-r D/E를정확히join하고, SOURCE FIT/DEV를유지합니다. 사전고정된거리≤3/>3 및obj7aux3·7/obj2aux2 대나머지로표를만듭니다. 누락거리는별도UNKNOWN_DISTANCE로행을보존하고빈분모는0/null로기록합니다. 가짜이미지나추가toyworkflow는만들지않았습니다.

저장q–r 거리는associated-r 주변의SOURCE구성query변위이며실제pallet검출기의위치오류로부르지않습니다. YCB native는UNKNOWN,physicalgold나literalq owner정답이아닙니다. SOURCE DEV는이미노출·재사용됐으며세부표로새문턱·모델·데이터·checkpoint를고르지않습니다.
