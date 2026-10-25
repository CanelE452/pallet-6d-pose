# V245 사전 고정12 SOURCE 비교 갤러리 준비

이전 V239의 같은12슬롯/11고유표본/9그룹만 사용합니다. 새점수로 사례나 빈bucket대체를 고르지 않습니다. 새baseline은 기존V239 TREAT86054이며 옛CONTROL76098이 아닙니다.

```bash
python -B build_fixed12.py --entry /absolute/Root_TRUE_ENTRY.json
```

Root의 실제fit/finalfreeze/독립fitQA/저장SOURCEreport 후 별도true entry가 있어야 새CSV와 기존PNG를 읽습니다. 원CASEmetadata가가리키는 exact81PNG만허용하고 원CARDcaption은읽지않습니다. 입력q/L/native/U와기존SOURCEteacher는변경하지않습니다. Context는기존F32의표시용양자화PNG임을표시합니다. YCB nativeUNKNOWN/associated-r를구분하며 q/reference를보정하거나GT/owner/pose를표시하지않습니다.

현재는metadataclosure81 및syntax만검사했고 이미지·새CSV·features·checkpoint읽기/forward·학습0입니다. SOURCE DEV는이미노출·재사용됐으며12사례가대표성·모델성공·원인증명은아닙니다.
