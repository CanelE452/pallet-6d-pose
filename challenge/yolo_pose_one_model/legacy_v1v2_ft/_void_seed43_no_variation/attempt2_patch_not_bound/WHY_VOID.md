# 2차 시도도 무효 — 패치가 trainer 에 안 걸렸다

generator seed 패치를 넣었지만 가중치가 여전히 seed42 와 비트 동일했다
(879 텐서 · different=0 · max_abs_diff=0.0).

원인: `ultralytics.models.yolo.detect.train` 이 import 시점에
`from ultralytics.data import build_dataloader` 로 **자기 네임스페이스에 바인딩**한다.
`ultralytics.data.build.build_dataloader` 만 갈아끼우면 trainer 는 옛 함수를 계속 부른다.

```
실측  _UB.build_dataloader is patched -> True
      _DT.build_dataloader is patched -> False   ← 여기가 실제 호출처
```

3차 시도는 바인딩된 모든 모듈(_UB/_UD/_DT)을 갈아끼우고, 패치 발동 기록을
`DATALOADER_SEED_PATCH.json` 으로 남겨 **먹었는지 확인 가능하게** 한다.
