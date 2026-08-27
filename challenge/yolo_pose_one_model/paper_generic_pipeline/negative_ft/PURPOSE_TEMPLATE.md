# PAPER_NEGATIVE_FT — PURPOSE (템플릿)

**[소비처]** _채울 것_
**[문장]** _채울 것_

## 실행 조건 (전부 만족해야)

```
positive STRONG_PASS        PAPER_YOLO_VERDICT.json
REAL_NEG_DEV 확보           real negative 평가셋이 있어야 FP 를 잴 수 있다
사용자 승인
```

## 허용 / 금지

```
허용   generic synthetic positive, target-free synthetic negative,
       필요 시 real negative-only
금지   target positive synthetic, target real positive
```

## 목적

**FP 억제만 추가하면서 pose 를 보존한다.** pose 가 나빠지면 실패다 —
dense negative suppression 이 seed2 pose safety 를 깬 전례가 있다.

## ratio

지금 정하지 않는다. `--negative-ratio` 로 명시하며 기본값이 없다.
