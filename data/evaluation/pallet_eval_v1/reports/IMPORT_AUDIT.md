# Import audit

```text
plastic total            140 / 140
plastic controlled       128 / 128
plastic excluded         12 / 12
wood total               45 / 45
multi-shape controlled   173 / 173
negative indexed         2689 / 2689
legacy annotation paths  679
legacy unique active     409
legacy unresolved        0
negative duplicate SHA groups  1
```

Legacy `409`는 2026-08-30 현재 감사값일 뿐 frozen membership target이 아니다.

## Source safety

```text
source image changed        0
source annotation changed   0
source size/mtime/count changed  0
DEV -> FINAL promotions     0
duplicate active frame SHA groups  1
```

현재 active duplicate group은 frozen `DEV_NEG2689`의 기존 중복 membership이며
삭제하지 않고 `DUPLICATE_AUDIT.csv`에 보존한다.

## Warnings

- 없음

## Unresolved legacy images

- 없음
