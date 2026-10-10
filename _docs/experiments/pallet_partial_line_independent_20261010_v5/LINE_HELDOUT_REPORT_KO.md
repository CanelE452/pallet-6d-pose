# 저장된 LOO 기록으로 미사용 선을 검증할 수 있는가

**현재 기록의 독립적인 다른 점 근거는 793개 미사용 선 중 17개에만 있다. 실제 자세 정확도가 개선됐다는 결과는 아니다.** 같은 선의 support가 서로 일관적인 것과, 그 선의 두 끝점 관측을 사용하지 않은 다른 점들이 선을 지지하는 것은 구분했다.

고정된 쉬움153+중간92=245장의 GT 없는 기하 원행을 읽었다. 244장에 미사용 선이 있고 1장에는 없다. 저장된 코너별 LOO 기록447개와 미사용 선793개를 연결한 1,377개 packet–edge 쌍을 모두 검사했다. 원래 코너의 채택 여부나 잔차의 크기로 packet을 고르지 않았다.

| 집합 | 기하 지지: RMS≤8px | 불일치: RMS>8px | UNVERIFIED |
|---|---:|---:|---:|
| 전체 저장 packet–edge 쌍1,377개 | 9 | 8 | 1,360 |
| 미사용 선793개 | 9 | 8 | 776 |

17개의 qualified packet은 12장에 걸쳐 있고 각각 다른 선에 대응한다. 따라서 이번 집계에는 동일 선의 qualified packet 간 충돌이 없다. 충돌이 있었다면 가장 작은 잔차를 고르지 않고 그 선을 UNVERIFIED로 남기는 정책을 먼저 고정했다.

## 17개가 의미하는 독립성의 범위

qualified packet마다 두 끝점 ID와 초기 H가 `used`인 scoring pool, 선택된 generator/fit/inlier, 그리고 저장된 모든 scoring 후보의 generator/실제 fit/inlier에서 빠져 있는 것을 확인했다. 수치 초기 자세·재투영·치수 prior가 없고, available `NEW_POSE`이며 미해결 다중해가 없는 기록만 사용했다. `used`나 후보별 근거가 빠진 기록은 독립성을 가정하지 않았다.

**17개 모두 한 끝점은 단일 코너의 temporary heldout이고, 다른 끝점은 원래 scoring pool에서 ineligible이었다. 다른 끝점이 H로 명시적으로 제외된 qualified packet은 0개다.** 저장된 `excluded=H∪{k}`에 두 번째 끝점이 없고 `used`에도 없으므로, `used=eligible−excluded`를 만드는 고정 v4 코드에서 그 끝점은 ineligible에 해당한다. 이 추가 행에는 ineligible의 상세 원인을 다시 복사하지 않았으므로 sentinel·비유한 좌표·영상 밖 중 어느 원인인지 새로 단정하지 않는다.

선의 두 끝점을 함께 temporary exclusion에 넣는 명시적 선 단위 LOO는 **0/793**이다. 기존 기록은 단일 코너 LOO이며, 이번 감사는 새로운 두 끝점 제외 자세를 계산하지 않았다. 현재 좌표와 고정 pool에서 두 끝점의 기여가 없다는 증거는 있지만, 다른 끝점을 유효한 영상 좌표로 바꿔도 해가 같다는 변조 불변성 실험은 아니다.

H는 초기 N3에서, 관측 제안과 역할 feature는 Base에서 얻었다. 따라서 이 검사는 조건부 수치 독립성만 다룬다. 전체 RGB 통계 독립성, 실제 물리 경계 소유권, 선 방향의 가시 구간, 전역 자세 유일성을 인증하지 않는다. 잔차는 저장된 자세가 투영한 등록 3D edge의 두 끝점에서 관측 **무한직선**까지의 법선 거리 RMS다.

## 검증되지 않은 기록의 이유

81쌍은 실행된 LOO pose packet이 저장되어 있지 않았다. 1,279쌍은 아래 witness 조건을 통과하지 못했다. 아래 수치는 각 이유 계열을 packet–edge 쌍당 한 번씩 센 것이며, 이유들이 겹치므로 합산해 프레임 수나 독립 관측 수로 사용하면 안 된다.

| 저장 witness의 실패 이유 | packet–edge 쌍 수 |
|---|---:|
| `used` scoring pool에 끝점 또는 H가 포함됨 | 1,270 |
| 저장된 후보 중 generator/실제 fit에 끝점 또는 H가 포함됨 | 각각1,230 |
| 저장된 후보 중 inlier에 끝점 또는 H가 포함됨 | 1,225 |
| 선택된 실제 fit ID에 끝점 또는 H가 포함됨 | 1,206 |
| 선택된 final inlier ID에 끝점 또는 H가 포함됨 | 1,202 |
| 선택된 generator ID에 끝점 또는 H가 포함됨 | 1,173 |
| LOO 새 자세 unavailable / 투영 없음 / 선택 generator witness 없음 | 각각83 |
| 미해결 ambiguity witness | 8 |

원행에는 후보 인덱스별 실패 항목97,158개가 보존되어 있다. 이는 후보별 반복 진단 항목 수이며 관측 수가 아니다. fit에서 빠졌어도 scoring pool에 들어간 끝점은 해의 합의 선택에 영향을 줄 수 있으므로, fit ID만 확인해 독립 검증이라고 부르지 않았다.

**UNVERIFIED는 physical no-match/NONE이 아니다.** 그 선이 없거나 틀렸다는 결론이 아니라, 저장된 기록만으로 이 독립 검증을 수행할 근거가 부족하다는 뜻이다. 불일치8개도 다른 점들이 만든 조건부 자세와 선의 기하 불일치이며, 실제 선 소유권 오류의 정답으로 바꾸지 않는다.

## 실행과 재현

코드를 먼저 봉인하고 저장 원행 산술을 한 번 실행했다. freeze1회, 실제 audit1회, 완료된 ROWS의 보고용 요약 읽기1회다. audit은13.620124989초에 exit0/PASS로 종료했다. 새 모델·PnP·optimizer·ray·학습·RGB 생성·GT 점수 계산은 모두0이다. 배포 정책과 기존 v5 수치 원행을 변경하지 않았다.

```bash
python3 -I -S scripts/research/pallet_partial_line_independent_20261010_v5/line_heldout_audit.py freeze
python3 -I -S scripts/research/pallet_partial_line_independent_20261010_v5/line_heldout_audit.py run
```

Python≥3.9 표준 라이브러리만 사용한다. 기본 입력·출력은 이 v5 문서 폴더다. 기존 audit PROTOCOL/STARTED/ROWS/CHECKS가 있으면 재실행을 거부한다. 재현 시 두 명령 모두에 `--output /tmp/pallet-partial-line-independent-private-20261010-v5/heldout-recheck`처럼 아직 audit 산출물이 없는 새 하위 폴더를 지정하면 원본 입력과 완료 receipt를 보존한다. 출력은 새 v5 문서 폴더 또는 지정된 v5 private subtree만 허용하며 symlink 경로를 거부한다.

| 산출물 | SHA-256 |
|---|---|
| 감사 코드 | `4f9f0ae50fe35f4db3a74ab55c5f7e8e2f1b11cc50f03c6e658b974cc8b3a5e0` |
| [고정 프로토콜](LINE_HELDOUT_PROTOCOL.json) | `14a3726ec8711606ba9c6b44c74e5a4dfff22f0b4ecffb6b799960f9e0e48ed0` |
| [시작 기록](LINE_HELDOUT_STARTED.json) | `9f4fbc1019d7f70417743bde5c87253f54a643b3dd81c41feccdb9b8d96a1305` |
| [793개 선의 모든 packet 원행](LINE_HELDOUT_ROWS.jsonl.gz) | `fc1d60250d29d32282e721df8b584c80326510547a349c388b4a56766b8bf8fe` |
| [완료 검산](LINE_HELDOUT_CHECKS.json) | `16ab4af62d8e5955168c033999326caeb5c11e068dbb4b917dc79da77fd5c185` |

입력 기하·관측·기존 프로토콜·완료 receipt와 당시의 point-only solver/selection 코드 바인딩은 감사 프로토콜에 있다. 실제 감사 중 이 입력들과 감사 코드의 SHA/byte 수가 유지된 것을 확인했다. 이것은 보존된 기록의 계약 검사이며, 부족한 명시적 두 끝점 LOO를 이미 실행했다고 주장하는 근거가 아니다.
