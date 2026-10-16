[확인] 이 보조 분석은 완료된 A의 최종 25방법(정식 6fit 포함) 점/자세 평가 행을 다시 묶어 `DONE`으로 갱신했어. 새 inference, PnP, optimizer update, 사람 레이블은 각각 0회야. 아래 명령으로 같은 고정 결과를 재집계할 수 있고 이전 19방법 중간 결과의 실행 시간도 ledger에 보존해.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_joint_action_handoff_20261006_v1.subgroups --source-root /home/minjae/Documents/github/pallet-pose
```

[확인] 최신 `LABEL_PROVENANCE_AUDIT.json`의 DEV319 사용자 입력 등급은 clean153/moderate92/severe74야. 기존 감사 PASS와 7개 원천의 현재 byte hash를 확인하고 A의 잠근 319 ID에 매핑해. 등급 기준 `NOT_CONFIRMED`는 유지하므로 외부 가림 정도의 정답으로 쓰면 안 돼. 거리 역시 기존 workspace `distance_bin`의 near155/mid103/far59/unknown2를 영상 경로+SHA로 연결해. 새로운 metre 경계나 거리 레이블을 만들지 않았고 독립 거리 계측을 뜻하지 않아.

[확인] 최신 사람 코너 상태는 DIRECT_VISIBLE1776/SELF_OCCLUDED462/EXTERNAL_OCCLUDED218/OUT_OF_FRAME43이고, 2499개 `(frame_id, corner_id)`가 A의 canonical reference mask와 정확히 같아. 기존 영상/annotation SHA가 등급 감사와 같은지 확인해. 상태가 있더라도 annotation 좌표 출처 `unknown`, 모델 예측 노출 `NOT_CONFIRMED`, 독립 물리 참조 없음은 그대로야. 사람 상태 입력이 좌표를 새 독립 정답으로 승인한 것은 아니야. UNKNOWN 상태와 좌표 출처 unknown은 서로 다른 항목이야.

[확인] 다음 파일의 열은 이번에 새로 정의한 스키마야.

- `A_REAL_DEV_DESCRIPTIVE_SUBGROUPS.json`: 입력 hash, 319개 frame mapping, 각 group의 ID/분모, seed별 기존 지표와 paired 차이·손상·pose 실패 4분모, 3 seed 통계의 산술평균을 담아. `human_corner_state_any:*`의 frame group은 해당 상태의 참조 코너가 하나 이상인 frame이므로 서로 겹쳐. frame 수를 합쳐 새 표본 수로 쓰지 않아.
- `A_REAL_DEV_DESCRIPTIVE_SUBGROUPS.csv`: 방법·seed/seed 통계 평균·group별 코너/자세/실패/NoOp 지표야. pose는 선택된 instance에 대한 실제 전체 F(q)의 결과를 full frame 분모로 보존해. detector 관측319와 GT 매칭311을 구분하고 GT 매칭 실패8을 검출 실패로 부르지 않아. px/cm/deg/m 및 fraction 단위는 열 이름에 표시했어.
- `A_REAL_DEV_HUMAN_CORNER_STATES.csv`: canonical GT identity에 정렬한 상태별 참조/관측 코너 수, 조건부 관측 오차 median/P90, 전체 unmatched penalty의 PCK10을 담아. 동일 frame의 코너를 독립 표본처럼 bootstrap하지 않아. 저장된 whole-object symmetry branch를 재사용해.

[확인] subgroup은 반복 사용한 DEV의 기술 통계야. 새 CI·유의성·방법 선택·채택 판정을 만들지 않고, 통제와 실용 기준선의 전체 평가 판정을 바꾸지 않아. 실제 clean/occluded 동일 자세 쌍이나 독립 6D 계측의 대체물이 아니야.
