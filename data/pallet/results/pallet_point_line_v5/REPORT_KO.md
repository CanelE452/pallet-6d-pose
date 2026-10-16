# 치수·대칭 일관 점·선 공동 자세 v5 — 차단 보고

- 상태: **`BLOCKED_MISSING_BUNDLE`**. 지시문 §2~§8(번들 검증·진단·export·학습·평가) 미실행.
- 수행: 지시문 §0 의 1단계 "기존 기록/소스/GT/치수/프레임 계약 확인" 만 완료.
- 새 측정·학습·추론 **0회**. 기존 v4 결과·GT·checkpoint·annotation 변경 **0**.
- commit/push·외부 알림 **없음** (§0 금지, 이전 작업 승인 미확장).

## 1. 문제와 차단 사유

v5 는 `pallet_point_line_v5_bundle.zip` 의 `plpose_v5/` 코어(geometry·solver·heads·runner·
assessment·diagnostics)를 실제 데이터 경로에 **연결**하는 작업이다. 그런데 전달된 것은
문서 5개뿐이다 — `CLI_INSTRUCTIONS_KO.md`, `COPY_PASTE_CLI.txt`,
`METHOD_AND_CONTRACT_KO.md`(2부, 내용 동일), `LOCAL_VALIDATION_KO.md`.
사용자가 이것이 전달분 전부임을 확인했다.

지시문 §1 은 "이미 제공된 `plpose_v5/` 를 새로 작성하지 마라"고 못박는다. 코어를 내가
구현하면 (a) 지시문 위반이고 (b) 저자가 72개 테스트로 검증했다는 **그 산출물이 아니게** 되어
비교 자체가 무의미해진다. 따라서 재구현하지 않고 차단으로 보고한다.

`LOCAL_VALIDATION_KO.md` 는 코어가 저자 환경에서 실제로 만들어져 돌아갔음을 기록한다
(72 tests, 3 arm × 8 = 24회 실제 AdamW update, 파라미터 point 24,927 / direct 26,879 /
hough 32,683). 즉 코드는 존재하며 이번 전달에 포함되지 않았을 뿐이다. **ZIP 본체를 보내면
§2 부터 이어서 진행할 수 있다.**

## 2. 번들 없이 확인한 것 — 소스와 좌표계

지시문 §3 이 지정한 13개 소스가 모두 실재하며 SHA 를 `SOURCE_REGISTRY.json` 에 등록했다.

**R0 checkpoint 대조 PASS** [확인]. `integration.py:28` 이 선언한
`970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7` 와 디스크 실측이 일치한다
(`challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt`).
이는 v4 가 쓴 `0960fb32…`(joint hough_joint_seed1)와 **다른 checkpoint** 이며, 지시문이
둘을 같은 R0 로 바꿔 쓰는 것을 금지한다. 두 SHA 를 레지스트리에 나란히 기록해 혼동을 막았다.

## 3. 상류 게이트 판정

### §4.1 object frame — `FEASIBLE_PENDING_TARGET_CONVENTION`

재료는 전부 있다. `GEOMETRY_METADATA_AUDIT.md` 의 **전수 60,000장**(train 55,980 / val 4,020)
에서 K·dims·pose(R,t)·8 cuboid corners·perm_v4 가 각각 **100%** 존재하고 파싱 실패 0건이다.

원점 검증을 실제 label 하나로 수행했다 — pose translation 을 K 로 투영한 값이 저장된
`projected_cuboid_centroid` 와 **최대차 5.7e-14 px** 로 일치한다. 즉 **object frame 원점 =
cuboid 기하 중심**이다. 중심이 바뀌지 않으므로 지시문 §2 가 요구한 대로 `reframe_pose`
(회전 전용)가 올바른 도구이고 **translation 을 건드리면 안 된다**. pose R 은 det=+1 정상 회전.

남은 일(번들 필요): 회전 A 를 `plpose_v5.geometry.SIGNS` 순서에 맞춰 유도하고 전수 재투영으로
검증. **SIGNS 가 결손 코어 안에 있어 목표 축 순서를 확정할 수 없다.**
주의 — 동결 기록에 축 순서가 두 가지 공존한다(world cuboid extents = depth,width,height /
`fixed_renderer_dimensions_m_xyz_model_input` = depth,height,width). 필드 이름으로 추정하지 말고
재투영으로 검증해야 한다. 또 `camera_dynamic_0123_v4` 는 프레임마다 앞면을 다시 고르는
규약이라 고정 object frame 이 아니므로, 매핑은 `perm_v4` 가 아니라 renderer world cuboid/pose
를 경유해야 한다. `SEMANTICS.md` 는 현행 60k `perm_v4` 를 쓴 외부 renderer 의 writer 를
로컬에서 확보하지 못했다고 기록하므로 그 계보는 부분 미검증이다.

### §4.2 치수·대칭 — `C4_NOT_EVALUATED` [확인]

지시문의 핵심 전제 하나가 이 데이터에서 성립하지 않는다.

| asset | 계약 상태 | 유효 대칭 | train 비중 |
|---|---|---|---:|
| scene.usd | CONFIRMED (Rz180 측정, 90도 미측정) | **C2** | 48.4% |
| scene_1.usd | CONFIRMED (Rz180 측정, 90도 미측정) | **C2** | 17.1% |
| eur_pallet_bk_cc0.glb | UNKNOWN (mesh 부재) | **C1** | 17.3% |
| woodpallet_block_jtoastie_ccby.glb | UNKNOWN (mesh 부재) | **C1** | 17.1% |

`LOSS_SYMMETRY_CONTRACT.json` 규칙상 C4 는 **confirmed_90 그리고** width==depth(1e-6) 를
동시에 요구하며 "치수만으로는 절대 승격 안 됨"이다. 90도가 확인된 asset 은 **하나도 없다**.
게다가 `SYNTHETIC_DIMENSION_AUDIT.md` 전수 60,000장에서 exact-square 는 **1장**뿐이고,
그 1장이 하필 C1 asset 것이다. → 두 조건 동시 충족 프레임 **0**.

따라서 v5 파일럿은 **C1/C2 만** 실행되고 C4 경로는 데이터로 검증되지 않는다. 지시문 §4.2 대로
`C4_NOT_EVALUATED` 로 보고하며 "정사각/직사각 혼합 성공"을 주장하지 않는다.

소급 함의: v4 의 "C4 후보" bank 는 **번호 재배열 제안이지 물리 대칭 주장이 아니다**.
v4 외부 감사가 지적한 내용이 계약 문서로 확인된다.

## 4. 한계

- §4.1 원점 검증은 label **1장** 범위다. 전수 재투영(≤0.1px)은 코어가 있어야 한다.
- 대칭 판정은 새로 측정한 것이 아니라 **동결된 로컬 계약 문서를 읽은 것**이다.
  두 `.glb` 의 mesh 가 이 기계에 없어 직접 측정이 불가능하다는 사실 자체가 계약의 근거다.
- feature 정합(§4.3), 입력/실패 모집단 ledger(§4.4), 기하 진단(§5)은 착수하지 않았다.

## 5. 다음으로 허용되는 행동 한 가지

`pallet_point_line_v5_bundle.zip` 본체 수령. 받으면 §2(verify_bundle + pytest)부터 이어서
진행한다. 보내기 전에 지시문 작성자와 정할 것: **C4 축은 이 데이터로 검증 불가**이므로
파일럿의 대칭 설계를 C1/C2 기준으로 재확인할지 여부.
