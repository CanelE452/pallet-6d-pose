# SAM으로 확인한 KP 가시성과 윤곽선 좌표 보정

**KP 위치의 색/전경 점유율만으로 가시성을 정하면 실제로 보이는 가상 cuboid 모서리를 버릴 수 있다. 반면, 물체 윤곽선 두 개의 교점은 독립적인 좌표 보정 근거가 됐다. 다만 pose 개선은 아직 검증되지 않았다.**

CAL 76장만 사용했다. 초기 pose의 self-H, 모델 치수, native KP 순서, 물리 좌표계 Q를 모든 후보에서 고정했다. 초기 self-H에 포함된 점은 보정·fitting에서 제외했으며, pose 재투영 좌표를 관측값으로 쓰지 않았다. TEST/FINAL 접근과 프로덕션 모델 변경은 없다.

공식 [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)의 고정 `vit_t` 체크포인트를 로컬에서 사용했다. 새 학습과 이미지 업로드는 없었다. 두 프롬프트는 원래 q8의 bbox만 사용하는 BOX와, 동일 bbox에 RGB 근거가 높은 내부 positive/외부 negative를 추가한 ANCHOR다. 세 mask 중 모델의 원래 predicted-quality 최댓값을 선택하고 동률은 원래 인덱스 순서를 유지했다. 이 점수는 pallet 식별 확률이나 실제 IoU가 아니다.

## 1. KP 위치의 mask 점유율로 가시성을 정한 실험

기존 raw RGB body 점수 ≥0.35와 SAM inward occupancy ≥0.5를 동시에 요구했다. 모델은 정답을 읽기 전에 152개 mask·prompt와 684개 pose row를 저장·봉인했다.

| 방법 | visible precision | DIRECT recall | EXTERNAL 비가시 판정률 | pose 승인 |
|---|---:|---:|---:|---:|
| 기존 source-CNN curriculum | 0.949 | 0.931 | 0.753 | 60/76 |
| SAM BOX | 0.984 | 0.489 | 0.961 | 23/76 |
| SAM ANCHOR | 0.979 | 0.492 | 0.948 | 25/76 |

두 SAM 후보의 5cm/5° 성공은 41/76으로 기존 source-CNN의 37/76보다 많았지만, 평균 translation/rotation은 BOX 116.445cm/19.481°, ANCHOR 116.448cm/19.472°로 기존 116.291cm/19.290°보다 악화됐다. visible recall 손실이 커서 이 정책을 채택하지 않았다.

ANCHOR에서 버린 DIRECT 191점 중 73점은 원래부터 reference 좌표와 8px 이내였다. 따라서 누락 전부를 위치 오차로 설명할 수 없다. [실제 사례](foundation_mask_v1/qa_overlays/01.png)처럼 둥근 물체의 가상 cuboid 모서리는 실제 표면 밖에 놓일 수 있다. 이 경우 KP 위치의 물체색 부재는 occlusion의 증거가 아니다. 초기 self-H 계산을 유지하면서 **가시성 의미와 해당 pixel의 물체 점유를 분리**해야 한다.

근거: [고정 프로토콜](foundation_mask_v1/PROTOCOL.json), [봉인](foundation_mask_v1/PREDICTION_SEAL.json), [결과](foundation_mask_v1/RESULTS.json), [누락점 QA](foundation_mask_v1/VISIBILITY_QUERY_FAILURES.json).

## 2. 기존 가시성은 유지하고 SAM 윤곽선으로만 좌표를 보정한 실험

첫 실험의 mask·prompt를 그대로 재사용했다. 기존 SEG appearance의 상태·점수·가중치도 그대로 유지하고, 두 관측 윤곽선의 교점으로만 좌표를 바꿨다. mask를 ROI 밖에서 0으로 만드는 처리는 하지 않았다. 고정 STRICT(r=12px, 최대 이동 6px)와 SPARSE(r=12px, 최대 이동 12px), 두 프롬프트의 4개 후보를 선언한 뒤 456개 pose row를 정답 접근 전에 봉인했다. 초기 실패와 큰 outlier를 포함한 76장 전체가 분모다.

| 후보 | 보정 DIRECT 점 / 전체 보정 frame | 보정점 평균 오차 전→후 | 개선 / 악화 |
|---|---:|---:|---:|
| BOX STRICT | 27 / 21 | 4.428→3.936px | 13 / 14 |
| BOX SPARSE | 88 / 43 | 5.887→3.896px | 58 / 30 |
| ANCHOR STRICT | 28 / 22 | 4.630→3.896px | 16 / 12 |
| ANCHOR SPARSE | 86 / 42 | 5.835→4.011px | 58 / 28 |

BOX SPARSE의 전체 DIRECT 376점 중앙값은 7.844→7.178px였다. pose 승인은 네 후보 모두 기존과 같은 60/76이었다. 기존 source-CNN 대비 BOX SPARSE의 평균 translation은 116.291→116.161cm, rotation은 19.290→19.294°, 성공은 37→39/76이었다. session bootstrap 95% CI는 ΔT −0.130cm **[−0.287, +0.048]**, ΔR +0.004° **[−0.057, +0.062]**였다. ANCHOR SPARSE도 ΔT −0.162cm **[−0.332, +0.046]**, ΔR +0.010° **[−0.035, +0.064]**였다. 네 후보 모두 pose CI에 0이 포함된다.

좌표 근거는 개선됐으나 pose 개선을 확인했다고 말할 수 없다. 공유된 윤곽선에서 나온 여러 교점은 서로 상관되어 있고, 가상·둥근 모서리의 정의 차이도 남는다. [개선 사례](foundation_mask_v1/independent_boundary_v1/qa_overlays/00.png)는 11.48→0.22px였지만, [악화 사례](foundation_mask_v1/independent_boundary_v1/qa_overlays/01.png)는 0.69→9.01px였다. 네 후보에서 같은 human SELF KP 한 점도 보정됐다. 그 점은 초기 self-H에는 없었고, 오차가 1.42→1.85–2.52px로 커졌다. 이는 공급된 self-H를 위반한 것이 아니라, 초기 pose/가상 cuboid와 human self 상태의 불일치다.

근거: [4개 후보 프로토콜](foundation_mask_v1/independent_boundary_v1/PROTOCOL.json), [봉인](foundation_mask_v1/independent_boundary_v1/PREDICTION_SEAL.json), [전체 결과·CI](foundation_mask_v1/independent_boundary_v1/RESULTS.json), [좌표별 QA](foundation_mask_v1/independent_boundary_v1/COORDINATE_AUDIT.json), [H/Q/관측값 검증](foundation_mask_v1/independent_boundary_v1/VERIFICATION.json).

## 재사용과 다음 개선

재사용 가능한 함수가 이미 있다. 아래 import는 실험에 저장된 `code_snapshot`의 boundary 함수를 사용한다. `config`는 위 4개 후보 프로토콜의 `profiles` 항목에서 그대로 가져온다. 원래 image 크기의 고정 SAM mask와 초기 self-H, 보정 전의 기존 appearance를 전달한다.

```python
from scripts.research.pallet_foundation_mask_20261013_v1.independent_boundary import infer_rgb_boundary_points

result = infer_rgb_boundary_points(
    image_bgr, q, initial_self_H, frozen_seg_appearance, config,
    material_mask=selected_sam_mask,
    mask_source="OBSERVED_RGB_MASK_ONLY_OFFICIAL_FROZEN_MOBILESAM",
)
# result["points"]: 보정 관측값; result["observed_refined_ids"]: 실제 이동한 KP
```

다음 개선은 **초기 self-H를 그대로 유지하면서, 두 윤곽선의 line-fit 잔차·지지 길이·교점 각도로 좌표 불확실성을 계산해 기존 detector 좌표와 가중 결합하는 것**이다. 현재 이미 정확한 점을 9px 악화시킨 사례까지 무조건 교점으로 교체하는 문제가 남아 있다. 이 개선은 point 색을 가시성으로 해석하지 않으며, 새로운 관측 근거가 약한 교점의 영향을 줄이는 방향이다. 아직 구현·검증한 결과는 아니다.

## SHA256

아래 값은 실제 저장 artifact의 전체 SHA256이다. 최초 occupancy scoring의 필드명 수정은 `observed_refined_ids`를 읽도록 고친 것뿐이며, 당시 inference runner는 [별도 snapshot](foundation_mask_v1/pilot_inference_snapshot.py)에 보존했다. mask·prompt·봉인된 추론은 바뀌지 않았다.

```text
MobileSAM checkpoint       6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
occupancy PROTOCOL         d49ad85648f4b12572f7bdcf9524bf61238d71cfedafb3092258ba1fe4c8e4d8
occupancy PREDICTIONS      0872f1d5bf024c6920fef8c6fab0722c0b5464b3a573327bca0d37c480b79bc7
occupancy PREDICTION_SEAL   921befb393ce1cefd7de405f886d269f16e49188fa2adcd88dd892f96d89415f
occupancy RESULTS          dcc85289834f26e40dcafb7062fcaef2f9fdd615e8e5a80d03e98e7db8d1f7c5
boundary PROTOCOL          6b971b9070e3f430b9dec4a544ec6d5976639cad2edb3604b8d7618a3381ec8f
boundary PREDICTIONS       4937c86b32d8739be363d478c99919042c5293ef90694b8a8d6912f8134dd80f
boundary PREDICTION_SEAL    b276377bde49e836580f3ca20e2c428758ae3a1401611e7d491a0605517b78c9
boundary RESULTS           60586f83612372889894abf19163717132f3cf71d36c4f1bc091256a14ad9ddc
```
