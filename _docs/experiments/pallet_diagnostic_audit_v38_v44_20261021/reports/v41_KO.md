> Public edition of an existing development audit. Identifiers and local paths are redacted. Publication adds no new experiment or improvement claim. Historical no-publication statements describe the original audit date; this edition is separately authorized.



# 사후 단일 큰 회전 오류 진단 E84

**기존 CAL12 표본·선정 규칙·비율을 바꾸지 않는 별도 부록이다.** 본래 질문의 큰 자세 분기에 해당하는 CAL의 유일한 R>45° 사례가 원 표본에 없어서 보완한 것이며, 앞선 결과를 뒤집으려고 유사한 그림자 사례를 고른 것이 아니다. 원 OCPR D1+H1 MLP seed20261018 대N0, 반복 사용된 실사 CAL이다. 저장 오차는 기존 기하 재구성 참조에 대한 값이며 독립 물리GT가 아니다.

## 처음 제외된 이유

같은 촬영 묶음 recording_4e5f9461dc35af510c53의 실패 원 ID 순서는 frame_321d2f78e984660763af, :000372, :000432, :001201이었다. 원 sampler는 다른 실패 촬영 묶음과 round-robin으로 총6장을 뽑고, 각 묶음에서 첫 ID 이후에는 이미 선택한 numeric suffix와 최소 거리가 가장 큰 것을 선택했다.

따라서 이 묶음의 선택 순서는312(C02) →1201(C04) →432(C06)이었다. 셋째 선택에서432의 기존 선택과 최소 거리는120,372는60이어서432가 선택됐다. 다른 촬영 묶음3장과 합쳐6장 예산을 채워372가 제외됐다. R 크기·그림자·실패 원인으로 순위를 정하지 않았기 때문이다. numeric suffix의 실제 시간 단위를 확인한 것은 아니다. 원 ID/순서 전체는 로컬 SELECTION_AND_EXCLUSION_PRIVATE.json에도 보존했다. 이번 부록을 원12장의7번째 실패로 합산하지 않는다.

## 이 한 장에서 확인한 것

| 항목 | 저장 N0 | 원 OCPR |
|---|---:|---:|
| T |14.791694cm|14.489192cm|
| R |83.822071°|84.116404°|
| 공동5cm/5° |실패|실패|
| H1 순위 |5위|1위|
| 선택 ID |N0|G10/ORIGINAL_WD/selected|

원 raw generated bank와 eligible H1 bank 모두 성공 후보가 없다. 따라서 좋은 N0나 성공 후보를 H1이 버려서84°가 된 사례가 아니다. N0부터 큰 분기가 존재했고 원 후보 생성/관측 체계에서 성공 pose를 제공하지 못한 단계까지 좁힐 수 있다. 그 이전의 raw/N3 점·대응·PnP·물리 참조 중 무엇이 직접 원인인지는 이번 저장 감사로 확정하지 않는다.

실제 solver는 full 점5개(native1/2/3/4/5)와 부분선2그룹(I0→neighbor1, B7→neighbor3)을 사용한다. partial로 대체된 full point를 중복 세지 않았다. native6은 initial_self_hidden/S로 조회하지 않았다. B4/B5는 VISIBLE_BACKBONE_FALLBACK이고 사진에서는 보이는 목재 영역 밖 배경 쪽에 놓여 있다. 이는 대응/관측 의미를 확인할 단서이며, 모델의 가상 corner와 실제 재질 silhouette가 다를 수 있어 독립 owner 오류 정답으로 명명하지 않는다.

동일 ORIGINAL_WD에서 두 후보는 모든5 mode에 feasible이고, 비용은 모두 최종 pose가 더 작다. A는46.103276→41.630411, AB는164.714048→123.260808이다. 같은 family 내 N0–최종의 비용 방향 충돌은 없었다. 두 후보 모두84° 근처에서 실패하므로, 이 비용 감소가 올바른 물리 방향으로 개선됐다는 뜻은 아니다. A/AB 모델과 관측 역할 B/I를 구별한다.

## 시각 대조 및 기존12장과의 차이

예측선 없는 crop에서 목재 판재 방향과 외곽은 대체로 보이고, 물체를 가리는 별도 물체는 뚜렷하지 않다. 주변에는 큰 어두운 바닥 영역이 있지만 physical W/D 이름과 optical shadow 정답은 UNKNOWN이다. 그 메모를 저장한 뒤 mask/관측/pose 그림을 직접 열어 대조했다.

full/tight mask는 목재 외곽을 대체로 따라가고 큰 바닥 그림자 모양 영역을 분리한다. 따라서 앞선 PRIMARY의 좋은N0→82° 악화 사례처럼 ‘mask가 어두운 바깥 영역을 크게 포함해 좋은 후보를 버렸다’는 동일한 설명을 이 한 장에 적용할 수 없다. 실제 B4/B5 fallback 위치·대응, 초깃값의 W/D 분기, 후보 생성에서 어느 정보가 부족한지는 별도 문제다.

기존 CAL12 실패6장 중 후보 부재5장·좋은 후보 미선택1장이라는 집계는 그대로다. 이번 E84는 **사후 진단1장**의 후보 부재·초기부터 큰 방향 오차 사례다. 원12장의 원인 비율이나 전체 CAL/PRIMARY 실패 비율로 합치지 않는다. 기존 성공 대조의 유사한 mask/가림 반례도 그대로 유지한다.

## renderer 최소 자료 경로 확인

TRAIN8의 원 renderer scene에서 **mask_visible 8개와 mask_amodal 8개가 실제 파일로 존재**했다. 대표 경로는 원 저장소의 data/pallet/training_data/paper_release/v2_prod40k_clean_merged/mask_visible/f0260.png 및 mask_amodal/f0260.png이다. 대응 label JSON과 RGB 경로도 있다. 경로/파일명/크기만 확인했으며 mask pixel·encoding·독립 foreground 의미는 아직 검증하지 않았다.

검사한 scene root의 하위 폴더는 labels/rgb/mask_visible/mask_amodal뿐이고, 그 경로에서 depth/objectID/faceID 전용 파일/폴더는 찾지 못했다. 보존 GT asset basename은 eur_pallet_bk_cc0.glb, scene.usd, scene_1.usd이나 이 원 scene·관련 G38 dataset 폴더에서 실제 mesh cache 경로는 확인되지 않았다. 로컬 전체에 mesh가 없다고 주장하지 않는다.

따라서 **독립 출처 자료가 모두 없다고 단정해 새 수작업 주석을 먼저 요구하면 안 된다.** 다음에는 이미 있는 visible/amodal mask의 저장 의미와 원 RGB 좌표 대응을 읽기 전용으로 확인할 여지가 있다. 이것으로 target 물체/배경 분리를 확인할 수도 있으나, 실제 mesh edge/face owner와 그림자의 광학 원인까지 알려 준다고 보장하지 않는다. 현 단계에서 물리 owner 확정 및 다음 모델 변경은 보류한다.

경로 확인 중 retained synthetic_data.py는 GT의 Xcf가 camera-facing dynamic native ordering일 수 있는 GT-only table임을 명시한다. v39 검산은 그 선언된 GT table에 대한 3D bijection/투영의 대수적 일치를 확인한 것이다. 독립 mesh vertex 의미까지 확인한 것으로 확장하지 않는다. v39 전체 owner permutation 진단을 ‘실제 물리 owner가 틀렸음’이나 ‘GT가 잘못됨’의 증명으로 읽으면 안 된다. 원 v39 수치와 결과를 수정한 것이 아니다.

## 실행과 파일

CPU1개2.306836초(5분 cap), 새 model forward/SAM/PnP/학습/규칙/전체 성능평가/공개 게시0. 원본·기존12장·기존숫자 원장을 보존했다.

- Pallet_CAL_E84_Posthoc_Appendix.png: clean/mask/실제 관측/저장 pose 4패널1장
- RESULTS/CASE_TABLE.csv / BOTH_FAMILY_COSTS.csv / STORED_CANDIDATE_RANKS.csv
- CLEAN_VISUAL_NOTES.json: overlay 전 메모
- RENDERER_PATH_INVENTORY_PRIVATE.json: 로컬 전체 경로 목록, 공개·전달용에는 제외
- analyze_one_cached.py: 기존 캐시 비용 및 그림; prepare.py가 사전 crop/선정 보존

기존 pallet-pose Python으로 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 환경에서 prepare.py →clean crop 검수→analyze_one_cached.py 순서로 실행했다. 원 output은 exclusive-create이므로 새 재현은 별도 root에서 한다. 새packet은 사용자 본인 Library 전달만 하며 GitHub에는 게시하지 않는다.
