# NEGATIVE_SYNTH_V1 DATA AUDIT

```
semantic 위반   0 / 10,000
collision       0
실제 팔레트 혼입 0  (육안 192장)
filtered manifest = 전체 (제외 프레임 없음)
```

## 구성

```
train 9,000 / dev 1,000    중복 id 0 · rgb 결손 0 · label 결손 0 · source_mode 전부 FRESH

negative_type          train    dev
N0_MATCHED_EMPTY        3,600    400     빈 장면
N1_STRUCTURAL_HARD      3,150    350     평행 rail·교차 구조·블록
N2_PALLET_LIKE_HARD     2,250    250     벤치·랙·사다리·게이트 등 팔레트 유사 topology

impostor_type   none 3,600 / parallel_rail_bundle 1,965 / junction_cluster 906
                / asset 276 / sparse_structural_hybrid 2,253
해상도          640x480 4,403 · 960x540 2,364 · 720x480 1,330 · 560x560 903  (BROAD 와 동일 4종)
```

## contract — 값으로 확인했다

```
object_present   = False        keypoints        = []  (len 0)
pose_valid       = False        structural_lines = []  (len 0)
objects          = []           projected_cuboid 없음
```

⚠ **내 첫 감사가 "위반 18,000" 을 냈는데 오류였다.** `keypoints`·`structural_lines`
키의 **존재**만 보고 위반으로 셌다. 실제로는 키가 있되 값이 빈 배열이고 그게 정상
contract 다. 값 기준으로 다시 세니 위반 0.

## collision

```
NEG train ∩ NEG dev     파일명 기준 1,000  →  실제 0
                        두 split 이 각자 f00000 부터 매긴다
                        train f00000~f08999 / dev f00000~f00999
                        같은 이름 RGB 10장 해시 비교 동일 0
                        neg_id 기준 교집합 0
NEG ∩ MH_TRAIN          0
```

⚠ 로더에서 두 split 을 같은 네임스페이스로 합치면 덮어쓴다. manifest 에 split 을 포함시킬 것.

## 시각 QA (카테고리별 64장 = 192장, deterministic seed 20260831)

```
1 실제 pallet 혼입              없음 ★
2 pallet-like 가 실제 pallet asset 인가   아님. 벤치·랙·사다리·게이트·프레임 계열
3 empty scene 과다              N0 40% — 과하지 않음. 나머지 60% 가 구조물
4 너무 작거나 멀기만 한가        육안상 N1/N2 에 작고 먼 impostor 가 많다 [추정]
                                라벨·manifest 에 크기/거리 필드가 없어 정량화 불가
                                → PHASE 9 의 category 별 score 분포로 경험적으로 답한다
5 단일 background/asset collapse  없음. 900장 dHash 고유 900/900, 중복 0
                                밝기 median 48~59, p10~p90 이 18~94 로 넓다
```

contact sheet: `negative_qa/qa_{N0_MATCHED_EMPTY,N1_STRUCTURAL_HARD,N2_PALLET_LIKE_HARD}.png`

## 제외 프레임

**없다.** 실제 팔레트가 보이는 negative 를 찾지 못했으므로 filtered manifest 는 전체와
같다. 데이터셋 자체는 수정하지 않았다.

## 한계

- 시각 QA 는 카테고리당 64장 표본이다. 10,000장 전수 육안 확인은 하지 않았다.
- impostor 의 화면 점유율·거리 메타데이터가 없어 QA 항목 4 를 metadata 로 답할 수 없다.
  `background_id`·`hdri_id` 도 manifest 에서 전부 null 이라 provenance 기반 다양성 감사도
  불가능했고, 대신 이미지 dHash 로 대체했다.
- `neg10k_persample.jsonl` 은 7,161 행으로 10,000 보다 적다(FRESH 생성분만으로 보인다).
