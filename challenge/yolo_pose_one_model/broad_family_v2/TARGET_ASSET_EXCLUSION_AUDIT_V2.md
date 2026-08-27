# TARGET ASSET EXCLUSION AUDIT — V2 (mesh 수준)

**EXACT_OVERLAP 0 / NORMALIZED_GEOMETRY_OVERLAP 0 / MESH_EXCLUSION_EXACT = PARTIAL**

V1 은 4 asset 중 **0 개**만 파일 단위 대조가 가능했다. `usd-core` 로
USD 2 개를 실제 파싱해 이제 **2/4**.

## 서명

```
asset                                    verts     faces   aspect    thick
TARGET pallet_full.obj                 186,036   180,040    1.182   0.0923
scene.usd (Pallet_0)                   413,451   426,540    1.200   0.1250
scene_1.usd (Pallet_1)                   4,539     4,390    1.203   0.1143
```

## 대조

```
scene.usd
  exact vertex hash        False
  회전불변 형상 hash        False
  vertex/face 수 일치      False / False
  형상 히스토그램 L1       0.2556   (0 이면 동일)
  치수비 L1                0.0455
scene_1.usd
  exact vertex hash        False
  회전불변 형상 hash        False
  vertex/face 수 일치      False / False
  형상 히스토그램 L1       0.3952   (0 이면 동일)
  치수비 L1                0.037
```

## 아직 닫히지 않은 것

```
woodpallet_block_jtoastie_ccby.glb           Pallet_2  파일이 이 머신에 없다 (렌더는 Windows)
eur_pallet_bk_cc0.glb                        Pallet_3  파일이 이 머신에 없다 (렌더는 Windows)
```

이름이 CC 라이선스 인터넷 모델처럼 보인다는 것은 **근거가 아니다.**
렌더 머신에서 두 GLB 를 회수하면 이 항목이 닫힌다.

