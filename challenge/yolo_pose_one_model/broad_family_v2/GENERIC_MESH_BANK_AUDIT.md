# GENERIC MESH BANK — 로컬 실측

목표는 파일 수가 아니라 **독립 topology** 다. 회전·평행이동·균일스케일
불변 서명으로 near-duplicate 클러스터링을 먼저 했다 — 스케일 변형은 같은
클러스터로 묶이므로 unique instance 를 부풀리지 않는다.

```
{
 "raw_mesh_files_examined": 6,
 "mesh_readable": 6,
 "independent_topology_clusters": 6,
 "clusters": [
  {
   "id": 0,
   "members": [
    "scene.usd"
   ]
  },
  {
   "id": 1,
   "members": [
    "scene_1.usd"
   ]
  },
  {
   "id": 2,
   "members": [
    "scene_2.usd"
   ]
  },
  {
   "id": 3,
   "members": [
    "scene_3.usd"
   ]
  },
  {
   "id": 4,
   "members": [
    "SM_PaletteA_01.usd"
   ]
  },
  {
   "id": 5,
   "members": [
    "SM_PaletteA_02.usd"
   ]
  }
 ],
 "new_topology_not_in_broad": 4,
 "license_safe_confirmed": 0,
 "license_note": "확인된 것 0 건. Isaac asset 은 NVIDIA Omniverse EULA, project USD 는 출처 미상. 논문 배포 전 반드시 확인해야 한다.",
 "coverage_cells_total": 9,
 "coverage_before": [
  "MID/MID"
 ],
 "coverage_after_local": [
  "HIGH/THICK",
  "LOW/THICK",
  "MID/MID",
  "MID/THICK"
 ],
 "target_cell": "MID/THIN",
 "THIN_stratum_available_locally": false,
 "BLOCKER": "독립 topology 가 6 개뿐이다. G_CONSERVATIVE(+8) 조차 로컬 자산으로 도달할 수 없다. 외부 mesh 확보가 렌더의 선결 조건이다."
}
```

## 후보별

```
asset_id              status           verts  aspect   thick          cell  cluster  in BROAD
scene.usd             OK             413,451   1.200  0.1250       MID/MID        0      True
scene_1.usd           OK               4,539   1.203  0.1143       MID/MID        1      True
scene_2.usd           OK               1,775   1.006  0.1815     LOW/THICK        2     False
scene_3.usd           OK             172,656   1.712  0.5842    HIGH/THICK        3     False
SM_PaletteA_01.usd    OK                 514   1.210  0.1740     MID/THICK        4     False
SM_PaletteA_02.usd    OK                 490   1.510  0.1715    HIGH/THICK        5     False
```

평가 대상 cell = **MID/THIN** (aspect 1.1818, thickness 0.0923)

★ target cell 을 겨냥해 asset 을 만들지 않는다. 위 표는 현재 support 가
어느 cell 에 몰려 있는지를 보이기 위한 것이다.

