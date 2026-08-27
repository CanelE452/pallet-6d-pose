# CURRENT ASSET FAMILY AUDIT

★ 같은 mesh 를 스케일한 frame 은 새 instance 로 세지 않았다.

```
{
 "total_frames": 40000,
 "unique_source_assets": 4,
 "unique_mesh_instances_verified": 2,
 "unverified_assets": 2,
 "effective_asset_count_exp_entropy": 3.999,
 "single_asset_max_share": 0.2545,
 "note": "frame 별 W/D/H 스케일 랜덤화는 mesh 다양성이 아니다. unique mesh 는 4 이고 그 중 2 개만 실제 mesh 로 검증됐다."
}
```

## asset 별

```
asset                                 type       frames   share  mesh?    verts  m.thick   frame thick min/med/max
eur_pallet_bk_cc0.glb                 Pallet_3    10182   0.255  False        0   0.0000   0.0897 / 0.1206 / 0.159
woodpallet_block_jtoastie_ccby.glb    Pallet_2    10099   0.253  False        0   0.0000  0.1173 / 0.1562 / 0.2065
scene_1.usd                           Pallet_1    10095   0.252   True    4,539   0.1143  0.0864 / 0.1144 / 0.1514
scene.usd                             Pallet_0     9624   0.241   True  413,451   0.1250  0.0956 / 0.1252 / 0.1666
```

평가 대상 두께비 **0.0923**, 종횡비 **1.1818**

