# 合성 part-ID visibility 경로의 실행 가능성

별도 part 마스크를 만드는 방향은 가능하지만 **지금 로컬 renderer로 바로 생산 가능한 상태는 아닙니다**. 기존 virtual cuboid8은 그대로 두고 새 asset-local partID를 따로 정의해야 합니다.

| 실제 로컬 자산 | 확인한 node/material 구조 | 현재 해석 |
|---|---|---|
| scene.usd |7Mesh·1Material·GeomSubset0, Object_n 이름 | 부분이 분리돼 있지만 foot/topplank 의미 미검증 |
| scene_1.usd |1Mesh·1Material·GeomSubset0 | authored part 분리 없음 |
| scene_2.usd |1Mesh·1Material·GeomSubset0 | authored part 분리 없음 |
| scene_3.usd |308Mesh·4Material, Baseboard/Topboard 이름의 독립Mesh 존재 | 신규 topplank part prototype 후보. 기존1003 nativeasset 대응은 UNKNOWN |
| pallet_full.obj |object1·material1·group0 | photogrammetry 전체mesh이며 part annotation 없음 |

실제 USD prim/type/material-binding metadata만 `LoadNone`으로 읽었습니다. 정점·face·subset indices·RGB·texture는 읽지 않았습니다. Topboard라는 이름만으로 물리부위 gold를 선언하지 않았습니다.

제안하는3pass는 선택part 단독의 **P_A**, 팔레트 다른 부위를 복원한 target-only의 **P_S**, foreign object를 추가한 **P_V**입니다. `P_A & ~P_S`는 자기부위 뒤로 가려진 projectedpart, `P_S & ~P_V`는 foreign cover, `P_V`는 기대part의 실제ID입니다. 다른 board가 보이더라도 다른 ID이므로 기대part를 D로 세지 않습니다. 기존 whole-target M4나 virtualray와 다른 감독입니다. 다만 topboard 전체가 조금 보인다는 것과 특정corner/part-line이 보인다는 것은 별개여서 part 내부 cue 정의는 추가 검증이 필요합니다.

로컬 구형 `render_blender_data.py`는 pallet children 전체 AABB로 virtualcuboid를 만들고, 가시성은 in-frame 비율입니다. part-ID export는 없습니다. 문서에 있는 production v2 writer·portable blend·GLB2개는 Windows 증거이며, 대응 로컬 위치에서는 찾지 못했습니다. Blender는 PATH와 알려진3경로에 없고 실행을 검증하지 못했습니다. 기존 pallet-pose env의 USD metadata reader는 실제로 작동했습니다.

작은 후속 prototype은 한 신규asset의 검증된Topboard 하나와 고정6camera·256²RGB+PA/PS/PVmask로 제한할 수 있습니다. 비압축 pixel 출력 추정은 약2.36MB이며 큰scene blend를 복제하지 않을 수 있습니다. 그러나 이는 저장 예산의 가능성일 뿐 실제 render0입니다. 먼저 renderer 실행환경·asset part 의미를 해결해야 하며 새fit은 하지 않습니다. nativecorner↔part 연결이 불명확하면 UNKNOWN으로 유지하고 nearestmesh/surface snap을 하지 않습니다.
