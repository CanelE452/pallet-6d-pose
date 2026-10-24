# V93 실제 scene_3 geometry export

정렬된308개 Mesh를 원본 geometry 그대로 내보냈습니다. 정점172,656개, triangle110,478개, material4개이며 모든 face가 이미 triangle이어서 triangulation이나 면 수정은 하지 않았습니다.

`SCENE3_GEOMETRY_ASSET_LOCAL_ONLY.npz`는3,082,047byte입니다. raw array 합계31,972,814byte로32MB 이하이며 object dtype 없이 `allow_pickle=False` roundtrip이 일치했습니다. local/world 좌표·USD rowmatrix·authored normal·perface asset-local partID/materialID·offset·numericvalid를 보존했습니다.

선언은 metersPerUnit1·Y-up입니다. world extent가 약11.97×1.44×9.89라서 실제 팔레트 물리척도로 검증된 metre라는 주장은 하지 않습니다. 선언 단위만 적용했고 임의1.2m normalization이나 Y/Z 변환·centering은 하지 않았습니다. column-vector 연산자는 저장 rowmatrix를 transpose해야 합니다. Gf transform과 독립 NumPy rowvector probe 차이는 최대1.11e-16입니다.

NaN/invalid index/zeroarea/repeatedvertex/missingnormal/negativehanded/singulartransform 진단은 전부0이며110,478면을 전부 유지했습니다. 모든Mesh가 rightHanded이고 holes0입니다. Watertightness나 part의 의미를 검증한 것은 아닙니다.

사전 규칙의 lexicographic Topboard 후보는 asset-local partID152, `/World/root/GLTF_SceneRootNode/Large_Topboard_001_449/Object_604/Object_300`입니다. 이름상 후보일 뿐 semantic role·native8 대응은 UNKNOWN입니다. 기존1003asset의 대체나 nativecorner surface-snap은 하지 않았습니다.

외부 payload/layerreference는 발견되지 않았으나4material은 모두 누락된 `gltf/pbr.mdl`을 참조합니다. source author/license도 현재 scoped local manifest로 확정하지 못해 UNKNOWN입니다. geometry ID prototype에는 활용 가능한 구조이나 photorealisticRGB·semantic teacher·학습 admission이 준비됐다는 뜻은 아닙니다.

렌더·라벨·학습·learnedmodel·RGB/texture decode·current223·PnP는0입니다. pre-export probe/compile의 문법 오류는 실행 전 멈췄고 provenance에 보존한 뒤 V2코드로 export했습니다. 기존 asset·V92 파일을 변경하지 않았습니다.
