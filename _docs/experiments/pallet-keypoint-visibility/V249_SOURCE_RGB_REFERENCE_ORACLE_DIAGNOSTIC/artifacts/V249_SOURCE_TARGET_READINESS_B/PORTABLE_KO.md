# V249 SOURCE q·r·reference 권한 정리

원 YCB 입력 q는 실제 initializer가 아니라, pose/K로 투영한 8개 model-surface r에 고정 ≤4px jitter를 더한 emulation이다. 라벨은 q 주변 RGB 소유자가 아니라 r의 geometry와 nearest-r 3×3 depth/mask 근거로 정한다. D/E만 binary로 쓰며 S/U는 계속 제외된다. 원 q/L/역할과 native UNKNOWN은 바뀌지 않는다.

현재 배포형 입력은 같은 actual RGB의 원 q8/L8 local crop과 q8 minmax context다. GT r/owner/object ID/pose/depth를 forward 채널로 넣지 않았다. 다른7 query crop도 입력만으로 읽을 수 있지만, teacher visibility로 보이는 query를 골라 reference로 삼을 권한은 별개다. strict input에는 cross-frame paired visible reference나 persistent physical-object track이 없다. GT-side object/model-point 관계를 쓰는 reference 선택은 oracle 진단으로 명시해야 한다.

Root가 검토하는 진단에는 기존 공개표 `YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv`(SHA8eab4fe1…/474209B)가 가장 작은 조인 경로다. writer는 row_id/sample_index/role/object/aux/q/r/distance/L을 보존한다. 원 canonical 특징을 exact row_id+sample_index로 연결할 수 있다. component/source-instance는 이 작은 표에 없으므로 UNKNOWN을 유지하거나 추후 별도 SOURCE metadata scope에서만 더한다. 이번에는 이 표나 원 SOURCE행/특징/이미지를 읽지 않았다.

SOURCE_FIT D prototype도 associated-r 라벨이 붙은 q crop의 특징이지, target-only 깨끗한 visible cutout임을 보증하지 않는다. q/r 축의 차이가 image-only 판별 불가능·라벨오류의 증명도 아니다. Root의 true-object(+slot) oracle vs fixed wrong-object cosine 진단은 이 한계를 유지해 보고할 수 있으며, 배포가능한 reference 검색이나 모델채택 근거로 바꾸면 안 된다. 기존 C10/C11에는 같은 영상 context/local cosine가 있었지만 정확히 같은 cross-source FIT-D prototype 진단은 이번 확인범위에서 찾지 못했다. 역사 전체 미시도 주장은 하지 않는다.
