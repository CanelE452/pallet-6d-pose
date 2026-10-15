# 초기 coarse/q와 clean cached query 출처

원 ORIGINAL/SWAPPED coarse, current B/I q/source/Σ, bounded S 중심 및 ORIGINAL_INITIAL guide에는 후속 e 숫자를 복사하는 경로를 찾지 못했습니다. 원 N3_SUBPIX initial geometry와 실제 영상 B/I 보정 archive를 이어받습니다. 같은 입력을 공유하므로 통계적 독립성 주장은 아닙니다.

반면 extra CACHED guide는 E-inclusive bank에서 native ranking으로 선택됩니다. 원 coarse와 구분해야 합니다. F31 q-anchor/signed-ray 검사가 generating poseguide의 fresh PnP 출처까지 인증하지는 않습니다.

후속 source 정책으로 cached 실제query를 고를 때 **FROZEN17C_ORIGINAL_MODEL_SEED + MEASURED_BI_ENDPOINT_DIRECTION 두 guide_kind만** 허용하면 원 INITIAL/extra CACHED physical-VP query는 모두 제외할 수 있습니다. 전자는 independent initial model normal과 원 qanchor, bothBI qdelta/unknown original model direction을 쓰고 후자는 원 qanchor와 qmate−qanchor만 씁니다. modelanchor를 q로 옮겨 새측정처럼 부르면 안 됩니다.

또한 rank에 옛 cached pose_guides를 넘기면 후보제외 후에도 weakguide/ordered score에 E 출처가 남습니다. pose_guides=None 또는 인증된 independent owncoarse만 사용하는 별도 계획이 필요합니다. retained actual q+known geometry/K의 fresh PnP 후보와 own coarse는 source ledger로 새로 생성해야 하고, e 숫자와 일치하는지만 보고 버리는 routing은 금지합니다.

원 raw/source/bank/model 및 현재실험 변경0, 32 실제실행0입니다. 근거: [함수·라인·SHA 및 설계 JSON](COARSE_Q_GUIDE_SOURCE_CLOSURE_V1.json).
