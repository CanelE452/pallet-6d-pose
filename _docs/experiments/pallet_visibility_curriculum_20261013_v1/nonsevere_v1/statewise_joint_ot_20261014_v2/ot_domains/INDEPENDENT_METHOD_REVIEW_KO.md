새 구현의 구조는 사용자 정의와 맞습니다. 초기 기하 S의 좌표를 영상 경로 전에 지우고, B/I는 현재 영상의 서로 다른 선 증거로 보정한 뒤 관측 좌표와 공분산을 고정합니다. O/T는 가짜 점 대응 없이 모든 개별 영역 제약을 함께 만족시키는 연속 SE(3) 최적화에 들어갑니다. P3P/SQPNP 및 전역 회전들은 시작점이며 최종 자세를 기존 유한 후보 목록에서만 고르는 방식이 아닙니다.

완료된 코드 검토에서 B/I-only seed의 None 처리와 없는 available 필드 사용, 실패 결과의 O/T 개수 집계, crop 입력 seal 문제는 수정된 최신 소스로 확인했습니다. 물리 좌표의 행 벡터 변환 X@Q와 기존 native mapping 사용도 맞습니다. 주 자세 평가는 CAL46/TEST49 전체와 fallback을 포함하고 crop을 별도로 구분하며, study의 GT 좌표 접근은 prediction seal 검증 뒤에 시작합니다.

이 검토 시점의 추가 확인 항목과 수정 결과는 다음과 같습니다.

- 최신 solver는 B/I seed의 강체 투영에 대한 2px 이동 보호와 별개로, 각 B/I 관측의 Mahalanobis residual ≤3 제약을 매 SLSQP 계산에 함께 적용합니다. 각 관측 residual과 두 보호 조건의 통과 여부를 seed/candidate/final audit에 기록하며, 공분산과 이 제약은 보정된 확률 보장이 아닌 가정으로 명시합니다. API도 이 관측 제약을 어긴 seed를 S fallback으로 쓰지 않습니다. 이 문제는 수정된 소스로 확인했습니다.
- 새 GEOMETRY_RAW_SPATIAL 판정기는 비-S 초기 투영 위치의 현재 RGB를 가이드로 조회합니다. 최신 소스는 해당 family에서 projected_points_sampled=True와 gated projected_RGB_queried_ids를 기록하고 projected_coordinates_used_as_observations=False를 분리했습니다. 이 진단 오류는 수정된 것으로 확인했습니다.

고정 S가 잘못 판정될 위험은 남습니다. 초기 자세가 틀리면 실제로 보이는 점도 제외될 수 있으므로 S 영상 항 0과 실제 가시성 판정의 정확성은 별개입니다. O 영역은 CAL의 실제 외부 가림 오차 7개, 두 촬영 세션에서만 학습한 보수적인 경험적 max 반경입니다. 95% 공동 보장은 없습니다. 실제 O 판정기의 일반화는 sealed TEST49의 전체 혼동표와 자세 결과를 확인해야 하며, CAL 교차 검증만으로 해결 완료를 주장할 수 없습니다.

자연 다중 O는 CAL 세 장이고 TEST에는 없으며, 자연 T 라벨도 없습니다. 실제 RGB crop 세 장의 expected T와 postseal GT로 확인한 physical T를 구분해야 합니다. Crop은 기존 backbone 좌표와 K를 정확히 옮기는 조건부 replay이며 새로운 backbone crop forward는 아닙니다. 방향을 모르는 T의 outside union이 256개 branch 제한을 넘으면 명시적인 실패를 반환하여 일부 제약만 만족한 성공으로 바뀌지 않습니다. 다중 시작점의 지역 최적화는 전역 유일성이나 전역 불가능성 증명이 아닙니다.

자체 domain 테스트 25개와 실제 crop 픽셀/K/query/domain binding smoke 세 개가 통과했습니다. Solver/API 담당자는 새 불일치 B/I 사례를 포함한 focused 테스트 18개 통과를 보고했습니다. 이 검토는 peer/root 코드에 대한 읽기 전용 구조 검토이고, study 결과나 TEST GT 좌표는 열지 않았습니다. 기존 TEST 재사용과 파생된 reference pose의 제한을 유지해야 합니다.

실제 첫 CAL 다중 O 영상의 guarded runtime smoke도 seal 뒤 contract audit으로 확인했습니다. O0·3의 모든 개별 제약과 B/I의 두 보호 조건이 통과했고, S6·7 원래 좌표의 극단적 변화에도 Tiny/SAM mask, 판정, B/I 관측, 영역 및 자세가 같았습니다. 최대 B/I 관측 residual은 4.601px이고 seed 투영 이동은 최대 0.203px였습니다. 따라서 2px 이동 보호를 관측 오차 2px 보장으로 쓰면 안 됩니다. 이 실제 영상 검사는 보인 CAL 사례와 첫 판정 head의 실행 구조 확인입니다.

현재 main frozen protocol의 코드 SHA와 검토 SHA가 모두 같고, 주 CAL46/TEST49와 별도 crop 세 장의 총 98개 prediction ID를 확인했습니다. Prediction 함수에 per-image GT 좌표나 human state가 들어가지 않으며, CAL GT의 영역 불확실성 학습은 별도 허용된 학습 단계입니다. 전체 TEST49의 성능과 crop physical T 판정은 main postseal scoring이 끝나야 완료입니다. 증거는 ../runtime_smoke_guarded_v2/CONTRACT_CHECKS.json, ../runtime_smoke_guarded_v2/PREDICTION_SEAL.json 및 ../evaluation_v1/PROTOCOL.json에 있습니다.
