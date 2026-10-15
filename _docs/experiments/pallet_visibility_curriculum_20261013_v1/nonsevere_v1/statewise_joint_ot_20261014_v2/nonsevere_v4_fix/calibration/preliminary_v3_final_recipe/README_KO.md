# CAL 전용 전체 오차 불확실성 모델

CAL 46건 중 고정 3D 대응이 맞는 44건으로 추가 분산과 prior covariance를 fit했다. 대응 불일치 2건은 제외 사유와 DIRECT annotation 오차를 별도 공개한다. TEST label/RGB를 읽지 않는다.

원 B/I `xy`와 `covariance_px2`는 유지한다. solver용 `fit_covariance_px2`는 원 조건부 covariance에 CAL에서 추정한 isotropic floor를 더한 값이다. `fit_reliability_weight`는 CAL human DIRECT 빈도를 부드럽게 반영하며 관측을 삭제하지 않는다.

원 조건부 covariance는 선 교차 resampling과 detector 2/3px 가정의 오차를 담는다. 새 floor는 zero-centered 전체 재투영 오차로 fit하여 semantic/대응/model bias를 일부 포함한다. 각 frame/session의 상관관계, 모델 부정확성, 가시성 착오가 모두 제거된다는 뜻은 아니다.

추정법은 session→frame→point 동일 가중 80% residual ellipse coverage다. B/I × corrected/fallback 그룹은 상태 pooled floor로 shrink한다. 큰 오차를 삭제하지 않고 원 오차 분포/상위 사례를 보고한다. 80% quantile 추정은 극단 오류 영향이 제한되는 robust fit이며 모든 오류를 덮는 hard bound가 아니다.

DIRECT annotation 오차와 fixed-native derived reference 재투영 오차는 별도다. 후자는 독립 물리 계측 GT가 아니다. 경험적 Mahalanobis 3 coverage를 물리적 99% 보장으로 해석할 수 없다.

LOGO는 held-out session을 제외해 scale normalization, floor, reliability, prior covariance를 모두 다시 fit한다. upstream state/BI head는 기존 all-CAL fit 입력이므로 전체 파이프라인의 완전 독립 LOGO가 아니다.

pooled floor(px²): B=72.8878, I=44.0966.

LOGO reference point count=254; Mah≤3 conditional=0.4251968503937008, effective=0.9330708661417323.
LOGO DIRECT annotation point count=254; Mah≤3 conditional=0.4409448818897638, effective=0.9291338582677166.

prior는 fixed-native geometry mapping으로 정한 reference와 coarse pose의 camera tangent/normalized translation 오차를 사용한다. MAD+bias scale, Huber weighted zero-centered second moment, diagonal shrinkage로 SPD covariance를 만든다. prior_weight [0,0.25,1,4] 선택은 별도 root CAL 검증 범위다.

런타임 apply API는 B/I records, frozen model, state 목록, nonS-only bbox scale만 받는다. human/reference/GT compatibility 조건으로 기능을 바꾸지 않는다. 초기 S는 좌표 접근 전에 차단한다. override covariance를 적용하면 pooled floor를 다시 더하지 않는다.

[MODEL.json](MODEL.json), [LOGO 보고서](CAL_LOGO_REPORT.json), [coverage 그림](CAL_LOGO_COVERAGE.png), [API 검증](API_CHECKS.json).
