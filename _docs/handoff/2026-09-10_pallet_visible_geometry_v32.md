# V32 결과 인계

사용자가 승인한 실제 가시 경계/끝점 감독과 점·선 정보 보강을 실행했다.

- 코드: `scripts/research/pallet_visible_geometry_v32/`
- 결과: `data/pallet/results/pallet_visible_geometry_v32/`
- [MD와 실제 이미지](../notes/point_line_fusion/04_failure_analysis/pallet_visible_geometry_v32.md)
- [현재 판단](../notes/point_line_fusion/STATUS_AND_NEXT_STEPS.md)
- 2USD 자산240RGB / 600요청 관측 중436검출. 외관16학습/8확인 그룹, 같은2자산 공유. CAD 가시 직선 체인과 물리 끝점, 근접 평행 cuboid 역할. 실제 인간 의미 경계의 완전한 정답이 아니다.
- 기존 부모 학습과 분리한153원본 관측과 새 렌더를8:8 혼합. 실제 수동 GT는 수정/학습하지 않았다. 실사297/9세션은 재사용DEV, 정본140/논문/봉인 결과와 구분한다.
- 주대조 point/latent_hough/virtual_hough/visible_hough 각3시드×2000, 이후 visible_aux_only 3시드×2000. 실제15모델30,000업데이트. aux-only는 사후 탐색 대조다.
- 잘림20 새 점16.331 / virtual17.030 / visible14.343 / aux-only15.297px. 기준점12.372/V30 13.385보다 나쁘다. 코너 가림14.185는 강한 점11.907의 이득을 보존하지 못했다. 큰 잘림17.550px는 좋아졌다. 전체 안정성 미통과.
- 잘림20 Hough만 끄기14.359 / 끝점만 끄기15.537 / 모두 끄기15.593. 다른 세션 역할맵14.627. 인접 batch donor는 같은 세션 프레임이 많아 약한 대조였으며 별도 cross_session_donor로 보완했다. 선/끝점 제거 재학습 대조와 혼동하지 않는다.
- 실제 경계 감독 vs 가상 감독 렌더 micro F1: edge .570 vs .393, endpoint .528 vs .218. aux-only .578/.540으로 분류는 더 좋지만 점 오차는 더 크다. 분류 능력과 보정 효용을 구분한다.
- AUDIT: 이전824개 bound 파일 무변경,660예측 파일 지표 재검산,15모델 학습 로그/동일 초기화·배치,수동GT297×8행 좌표/valid/hash,6회 GPU 재현,원래 학습 소스 정상forward 동등성, crop가 물리 끝점을 만들지 않는 검사 통과.
- 추론만의 Hough/endpoint 끄기 추가는 `INFERENCE_ABLATION_AMENDMENT.json`과 원래 소스 스냅샷에 기록. catastrophe용 original_points 메타데이터가 base로 저장됐던 것을 stock YOLO로 수정하고 재집계했으며 출력점/GT/체크포인트는 변경하지 않았다.
- HTML 실제 창·결과 해시·Discord 전송 여부는 `DELIVERY.json`, `COMPLETION.json`, `DISCORD_NOTIFICATION.json`을 확인한다. 이미 전송됐으면 재전송하지 않는다.

후속 우선순위: 동일 물리 감독 하에서 전역 pooled Hough/끝점과 점 역할별 선 support/끝점 대응을 비교하고, 좋은 점 부모 보존을 별도 검증한다. 공유 trunk의 점/보조손실 gradient 충돌 측정과 전용 trunk 분리를 먼저 보고 PCGrad를 판단한다. V32는 PCGrad 미실행. 원천적 융합 불가능이나 안정적 성공으로 일반화하지 않는다.
