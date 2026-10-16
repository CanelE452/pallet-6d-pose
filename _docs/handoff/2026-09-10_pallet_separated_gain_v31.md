# V31 완료 인계

사용자 승인한 V30 우선순위 후속: 점 선택/선 추가 이득 분리, 부모 미학습 원본, 일반 방향 대조를 실제 실행했다. 운영 모델 교체나 push는 하지 않았다.

- 코드: `scripts/research/pallet_separated_gain_v31/`
- 결과: `data/pallet/results/pallet_separated_gain_v31/`
- [MD 결과와 실제 이미지](../notes/point_line_fusion/04_failure_analysis/pallet_separated_gain_v31.md)
- [현재 판단](../notes/point_line_fusion/STATUS_AND_NEXT_STEPS.md)
- 새281원본=153학습/64하한 진단/64합성 확인. 기존 YOLO train 및 V6/V21/V23 요청 그룹·정확한 이미지 해시 중복 제외. 역사적인 YOLO validation 노출은 남는다. 새 scene/asset 독립성이나 자연 실사 holdout이라고 주장하지 않는다.
- 3시드21체크포인트·실제27,000 optimizer updates. Joint old/new 각2000, selector old/new 각1000, split old/new 및 generic new 보정 각1000. Selector 복사를 독립 학습으로 중복 계산하지 않는다.
- 분리 모델은 실사 모든 조건에서 선 적용률0. 잘림20에서 selector_new=split_new=generic_new=15.6288px. V30=13.3845, base=12.372px. 큰 잘림19.386px로 강한 점19.382px와 유사하나 전체 안정성 미통과.
- 하한 제거15.842px, 평균 이득만15.900px로 잘림20이 좋아지지 않는다. GT로 선택된 부모의 현재 후보만 고른 진단도15.154px다. 이는 다른 부모 선택까지 허용한 oracle과 다르다.
- 신규 학습 부모 우열 AUROC0.643, 선 후보0.534. 별도 합성 잘림20 부모0.500/선0.487. 실사 해당조건 부모0.429/선0.541. 학습 전체와 조건별 평가의 모집단 차이를 명시한다.
- 추가 읽기 검사에서 실사 원본/코너가림20/잘림20/잘림40 모든 시드의 positive-lower candidate 수0 확인. Positive-rank 후보는 대부분 존재한다. 선 중단의 직접 조건은 하한이며, 그것을 없앤 추론도 개선하지 못했다.
- 데이터 전환 시 features25의 전역 캐시 디렉터리가 유지되는 문제는 seed2 학습 시작 전 assert에서 잡혔다. legacy read 동안 디렉터리를 복원하고 재개. Seed1 원래 코드 해시와 범위는 CACHE_SCOPE_FIX.json, 입력/모델/손실 변경 없음.
- AUDIT.json: 이전V30 86개 bound파일 무변경,402개 분리 예측 지표/기하 수치 재검산, 실제 부모+모듈 재추론, 학습된 selector 고정 및 generic 선 불변성 검사 통과.
- HTML 표84셀 검증, 실제 Chrome 창 열림, Discord 완료 한 문장 HTTP204 전송 완료. 재전송하지 않는다. COMPLETION.json으로 결과 봉인 완료.

현재 우선순위: 동일 특징에서 선택기/손실 변경 반복을 낮추고, 가시 경계·끝점·점 역할 대응과 남은 면의 관측 정보가 후보 우열에 추가 정보를 주는지 검증. 물리적 선 GT 감독 효과는 아직 미실행/미확인이다. 이번 제한된 예산의 실패를 adaptive 융합 전체의 불가능성으로 일반화하지 않는다.
