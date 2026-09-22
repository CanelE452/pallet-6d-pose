# 실행 완료 및 공개 범위

2026-09-22 Downloads의 최신 두 문서 `pallet_crop_completion_cli_v2.txt`, `pallet_crop_completion_experiment_plan_v2.txt`를 읽고 실행했다. 더 최근 계획의 두 축 판정을 적용했다. 이전 E0 판정은 보존했다.

- 신규 학습: C 한 번, GPU 300 update. A는 기존 FULL125, B는 같은 가중치의 확대 추론, D는 C 가중치의 원래 크기 추론이다.
- 동일 실사 253장 및 source 순서·원영상 좌표 교란을 유지했다. 새 학생 self-training, sweep, rescue 학습은 없다.
- 최종 테스트 32개 통과. 과거 파일 1,431개 불변, BN 통계·affine 동결, A/B/C/D 예측을 모두 고정한 뒤 평가했다.
- GPU 최대 68°C, 최대 3,332 MiB. 재부팅·드라이버 변경 없음.
- N14 복구는 모든 조건에서 ≤10px/≤20px 모두 0개다. C의 primary ≤10px 정답은 A와 같은 359/713개다.
- D는 기존 hard 중 11개를 복구했지만 primary 총 정답은 354개로 A보다 5개 적다. 진단 조건 D를 사후 선택하지 않았다.
- C의 고정 real TRAIN probe는 ≤10px 100%다. 이는 전체 TRAIN 또는 독립 평가의 성공을 뜻하지 않는다. 표현 가능한 hard 148개 중 130개에 시험한 자기 채널 top5 후보가 없었다. RGB 정보 자체가 없다는 결론은 아니다.
- source clean 정답은 1,847→1,817개로 감소하고 stress는 1,369→1,418개로 증가했다. 단순한 전반적 개선으로 보고하지 않는다.
- 사람 검토는 **REVIEW_PENDING**이다. U29 중 기존 검토 양식 연결 1개, 동일 양식 추가 28개를 준비했지만 사람 응답을 만들어 넣지 않았다. 독립 물리 좌표·가시성 검증 완료를 주장하지 않는다.

준비 중 JSON 정수 직렬화와 첫 평가 집계의 빈 band dtype 문제는 각각 [준비 기록](PREPARATION_IO_NOTE.md), [집계 기록](SCORING_IO_NOTE.md)에 남겼다. 프로토콜·좌표·예측·학습 횟수는 바꾸지 않았으며 실패 시도 기록도 로컬에 보존했다.

공개: [보고서](REPORT_KO.md), [갤러리](GALLERY.html), 비교 이미지 38장, 집계 수치 `PUBLIC_RESULTS.json`, 실험 구현·검산 코드. 비공개: checkpoint, 원본 prediction/heatmap, 원본 이미지 목록·해시 결합표, 블라인드 검토 매핑·사람 응답. 관련 없는 미추적 연구 파일은 포함하지 않는다.

코드는 기존 로컬 연구 모듈과 동결 자료에 의존한다. 공개본만으로 독립 재실행 가능한 패키지는 아니다. [코드 실행 순서와 의존성](../../../scripts/research/pallet_posefix_crop_completion_v2/README.md)을 함께 제공한다. 최종 모델·논문 표는 변경하지 않았다.
