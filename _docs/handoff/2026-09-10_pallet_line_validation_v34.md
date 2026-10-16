# V34 실행 인계

사용자가 V33 후속 검증을 승인했다. 선만 추가한 점 보정 구조의 결정적 반복 학습과 선 위치에 따른 보정 방향을 검증한다.

- 코드: `scripts/research/pallet_line_validation_v34/`
- 결과: `data/pallet/results/pallet_line_validation_v34/`
- V33 구조/손실/초기값과 V32 데이터/배치를 그대로 사용. line_only/pooled ×3시드×별도 프로세스2반복×2000=12모델24,000업데이트 계획.
- strict deterministic algorithms, cuDNN deterministic, CUBLAS :4096:8, FP32/TF32 off. 반복별 state/전체 trace/예측 일치와 성능 안정성을 별도 판정.
- V33 기존 line_only와 V34 결정적 line_only를 대상으로 고정 영상/점에서 선 geometry만 ±4/8px 이동, geometry donor/appearance donor 분리. 개입 선택은 GT 없이 attention으로 한다. 원본/가림/잘림 및 분리 합성에서 분석.
- 반복 재현은 모델 정확도나 미지 데이터 일반화 보장이 아니다. 반사실 특징은 off-manifold일 수 있다. 기존 물리 CAD 체인과 가상 cuboid 점 역할 차이는 유지.
- 실사297/9세션 재사용DEV 계보이며 정본140/논문/봉인과 구분. 수동GT/기존 완료 결과 수정 없음.
- 완료 여부/고정 기준은 PROTOCOL.json, TRAINING_COMPLETE.json, REPLICATION.json, SUMMARY.json, COUNTERFACTUAL.json, AUDIT.json, COMPLETION.json에서 확인한다.

## 실행 중 발견 및 수정

- Strict deterministic adaptive_avg_pool2d backward가 지원되지 않아 첫 시도는 optimizer 업데이트0회에서 중단됐다. 원본 코드/로그/빈trace는 `archive/unsupported_adaptive_pool/`에 보존. 모든 크기가 나누어지는 점을 이용한 fixed avg_pool2d로 대체했고 float64 출력/gradient 동등성과 전체 strict backward 사전검사 통과.
- 그래도 첫12모델 반복은 달랐다. 고정 실제 batch/최종가중치의 forward는 정확히 같았지만 gradient가 달랐다. 개별 grid_sample input gradient에서 최대1.19e-7 변동 관찰. `NUMERIC_REPLAY.json`에 flags와 차이 보존.
- 고정 grid의 bilinear zero-padding/align_corners=True를 dense 선형연산자+bmm으로 계산하는 `sampler34.py`를 추가. float64 기존 출력/gradient 차이<1e-12, float32 반복 gradient 정확 일치 사전검사 통과. Grid 좌표 gradient는 지원하지 않으며 현재 frozen 부모에 한정한다.
- 새12모델을 `dense/models/`에 별도 학습. 최초12모델은 그대로 보존. 총 계획/실제 예산은24모델48,000업데이트로 늘었고 동일 데이터/초기값/손실/판정 기준 유지. `DENSE_PROTOCOL.json`에 이유와 source hashes 등록. 두 구조의 세 시드 모두 가중치와 전체trace가 반복 간 정확히 일치했다. 최종300평가 예측도 반복 간 정확 일치했다. `DENSE_REPLICATION.json` 확인.
- 후속 evaluation은 최종dense12모델, counterfactual은 기존V33 line_only와 최종dense line_only repeat0의3시드. 코드의 source 이름 deterministic은 이 후속 모델을 뜻한다.

## 최종 검증 결과

24모델48,000업데이트 완료. 최종dense12모델은 두 반복의 모든 state/trace/예측 정확 일치. 점+선13.674px vs 전역15.937px, 세시드15.36/7.76/18.67% 개선,9세션차이95%구간[-2.927,-1.628]px. 기준12.372px 및 가림 강한 점11.907px 보존은 실패. 전체 accepted=False.

V33/최종V34 모두 선 위치 방향 반응·geometry donor 효용 gate=False. V34 geometry donor 잘림20outside 변화+0.00027px,CI[-0.00204,+0.00203]; appearance donor+0.16986px,CI[+0.03575,+0.33774]. 명시적 위치보다 영상 특징에 민감한 자유 head 경로가 남은 원인 가설. 선 정보 전체가 무용하다는 결론은 아니며 역할/offset 원인은 독립 미확정.

AUDIT PASS: 24모델48k,300평가파일36반사실파일,2376행(297×8조건)수동GT 전달검사,이전1061bound파일/기존입력 불변. 클릭 정밀도 새 교정 없음. 보고서는 `_docs/notes/point_line_fusion/04_failure_analysis/pallet_line_validation_v34.md`. 후속 역할/offset-잔차 설명력 대조는 아직 미실행. 실제 HTML 표시와 최종파일 hash/알림은 COMPLETION/DELIVERY/DISCORD_NOTIFICATION에서 확인한다.
