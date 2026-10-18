# D1의 같은 자동 입력 절차

이 문서는 알고리즘 계약과 전체 생산·검산 결과다. 근거는 `DATA_PIPELINE_PARITY.json`과 비용·실행 원장이다. 자세 정확도는 별도 최종 평가 보고서에서 다룬다.

`d1_producer.observe(frame, bundle)`와 `solve_e(row, frame, mask, config, model, interval)`를 실사 238장과 기존 합성 1,003장 모두에 적용한다. domain/ID는 처리 규칙을 고르지 않는다. 원본 영상·N3/SubPix/F·K·팔레트 등록 치수는 봉인된 예측 캐시에서 가져온다. synthetic train768/validation235와 real TEST49/REMAINDER143/CAL46을 바꾸지 않는다.

순서는 다음과 같다.

1. 원 N3/SubPix/F 예측의 native cuboid와 Q로 S와 B/I 기하 역할을 정한다. S 검출 좌표를 영상 단계 전에 제거한다.
2. 원 `StatewiseIntervalV4_2.observe`를 호출한다. 실제 Tiny 마스크와 고정 ResNet 특징·DIRECT/EXTERNAL head, 원 T interval와 O/T domain 생성기를 실행한다. O=0 placeholder를 공급하지 않는다. 게이트 때문에 RGB query가 없으면 원 API가 forward를 생략하며 그 실제 호출 수를 기록한다.
3. non-S raw box의 tight MobileSAM과 등록 cuboid+non-S raw box의 full MobileSAM을 사용한다. tight/full support recovery와 corner-reach recovery를 원 순서로 호출한다. 기존 성공 B/I와 그 시점의 CAL covariance를 보존하는 `fixed_fit_measurements`를 사용한다.
4. 원 U region, shared supported B partial, face-aware B/I partial를 실행한다. 측정되지 않은 점을 모델 투영 관측으로 바꾸지 않는다.
5. 원 설정의 P1 seed, 두 W/D prior0 search, numeric G10, typed G13를 실행한다. G10이 정한 family를 고정한 채 같은 family G13를 최종 E0로 읽는다. typed 관측이 없으면 원 정책대로 G10을 재사용한다.
6. 이번 고정 실행에서 실제 생성된 P1/WD0/G10/G13 feasible sample·selected pose·방문 optimizer endpoint와 N0/E0만 common bank에 넣는다. 과거 연구 버전의 history-only union, 새 B finite/local 또는 A 후보는 넣지 않는다. 잘못된 endpoint는 원 journal에 남고 positive-depth/proper pose eligibility에서만 제외된다.

새 출력 이름은 **E0_REGENERATED**다. 역사적 E0=114/192 결과를 새 입력 결과로 복사하지 않는다. 두 값이 우연히 같더라도 독립 실행 증거와 실제 pose 동일성으로 구분한다. N0/E0 alias는 strict native numeric duplicate tolerance(R 최대차 1e-7, 중심 최대차 1e-7×object diagonal m)로 보존한다. C2 대칭 phase나 90° W/D 혼동은 duplicate로 없애지 않는다.

특징은 이전 31개 scalar의 signed-log1p와 31개 validity bit를 그대로 계산한다. 기존 unrepaired A/B/edge 비용을 **고정 pose에서 특징으로만** 계산하며 새 A/B/edge 후보를 생성하지 않는다. frame summary와 normalization은 L 코드가 계산한다. 모든 2×2 모델은 같은 D1 실사 feature bank를 받는다. 관측 수·covariance·state 요약은 진단 metadata이며 62차원 model feature를 늘리지 않는다.

D1은 synthetic-real 영상 차이를 해결했다는 뜻이 아니다. 기존 synthetic provision에서 허용된 고정 per-asset 등록 축과 camera를 재사용한다. 이번 producer는 per-frame GT pose/box/visibility/mask를 읽지 않는다. 원점·회전·camera/crop 규약은 upstream seal로 연결하며 영상별 GT 재등록을 하지 않는다. 새 producer API에는 참조/레이블 인자가 없고 label 생성은 별도 프로세스다.

기술 throughput은 사전 입력 순서의 CAL 1장과 synthetic 1장에서 측정했다. 최초 runtime descriptor 경로 오타는 model forward 전에 실패했으며 해당 시도를 보존했다. 경로만 수리한 관측은 각 2회의 SAM encoder/decoder, 1회의 Tiny/ResNet feature/head를 실제 호출했다. synthetic 한 장에서는 O 1개가 실제 검출됐다. 기술 시험은 일반화 평가가 아니다.

## 같은 고정 실행의 계산 재사용

기존 REMAINDER143의 P1/WD0/G10/G13 TYPED_MULTI는 현재 관측·마스크·전체 수치 spec·원 warm start·typed/legacy partial·outline·native/Q와 원 helper source가 일치했다. 이 경우에만 `d1_stage_memoization.py`가 **한 고정 e 실행의 수치 단계**를 재사용한다. 원 입력 validator, family 선택, G10 family를 따르는 G13 readout, 후보 추출, 특징 계산은 현재 producer에서 그대로 실행한다. SINGLE, V15/V17, 다른 초기화, history 후보 합집합은 허용하지 않는다. 이는 새 모델이나 선택 정책이 아니다.

정답을 열기 전에 입력 순서의 첫 REMAINDER 1장을 실제로 다시 풀어, cache 경로와 모든 후보 ID·62 특징·pose·alias·방문 endpoint·최종 E0가 정확히 같은지 검사했다. 이 실제 재실행 행은 최종 bank에 재사용한다. 원래 실행 중이던 실사 배치는 시작된 행이 모두 seal될 때까지 drain한 뒤 유휴 worker만 종료했다. 원 완료 행·소스·실패 기록은 유지하며, 새 shared dispatcher는 실사와 합성을 같은 CPU pool에서 처리한다. 도메인별 feature seal이 같은 `execution_batch_id`를 가리키므로 비용 원장은 공통 parent wall을 한 번만 센다.

전체 1,241장 관측에서 frame별 hook을 새로 만드는 계측은 이미 로드된 SAM predictor의 후속 forward를 누락했다. 원 관측과 마스크는 재생성하지 않는다. 실제 SAM 호출 수는 성공 provider receipt로 별도 재구성하고, 기록된 CUDA event 시간은 하한으로 표시한다. 전체 관측 host wall은 GPU 작업 시간의 상한이다. 고정 첫 CAL 1장의 별도 cold/warm persistent-hook 기술 실측은 전체 분포 또는 Jetson 성능으로 일반화하지 않는다. 최종 비용·실행 원장은 이 계측 공백과 모든 실패·재실행 비용을 함께 보존한다.

## 중단된 CPU 실행의 기술 복구

첫 shared 실행은 tool session exit 143으로 중단됐다. 종료 주체와 구체 원인은 확인되지 않았다. 이미 봉인된 실사 238장·합성 454장과 원 lock/journal/log는 유지한다. 봉인 전이던 합성 4개 폴더도 별도 attempt archive로 보존한다. 이 네 시도의 optimizer/PnP 호출 수와 CPU·행별 시간은 저장되지 않아 `null`로 남기며 0으로 대체하지 않는다.

`d1_recovery_batch`는 새 runtime namespace에서 **수정하지 않은 원 shared dispatcher**를 호출한다. 같은 입력·관측·기존 완료 행을 hash 검증하고 남은 549장만 같은 producer로 실행한다. 새 실행은 다른 batch UUID와 journal을 사용하며, detached supervisor가 실제 자식 exit를 기록한다. 이 실행 방식 변경은 과학 설정이나 후보·특징 정의를 바꾸지 않는다. 최종 독립 audit는 두 journal의 `(domain,index)`가 중복 없이 원 pending 1,201개를 완전히 채우는지, 전체 238+1,003장이 같은 원 source·입력·관측 bytes를 사용했는지 확인한다. 비용은 원 중단 batch의 보수적 parent-wall 상한과 재개 batch의 실제 parent wall을 각각 한 번 센다.

## 전체 생산·독립 검산 결과

`DATA_PIPELINE_PARITY.json`은 **PASS**다. 원 703개 source 파일과 실제 사용한 13개 모델·descriptor·보정 파일의 SHA를 직접 재확인했다. 실사와 합성의 모델 binding은 동일하다. 원 692개 완료 frame seal은 변하지 않았으며, 두 journal의 합집합은 원 pending 1,201개와 일치하고 중복은 0개다. 최종 재개 프로세스의 실제 exit code는 0이다. 이 검산은 참조 자세나 정확도 평가를 읽지 않는다.

| 도메인 | 전체 행 | 새 고정 e 실행 | 원 고정 단계의 exact 재사용 | O 검출 프레임 | 고유 후보 | eligible 후보 |
|---|---:|---:|---:|---:|---:|---:|
| 실사 | 238 | 96 | 142 | 79 | 8,949 | 8,699 |
| 합성 | 1,003 | 1,003 | 0 | 586 | 33,982 | 33,586 |

E0와 N0의 물리 자세가 같은 행은 실사 5개, 합성 386개다. 각 행의 실제 e 실행·원 endpoint·최종 E0 alias를 별도로 확인했으므로 이 동일성은 E0를 N0 placeholder로 공급했다는 뜻이 아니다. 두 도메인의 전체 state 집계에서 T는 0개였으며, 이번 bank는 T 처리의 유효성을 입증하지 않는다. 절차 일치는 영상 분포 일치나 실제 물리 6D GT의 인증도 아니다.

아래 시간은 worker preload 뒤 **e + 원 62개 특징 계산 + 출력 직렬화·provenance** 전체의 행별 service 시간이다. optimizer와 특징 계산의 시간 분할은 계측하지 않았다. 실사 새 실행에는 원 2-worker, 기술 parity 1-worker, shared 4-worker 환경이 섞여 있다. 이 표를 fresh RGB 입력의 배포 latency나 Jetson FPS로 해석하지 않는다.

| 행 종류 | 행 수 | 평균 초 | P50 초 | P90 초 | 최대 초 |
|---|---:|---:|---:|---:|---:|
| 실사 새 실행 | 96 | 66.49 | 52.92 | 117.37 | 238.10 |
| 실사 exact 재사용 | 142 | 7.63 | 7.18 | 11.45 | 18.87 |
| 합성 새 실행 | 1,003 | 70.22 | 35.95 | 146.48 | 2,933.05 |

`D1_COST_AND_EXECUTION_RECEIPT.json`은 서로 다른 26개 측정·상한 batch wall을 각 한 번 계수한다. D1의 알려진 시간과 선언한 중단 상한 합은 21,345.40초(약 5.93시간)다. 미계측 과거 명령·기술 실패 시간은 별도 unknown이며 exact 총비용은 `null`이다. 재개 whole-process wall은 9,257.78초이고 원 중단 shared batch 상한은 9,876.05초다. 미봉인 4개 시도의 부분 연산 수·CPU는 0으로 채우지 않는다.

전체 1,241장 관측 host wall은 545.00초다. 직접 계측한 SAM encoder/decoder forward는 각각 4회이며, 성공 provider receipt로 재구성한 실제 bulk 호출은 각각 2,482회다. CUDA event 합 5.36초는 계측 공백을 포함한 하한이다. 별도 기술 관측 4회와 추가 top-level neural forward 24회도 원장에 포함한다. 기존 detector/N3/SubPix cache의 생성 시간은 이번 측정 범위에 들어 있지 않으며 0으로 간주하지 않는다.
