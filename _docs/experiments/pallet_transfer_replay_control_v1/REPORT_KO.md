# 소량 실사 전이학습/replay 통제 실험 — CPU 사전 감사 후 기술적 중단

2026-09-14. 시작 main: `d0d416eb21fea1a2a932d61ab94ded98edbe638d`.

현재 정본은 이 보고서와 `CURRENT_RESULT.json`이다. **12개 fit 중0개 실행,
본 학습0update, smoke0update, 새 모델 평가0회. 성능 결과 또는 과학적 FAIL이 아니다.**
이 보고서는 학습 완료 보고서가 아니며 CPU 사전 감사만 완료했다.

## [확인] 자원 중단

샌드박스 내 실패 뒤 호스트 권한으로 세 차례 확인했다. 모두
`Failed to initialize NVML: Driver/library version mismatch`(exit18)였다.
로드된 커널 모듈은 `580.173.02`, NVML은 `580.178`이었다. GPU ID/VRAM/다른
작업은 조회할 수 없었으므로 GPU busy 또는 CUDA 부재로 단정하지 않는다.
드라이버/전력 설정 변경, 프로세스 종료, 재부팅, 외부 알림은 하지 않았다.
세 확인 사이에 CPU 감사를 진행했으며 더 이상 GPU를 반복 조회하지 않는다.

## [확인] 완료된 CPU 범위

- R0 파일 SHA256 `970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7` 일치.
- 기존 random 선택30장, pool174장, 평가145장·4촬영세션을 그대로 확인했다.
  pool/evaluation의 파일 SHA와 연결촬영 세션 교집합은0이다.
- negative2689장과 replay1440장의 실제 이미지 접근 및 SHA를 확인했다.
  실사319/negative 사이 이미지 SHA 교집합도0이다.
- 기존 합성 heldout1985장 중 replay와 파일 SHA 중복0. 고정 salt와 SHA 순서로
  512장(484scenario)을 선택하고 실제 prepared image/원본 YOLO label SHA를 검증했다.
  기존100px 패딩/원본좌표 변환을 유지해야 하며, 새 평가 adapter는 아직 없다.
  이 검사는 파일 SHA 기준이다. 모든 근접 중복이나 새로운 독립 source test를 보장하지 않는다.
- 설치 버전은 PyTorch `2.1.1+cu118`, Ultralytics `8.4.60`.
  실제 R0를 CPU로 읽어 criterion이 `E2ELoss(PoseLoss26)`임을 확인했다.
- 실제 loss source에서 `loss * batch_size` 반환과 E2E branch 가중치를 확인했다.
  `ell=C8/8`, `L_backward=32*J`에 따라 아래 계수를 고정했다.
- BN126개에 유한 running buffer가 존재하며 affine은 학습 가능하다.
  실제 gradient/학습 후 buffer 불변 검사는 미실행이다.
- DFL은 `Identity`, `reg_max=1`, 파라미터0개다. 따라서 별도 fixed parameter
  목록이 비어 있는 것은 이 checkpoint의 실제 구조이며 DFL 해제를 뜻하지 않는다.
- CPU helper6개 테스트 통과: 노출계획, C8계수, toy gradient 항등식,
  PCK 고정분모/10px 경계/비유한·미검출 처리, 점역할 오류, RNG stream seed.
  실제 YOLO gradient/loader parity 통과로 확대하지 않는다.
- 바인딩한 기존44개 파일(참조 코드·checkpoint·manifest·paper final 포함)은
  감사 전후 SHA가 같다. 기존 실험이나 GT에 쓰기 작업을 하지 않았다.
  모든 역사적 대형 cache 전체를 다시 해시한 감사라는 뜻은 아니다.

## [설계·미실행] 군과 노출

각 seed당300update, seed1→2→3 순서로 각 행의 네 군을 실행하도록 설계 고정했다.
모든 군은 BN running statistics만 R0에 고정, affine 포함 학습 파라미터 업데이트,
stock loss/FP32/SGD/last-step300을 사용한다. 현재 구현된 것은 계약/helper/CPU 감사이며
loader와 학습 runner는 아직 구현되지 않았다.

| 군 | 실제 backward 계수(C8 기준) | 계획 실사slot | 계획 합성slot | 실제 update(seed1/2/3) |
|---|---|---:|---:|---|
| T8_FULL | 4×target_base | 2400 | 0 | 0/0/0 |
| T8_QUARTER | target_base | 2400 | 0 | 0/0/0 |
| REPLAY | target_base+source1+source2+source3 | 2400 | 7200 | 0/0/0 |
| T32_COMPUTE | target_base+target_extra1+2+3 | 9600 | 0 | 0/0/0 |

이는 nominal 계획이고 현재 실제 영상 노출은 전 군0이다. Mosaic 보조 참조량은
별도 집계해야 한다. ell은 독립 sample loss들의 엄밀한 평균이라는 주장이 아니다.

## [미실행] 성능과 비교

| 모델 | Target ALL_GT_PCK10 | Source ALL_GT_PCK10 | 2D/6D/detection |
|---|---|---|---|
| R0 | 미평가 | 미평가 | 신규 panel 미평가 |
| T8_FULL | 미학습/미평가 | 미학습/미평가 | 미실행 |
| T8_QUARTER | 미학습/미평가 | 미학습/미평가 | 미실행 |
| REPLAY | 미학습/미평가 | 미학습/미평가 | 미실행 |
| T32_COMPUTE | 미학습/미평가 | 미학습/미평가 | 미실행 |

REPLAY−T8_QUARTER, REPLAY−T8_FULL, REPLAY−T32_COMPUTE,
T8_QUARTER−T8_FULL 효과와 CI는 계산하지 않았다. 기존 AL/P/line 결과를
새군 대신 넣거나 기존319장 성능을145장 분모와 섞지 않았다.
원래 모델보다 좋아졌는지, replay가 loss축소/추가실사노출보다 유리한지는 모두 미확인이다.

## 논문 문장과 재개

현재 쓸 수 있는 내용은 **BN 통계 고정 조건에서 동일30-label replay를 비교하는
통제 실험을 설계하고 데이터/수식의 정적 사전 감사를 수행했으나 자원 오류로 학습을
시작하지 못했다**는 실행 기록이다. replay 성능 향상·원본 능력 보존·전이학습 실패,
새 알고리즘, 독립 일반화는 주장할 수 없다.

`RESUME.md`에 CPU 재감사 명령과 남은 구현/학습/평가 순서를 기록했다.
모델이나 optimizer의 재개 state가 없으므로 처음부터의 본 학습은 재실행이 아니라
아직 수행하지 않은 최초 실행이다. 사전 승인된3600update 예산은 전부 남아 있다.
GPU가 복구되어도 남은 실제-model/loader gate를 건너뛰면 안 된다.

## Git 공개 범위

이번 신규 코드와 작은 감사 결과만 main에 commit/push한다. RGB/가중치/원본 라벨을
새로 추가하지 않는다. 데이터 경로·이미지 SHA·membership·소프트웨어 source SHA는
요청한 provenance로 포함한다. 기존 capacity_screen의 무관한 untracked 세 경로는
그대로 남긴다. 최종 commit과 실제 remote 일치는 실행 응답에서 별도 검증한다.
