# V44 두 검증과 종료 판정

새 장면에서 고정 V41의 작은 효과를 검증하고, 부모가 보지 못한 새 장면의 실제 예측으로 보정기를 학습한다. 같은 후보에 문턱·가중치·선택기를 반복하는 실험은 하지 않는다. 등록 기준 실패로 현재 고정 특징·선 후보 보정 연구 계열을 종료했다.

## 데이터와 모델

같은 CAD 2개 × 새 그룹12개 × 구도5개 × 가림판2상태=240 RGB, 가림/잘림 파생600관측. 난수440000+asset100+group, v44_ ID. 기존 난수32000계열과 원본/그룹 중복0. 12새fit/4별도확인/8최종check 그룹. 검출210/300,85/100,137/200,최종1096점. 미검출 제외를 명시하며 검출 복구는 평가하지 않는다.

부모 V12/V25/V34/V36/V41을 고정했다. 새 학습 영상까지 부모에게 미관측이므로 학습 밖 예측 감독이지만 동일 원본에서 부모를 교차 재학습한 OOF는 아니다. 기존/새 원본 비교는 장면 분포 효과도 포함한다. 같은 CAD·같은 렌더러 계열의 새 장면이며 새 CAD/실사 일반화는 아니다.

기존fit12/새fit12 × 점 특징/점+선 특징 ×3시드 ×x/y회귀2개 =24 HGB 절대오차 회귀, 각60라운드=1440라운드. 감사 재학습4개 별도. 신경망 업데이트0. 이동은 V34 기준점에서 bbox 대각선15% 이내. 점 특징47차원은 base/strong/V34좌표,query32,aspect,pointID다. 선 대조는 동일 점 입력에 물리3역할 특징 평균/표준편차/관측여부를 추가한다. 공통 V34 기준점과 query는 선을 학습한 부모에서 오므로 순수YOLO 점전용 아키텍처 대조는 아니다. 새cal/check로 학습·기준 선택하지 않았다.

## 실행

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_fresh_decision_v44/run44.py --phase all
```

단계: render, prepare, parity, infer, replay, train, evaluate, audit, report, deliver. all은 완료 마커를 건너뛴다. 개별 단계는 결과를 덮어쓰므로 완료 후 변경은 새 버전으로 진행한다. run44는 등록된 종료 판정을 번복하거나 후속 실험을 생성하지 않는다.

렌더는 pallet-pose 환경(USD/pxr·pyrender), 추론/CPU학습은 pallet-yolo26 환경이다. render/prepare/infer/replay는 GPU 접근 권한, deliver는 Chrome/X11 접근 권한이 필요하다. Discord는 전체 검증 후 별도로 승인된 완료 사실만 전송한다.

실행 당시 NVIDIA 커널580.173.02/설치라이브러리580.178.04 불일치로 CUDA804가 났다. NVIDIA 공식 파일 https://us.download.nvidia.com/XFree86/Linux-x86_64/580.173.02/NVIDIA-Linux-x86_64-580.173.02.run 을 /tmp에 추출하고 필요한 libcuda/libnvidia/EGL_nvidia 라이브러리만 /tmp/pallet-v44-cuda-libs에 연결했다. 시스템 설치·교체·재부팅 없음. 드라이버가 정상인 환경에서는 우회가 필요 없다. `PALLET_V44_DRIVER_LIBS`로 실험별 경로를 지정할 수 있다. /tmp 라이브러리는 영구 배포 의존성이 아니며 실행 환경 기록은 ENVIRONMENT.json에 있다.

## 검사

FEATURE_PARITY: 기존436관측의 특징·V41 점 출력 세시드 최대차이0. UPSTREAM_REPLAY: 기존8프레임/시드에서 고정신경망 forward를 현재GPU로 재현, 점 최대차이약3e-5px. TARGET_AUDIT: 독립카메라투영 최대약1.7e-13px. AUDIT: 새12/4/8그룹분리/이미지중복0,24모델재현,24점배열조건별재계산,비학습영역GT/입력교란재학습4개. 추론 특징 함수에 GT묶음을 전달하지 않는다. 원본·가림/잘림 사례와 최종그림을 직접 확인했다.

## 결과 및 종료

새평가 V34 9.058px,고정V41 9.025px. 작은3시드 평균효과는재현,매시드1%/모든조건보존은실패. 새점특징8.767px,새점+선8.674px. 선의순수추가이득 .43/.96/1.79%,paired95%구간[-.1798,+.0026]px로0포함. seed3crop20쉬운135점3.011→4.018px,crop40쉬운105점3.087→4.152px. 평균개선과조건안정성은별개다.

판정 `CLOSE_CURRENT_FROZEN_FEATURE_CORRECTION_FAMILY`. 이번계열의 추가gate·문턱·비중 실험은 종료한다. 모든DHT융합의원천적불가능을의미하지않는다. 실제운영모델교체없음.
