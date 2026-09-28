# 실행 이력 — 사후 개발과 기술 재시도를 구분

시작: 2026-09-28 01:42:02 UTC / 10:42:02 KST. 기준 main `cf52dc624fb5c83e305ff3b951f3c362b9ca1842`. 기존 dirty/untracked를 보존했다. 종료 시간·누적 계측·실제 fit 수는 RESOURCE_LEDGER와 STATE를 기준으로 한다.

## 실행 순서와 판단

1. HEAD/원격/branch/status 확인, 기존 baseline·평가·교사/학생·원고31개 입력 hash를 고정했다. host RTX3080 CUDA 작동을 확인했다. sandbox NVML 실패를 GPU 고장으로 해석하지 않았다.
2. 과거30개 시도의 실제 실행 여부·계약을 조사하고, 19개 원문/공식 자료를 열람 범위와 함께 정리했다. 현재 frozen 예측의 W/D·whole-output·per-point·조건부 검출 oracle를 계산했다. 기존 기하경로의 재현과 정확 합성/참조 입력 sanity를 수행했다.
3. Plastic native TRAIN cache를 재사용하고 Wood361장은 frozen R0/RAW/REF 추론만 보완했다. TRAIN 실사/합성 gradient 및 증강 on/off probe는 optimizer step0이었다.
4. C1: 같은 pose 후보·9점 residual·penalty에서 Huber12 점수만 적용했다. 선택 변화0, AUC 차이0으로 종료. scale sweep/새 selector fit 없음.
5. C2: raw/corrected 짝 양쪽에 실사 translate/scale OFF를 적용해 Plastic2+Wood2 fits를 수행했다. 모두 seed42/320 update/last-only. 전체 primary에서 기존 REF보다 낮아 채택하지 않았다. 사전에 고정한 seed43 재현 조건은 충족되지 않아 추가4fits를 실행하지 않았다.
6. C2의 TRAIN outlier를 기존 frozen checkpoint로 다시 추론했다. 큰 평균 오차 차이가 instance 선택 전환에 지배됨을 확인했다. GT/타깃 기준 후보로 교체하지 않았다.
7. C3: 기존 교사 TRAIN9/38 direct-click 감독을 학생이 직접 사용할 수 있는지 RAW9/MANUAL9 paired control을 별도로 명세했다. 원자료·teacher 당시 hash·173장 DEV와 ID/SHA/recording 무중복을 확인했다. 두320update fit을 완료했고 TRAIN 수동점 PCK10은32→35/38이지만 큰 Wood3점은 남았다. DEV MANUAL9−RAW9는 Plastic+5점, Wood−22점이고 두 재료 모두 기존REF보다 낮았다. 전체217/361장의 oracle student가 아니며 추가fit 없이 종료한다.

원문/소규모 진단/이전 사이클 결과를 본 뒤 다음 가설을 고르는 개발 이력이므로 전체를 사전 등록된 독립 TEST로 부르지 않는다. 각 실행 전에 해당 사이클 명세를 잠그고, prediction 생성 후 reference scoring을 분리했다.

## 기술 오류와 보존

| 단계 | 기술 문제 | 처리·영향 |
| --- | --- | --- |
| 초기 CUDA 확인 | sandbox 장치 접근/NVML 실패 | 승인된 host에서 확인·실행. 환경/드라이버 변경0, fit0 |
| pose oracle 준비 | 실행 전 문법 오류 | 실행 전에 고침; GPU/fit/update0. wall 별도 계측 없음 |
| C2 CPU preflight 첫 실행 | external-module path를 repo-relative binding에 전달해 serialization ValueError | 이미 수행된 parity 검사 뒤 실패. absolute-path/hash metadata binding으로 같은 명세 재실행; fit0/update0/GPU0, 첫 wall NA |
| 첫 그래프 실행 | whole-pose oracle scalar의 실제 schema와 다른 키 참조 | 그림 저장 전 실패, 올바른 키로 같은 plot 생성; wall .6725초, 학습0 |
| C3 GPU 시작 전 정적 검토 | `__main__` subprocess module 이름, 보고서 median/P90 키, 공개 fixture의 실제 좌표 | 명시적 module·실제 metric key·합성 test fixture로 수정. 실제 좌표는 private만 유지. 첫 manifest 보존·활성 source hash 재잠금, 변경 당시 fit0/update0 |
| C3 첫 GPU child 시작 | sandbox CUDA 접근 검사 실패 | 모델/optimizer 생성 전0step·0fit·0GPU. 실패 log 보존 후 정상 host 승격으로 같은 명세의 실제2fit 실행. 환경/드라이버 수정 없음 |

현재 설치된 Albumentations 인자 호환 경고는 기존 paired/source 런타임에 동일하게 존재한다. 이를 해결하려 환경 전체를 업데이트하지 않았다. 추정 runtime를 계측값으로 만들거나 NA를0초로 채우지 않았다. 각 실제 학습 오류/미완료 상태는 cycle의 RUN_STATE/기술 정정 기록에 남긴다.

## 커밋·원격 반영 이력

- `4bc8e41a02da884498f1e3897b8c48fc3eb563ac`: prior/literature/nativeTRAIN/pose oracle/C1 진단. push 후 HEAD=origin/main 확인.
- `6fb81b1eb23d4af717cfacff865a956ba5938f5d`: coordinate oracle·C2 실제4fits 요약·공개 비교 이미지. push 후 HEAD=origin/main 확인.
- 최종 통합 커밋은 이 파일을 포함하는 Git commit으로 식별한다. 파일 안에 자신의 commit SHA를 하드코딩하는 순환을 만들지 않는다. 최종 실제 SHA/원격 확인은 사용자에게 보고하며, 기록된 원격 확인은 CLI_REPORT의 release verification 절차와 함께 해석한다.

Git에는 관련 namespace만 추가하고 큰 checkpoint/원본RGB/좌표·카메라 배열은 올리지 않는다. 기존 main 원고·표·GT·완료 결과를 덮어쓰지 않는다.
