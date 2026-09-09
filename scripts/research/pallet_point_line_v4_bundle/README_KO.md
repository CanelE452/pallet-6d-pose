# 팔레트 점·선 증거 실험 패키지 v4

**CLI에 전달할 본문:** `CLI_INSTRUCTIONS_KO.md`.

실험 목적은 source-only 팔레트 예측에서 선 증거의 증분 효과와 same-ID baseline reference의 영향을 분리하는 것이다. P/S/H/HA 네 군의 실제 모델, 학습/추론/평가 유틸리티, 테스트를 포함한다. 실제 repository data adapter와 전체 네트워크 online 학습 연결은 CLI에서 해야 한다.

## 중요한 정정

원래 structured-v2 코드에는 이미 끝점 패치·유한 선분 내부 읽기·전체 배치 scoring·후보별 정답 오차 감독이 있다. 따라서 이 기능들을 새 아이디어로 포장하지 않는다. 이번에는 모든 군이 P3/P4 읽기를 공유하고, **선분 내부 추가 / 명시 DHT 추가 / 같은-ID reference 유무**를 분리한다. `SOURCE_REVIEW.md`에 근거를 적었다.

## 구성

- `pointline_v4/`: 직접 작성한 실행 가능한 core. 기존 저장소의 모듈을 내부에서 몰래 호출하거나 다운로드하지 않는다.
- `tests/`: 단위 및 실제 4-step trainer/checkpoint/GT 없는 scoring 통합 테스트.
- `PROTOCOL_TEMPLATE.json`: 실행 전 실제 SHA와 경로를 바인딩해야 하는 템플릿. 기본 `locked=false`로 오실행을 막는다.
- `evidence/`: 이 환경의 실제 검증 기록. 팔레트 정확도 결과가 아니다.
- `CLI_INSTRUCTIONS_KO.md`: data adapter, 비교군, gate, 캐시 및 조건부 전체 학습, 평가·실패 처리 지시.

## 로컬 소프트웨어 테스트

```bash
cd pallet_point_line_v4_bundle
python -m pytest tests -q
```

Python 3.10 이상 문법과 PyTorch/NumPy/pytest를 사용한다. 실제 검사 환경/버전은 `evidence/LOCAL_VALIDATION.json`에 기록한다. 설치된 PyTorch를 자동 교체하는 설치 스크립트는 제공하지 않는다.

## 이 패키지가 하지 않은 것

실제 팔레트 학습·실사 추론·GT 감사·canonical scorer 재현·CUDA 검증·전체 네트워크 학습은 수행하지 않았다. CPU 생성 fixture의 PASS는 소프트웨어 계약 확인이지 연구 가설이나 정확도 향상의 증거가 아니다. 제공 code도 현장 adapter 및 독립 검증이 필요하다.

P/S는 같은 joint backbone과 후보를 사용하므로 완전한 no-Hough 모델이 아니다. 네 군의 등록 파라미터 수가 같아도 explicit Hough cue를 빼면 활성 입력 열은 달라진다. 이 한계를 숨기지 않는다.

## 실제 실행 범위

작은 모델 4군×3seed를 먼저 학습한다. 사전 기준을 넘지 못하면 거기서 종료한다. 합성 및 실사 DEV 기준을 모두 만족한 경우에만 지시문의 제한된 전체 네트워크 비교를 진행한다. 독립 FINAL과 실사 학습은 제외한다.
