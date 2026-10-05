# 정적 팔레트 사람 검수 재개 안내

이 화면은 모델 이름·예측점·오차를 숨긴 채 정사각형 119장의 프레임 가림 등급과, 정적 438장의 코너 가시성을 입력한다. 직사각형 319장의 완료된 가림 등급과 기존 승인 코너 상태 71개는 잠겨 있다. CLI가 새 사람 판정을 만들거나 제출 상태로 승격하지 않는다.

저장소 루트에서 manifest를 다시 검산하려면 다음 명령을 실행한다.

```bash
PYTHONPATH=. python -m scripts.research.pallet_static_registry_review_20261003_v1.build_review \
  --source-root /home/minjae/Documents/github/pallet-pose
```

검수 화면을 시작한다.

```bash
PYTHONPATH=. python -m scripts.research.pallet_static_registry_review_20261003_v1.serve_review \
  --source-root /home/minjae/Documents/github/pallet-pose \
  --port 8767
```

브라우저 주소는 `http://127.0.0.1:8767`이다. **초안 저장**을 누르면 다음 파일에 원자적으로 저장되며, 서버를 다시 시작하면 자동으로 재개된다.

```text
data/pallet/results/pallet_static_registry_review_20261003_v1/STATIC_REVIEW_IN_PROGRESS.json
```

화면의 **JSON 내보내기**와 **JSON 가져오기**로 다른 컴퓨터의 초안을 옮길 수 있다. 가져오기 때 manifest SHA-256, case ID, 잠금 필드, 허용 상태를 다시 검사한다. `unknown` 프레임은 구체적 사유가 필요하다.

**사람 검수 제출본 확정**은 실제 검수자 이름과 정사각형 119장의 가림 등급을 모두 요구한다. 코너 가시성이 덜 끝났다면 제출본에 `PARTIAL_HUMAN_REVIEW`로 기록되며, 완료로 가장하지 않는다. 생성 파일 이름은 `STATIC_REVIEW_SUBMITTED_<hash>.json`이다.

현재 준비 상태는 다음과 같다.

- 전체 화면: 438장
- 직사각형 가림 등급: 319/319 완료·잠금
- 정사각형 가림 등급: 0/119 새 사람 검수 대기
- 기존 승인 코너 가시성: 71점 잠금
- 평가 참조가 있는 코너 가시성: 3,030점 새 사람 검수 대기
- 참조가 없는 403슬롯: 상태 입력은 선택이며 기존 2D 분모에는 들어가지 않음

테스트는 저장→종료 상태 재로딩→export→import와 제출 차단 규칙을 localhost에서 실제 왕복한다.

```bash
PYTHONPATH=. python -m unittest -v \
  scripts.research.pallet_static_registry_review_20261003_v1.test_review
```
