# Pallet keypoint visibility

현재 목표: **초기 추론·예상 기하 구조에서 만든 키포인트 중심 crop으로 대상 팔레트가 있는지/가려졌는지 판별한다. 실사 키포인트 가시성 주석은 평가에만 사용한다. 자세 개선은 현재 목표가 아니다.**

학습·설정 선택은 합성만 허용한다. 실사 이미지/파생 특징/레이블로 모델·normalizer·threshold·checkpoint를 적합하지 않는다. 범용 frozen 사전학습은 허용한다.

먼저 [HANDOFF_KO.md](HANDOFF_KO.md)를 읽고 [PATH_INDEX.csv](PATH_INDEX.csv)의 기존 결과와 annotation을 확인한다. 원본 데이터·모델은 복사하지 않았고 이 폴더에서 새 학습/추론은 시작하지 않았다. 원 OCPR·v60·v61·원고·기존 dirty checkout은 그대로다.

## 첫 대화에 붙여 넣을 지시문

```text
HANDOFF_KO.md와 PATH_INDEX.csv를 먼저 읽어 줘. 지금 목표는 자세 개선이 아니라 초기 예측 키포인트 중심 crop에서 대상 팔레트가 있는지/가려졌는지 판단하는 거야. 학습은 합성만 하고 실사 키포인트 visibility annotation은 평가에만 사용해.

v61의 query-pixel presence와 semantic keypoint visibility가 다르다는 점을 유지하면서, 이미 있는 실사 annotation의 이미지 hash·코너 순서·고정 native mapping·missing/unknown부터 확인해 줘. v60/v61/기존 real-CAL 모델의 동결 저장 출력을 같은 query와 같은 semantic annotation 분모에서 비교해 줘. GT로 crop/threshold/모델을 고르지 말고 초기 좌표 오차를 별도로 집계해. mapping 미확정 항목을 임의로 identity/GT-best로 채우지 마. 새 학습이나 새로운 대형 실험은 우선 하지 말고, 확인된 결과와 정확한 남은 차단을 먼저 설명해 줘.
```

## 현재 Ubuntu Codex에서 이어가기

이 폴더는 **로컬 파일 프로젝트 준비 완료**, Codex 앱의 새 프로젝트 등록은 **미완료**다. 현재 도구의 native app inventory가 비어 있고 로컬 프로젝트 등록 도구도 노출되지 않아 앱을 클릭하여 등록할 수 없었다. 사용자 앱에서 프로젝트/폴더 열기 기능으로 다음 폴더를 선택하면 된다:

`/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility`

사용자는 현재 Ubuntu Codex 화면의 **GPT-6.1 Sol Ultra**와 직접 이어서 대화하려는 것이다. 모델 전환이나 ChatGPT Pro 연결이 필요한 요청이 아니다. 위 폴더를 선택하고 START_HERE_KO.md를 읽도록 요청하면 된다. 이 작업에서 앱 등록·모델 전환을 했다고 주장하지 않는다.

이전 설계의 참고자료로만 남기는 대화:
[기존 Pro 대화](https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a)

해당 대화로 이동하거나 새 prompt를 보낼 필요가 없다. Ubuntu Codex는 HANDOFF의 전달된 요약과 링크를 참고하면 된다. 이 작업에서 브라우저 메시지 전송·API 키 생성·외부 업로드는 하지 않았다.
