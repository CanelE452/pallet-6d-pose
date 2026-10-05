# 남은 사람 행동과 실제 재개 명령

모델 추론과 정적 319장 재집계는 끝났다. 남은 작업은 사람 판정이 필요한 세 갈래이며 서로 다른 로컬 화면과 저장 파일을 사용한다. CLI는 빈 응답을 사람 완료로 만들지 않는다.

## 1. 리프터 직접 가시 코너 검수: 포트 8765

저장소 루트에서 다음 명령을 실행하고 <http://127.0.0.1:8765>를 연다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10   scripts/research/pallet_lifter_case_review_20261003_v1/start_review.py
```

고정 120장과 반복 24장을 실제 물리 코너 정의로 검수한다. 직접 보이는 코너만 클릭하고 숨은 가상 코너를 추정하지 않는다. 로컬 저장은 `data/pallet/results/pallet_lifter_case_review_20261003_v1/review/annotations_in_progress.json`이며, 화면에서 `LIFTER_REFERENCE_REVIEWED.json`을 export한다. 다른 컴퓨터의 export가 이미 있으면 새 클릭을 반복하지 말고 다음 명령으로 검증·재개한다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10   scripts/research/pallet_lifter_case_review_20261003_v1/resume.py   /path/to/LIFTER_REFERENCE_REVIEWED.json
```

## 2. 리프터 객체 대응: 포트 8766

1번 export를 만든 뒤 queue를 만들고 <http://127.0.0.1:8766>을 연다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10   scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py prepare-object-review   --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl   --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json   --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json   --output data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json

/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10   scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/object_match_review.py serve   --queue data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json   --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json   --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl   --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json   --store data/pallet/results/pallet_lifter_case_review_20261003_v1/review/object_match_in_progress.json   --port 8766
```

각 프레임에서 frozen 선택 객체가 사람 참조 대상과 `same / different / undetermined`인지 판정한다. 모델 이름·코너 오차는 화면에 나오지 않는다. export한 `LIFTER_OBJECT_MATCH_REVIEWED.json`은 `combined_integration/README_KO.md`의 import/apply 명령으로 evaluator 파생본에 연결한다.

## 3. 정적 정사각형 가림과 코너 가시성: 포트 8767

저장소 루트에서 manifest를 검산한 뒤 <http://127.0.0.1:8767>을 연다.

```bash
PYTHONPATH=. /home/minjae/anaconda3/envs/lifter/bin/python   -m scripts.research.pallet_static_registry_review_20261003_v1.build_review   --source-root /home/minjae/Documents/github/pallet-pose

PYTHONPATH=. /home/minjae/anaconda3/envs/lifter/bin/python   -m scripts.research.pallet_static_registry_review_20261003_v1.serve_review   --source-root /home/minjae/Documents/github/pallet-pose --port 8767
```

초안은 `data/pallet/results/pallet_static_registry_review_20261003_v1/STATIC_REVIEW_IN_PROGRESS.json`에 저장된다. 319장 직사각형 가림 등급과 기존 71개 코너 상태는 잠겨 있다. 정사각형 119장 가림 등급은 제출 전에 모두 필요하지만, 3,030개 참조 코너 가시성은 부분 완료 상태로도 별도 기록할 수 있다. JSON export/import와 서버 재시작 재개가 검증되어 있다.

실제 사람 export가 생기기 전에는 가시 코너 정확도·정지 잡음·물리 정확도의 `x`를 숫자로 바꾸지 않는다.
