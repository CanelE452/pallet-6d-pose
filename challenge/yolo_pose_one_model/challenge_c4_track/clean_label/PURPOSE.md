# PURPOSE — 정본 라벨 + flip 제외 C4 재학습

[소비처] 과제 트랙 배포 모델 교체 결정 — HuggingFace `CanelE452/pallet-pose-yolo26n-c4`
를 갱신할지 말지. 판정과 근거는 `_docs/notes/c4-rotation-symmetry.md` §1.

[문장] 라벨 소스를 규약 위반 23.3% 인 `projected_cuboid` 에서 위반 0% 인
`keypoint_annotations` 로 바꾸고 좌우 flip 증강을 빼면, C4 대칭 학습의 코너 위치추정
정확도가 F2 대비 개선된다.

가설·방법·사전등록 판정 기준·중단 기준은 위 notes 문서에 있다. 여기 복사하지 않는다.
