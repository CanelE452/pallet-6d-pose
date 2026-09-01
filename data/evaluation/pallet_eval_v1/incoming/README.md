# Incoming real captures

여기는 아직 positive/negative, plastic/wood membership이 확정되지 않은 연속 촬영본을
보존하는 staging 영역이다. `incoming` frame은 논문 평가 모집단이 아니며
`manifests/frames.csv`와 combined evaluation target에 검수 전 자동 합산하지 않는다.

각 `sessions/<session>/`에는 다음이 들어간다.

- `rgb/`: source ZIP에서 CRC 검증과 동시에 독립 추출한 원본 frame
- `camera_info.json`: archive 안의 원본 카메라 metadata
- `cam_K.txt`: 같은 metadata의 3×3 `K`를 annotation tool 형식으로 옮긴 파일
- `session.json`: archive SHA256, 촬영 lighting, resolution, import/검수 상태
- `manifests/frames.csv`: frame별 SHA256과 기존 평가 데이터 중복 감사
- `manifests/frame_review.csv`: 모든 raw frame의 픽셀 검수 결과
- `manifests/frame_review_plan.json`: 재질 경계와 제외 구간의 재현 가능한 검수 계획

`SESSION` 목록에서는 DAY와 NIGHT capture가 각각 `PLASTIC`, `WOOD` 두 개의
zero-copy `STAGING EDIT` 행으로 보인다. 각 행은 같은 raw capture를 복사하지 않고
참조하되 서로 겹치지 않는 실제 frame subset만 표시하며, registry의 object별
geometry로 PnP를 푼다. raw capture 파일은 수정·이동하지 않는다.

분류 정본은 모든 raw frame을 정확히 한 번 기록한 `frame_review.csv`다. 실제
픽셀을 프레임 단위로 검수해 `plastic`, `wood`, `exclude`로 나눴으며,
`exclude`(파렛트 없음, 카메라 이동, 심한 motion blur)는 두 객체 view에서 모두
숨긴다. 최종 수는 DAY `PLASTIC 17,917 / WOOD 9,362 / EXCLUDE 1,749`,
NIGHT `PLASTIC 7,913 / WOOD 4,546 / EXCLUDE 1,124`이다. 정확한 재질 경계와
제외 구간은 capture별 `frame_review_plan.json`에 남겨 둔다.
partition view 안의 `frame N/M`과 goto는 view-local 번호다. 패널에는 원본 기준
`source ordinal N/raw_total`도 함께 표시하며, 경계 추적은 source ordinal 또는
filename을 사용한다.

객체별 출력은 다음처럼 분리한다.

```text
incoming/annotations/<capture>__plastic/
incoming/annotations/<capture>__wood/
```

PnP GT JSON, 호환 PNG, `frame_tags.csv`, `_overlays/<stem>.png`는 해당 출력 아래에만
생성된다. 제공된 camera intrinsics는 아직 검증되지 않았으므로 GT에는
`intrinsics_quality=UNKNOWN`으로 기록한다. raw의 `PROVIDED_UNVERIFIED` 품질과
`camera_info.json` 출처는 `intrinsics_source`에 그대로 남긴다.

staging save는 top-level 평가 manifest나 progress/report MD를 갱신하지 않으며
evaluation member를 자동 생성하지 않는다. 검수한 frame의 DEV/FINAL promotion과
평가 활성화는 별도 절차다. 동일 SHA frame은 active evaluation에 중복 승격하지 않는다.
