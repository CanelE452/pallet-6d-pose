# V217 · 후보 mask/depth와 원 query 진단 준비

원 SOURCE 12프레임의 후보 obj1 인스턴스 15개를 모두 유지해 native8별 120개 진단 행을 만듭니다. 원 입력은 고유96행/U11, SOURCE_FIT64/DEV32로 구분합니다. candidate 반복 행의 U 수는 원 U11 분모와 따로 기록합니다. 선택 인스턴스는 여전히 UNKNOWN이며 첫/가장 가까운/GT-best 후보로 고르지 않습니다.

이 코드는 실제 payload를 아직 열지 않았습니다. [정확42 PNG 바인딩](EXACT42_KNOWN_MASK_DEPTH_PNG_BINDINGS.json)은 과거 raw SHA가 있는 전체 마스크15·가시 마스크15·depth12만 포함합니다. RGB/CAD/원NPZ/가중치/원GT JSON은 열지 않습니다. 나중에 root가 실제 selected-GT COMPLETE를 바인딩하고 새 actual entry를 승인한 후에만 실행합니다. camera 정보가 없으면 metric depth는 UNKNOWN이며 raw depth·마스크 수치는 진행할 수 있습니다.

## 수치와 질문의 분리

- 후보별 dimensions/dtype, full/visible 면적·부분집합/비율, 양의 유한 raw depth 면적 및 mask 내부 지원을 기록합니다. 분모0은 UNKNOWN입니다. visible mask가 full 밖에 있는 경우도 수치를 남기며 semantic 오류나 외부 E로 단정하지 않습니다.
- 원 q의 최근접 정수 픽셀은 `floor(q+0.5)`로 고정하고 해당 픽셀과 경계에서 잘린 3×3 창을 진단합니다. q를 clamp하거나 보정하지 않습니다. **원 영상 픽셀 진단이며 RAW96 bilinear sampler/ROI support를 재현한 값이 아닙니다.**
- 원 U는 finite q·양성 mask가 있어도 U입니다. A/C 원 RAW96 ROI와 root의 기존 weak 부위 상태를 옆 열로 보존하지만 이 새 mask 수치로 관찰값을 수정하지 않습니다. ROI→원 영상 sampling 지원은 계산하지 않습니다.
- metric depth는 실제 selected camera의 유한 양수 `depth_scale`이 있을 때만 기존 BOP 단위 계약에 따라 계산합니다. 없으면 기본1 등을 넣지 않습니다.

FULL/visible/depth-valid의 mask-only 패널은 원 q/native/U 표시만 더합니다. 표시 축소는 계산/모델입력 리샘플링이 아닙니다. source RGB는 사용하지 않습니다. dimension 불일치는 해당 후보·8행을 실패/UNKNOWN으로 남기고 그림·수치용 임의 resize를 하지 않습니다. hash/decode 실패는 중단·보존하며 자동 재시도하지 않습니다.

모든 새 native D/E/SELF label과 학습 admission은 UNKNOWN/false입니다. mask 음성, query OFF, 다른 표면·물체가 q에 있다는 사실이 native E가 아닙니다. native part/instance 대응은 별도 양성 근거가 필요하지만 완벽한 mesh/corner를 blanket gate로 요구하지 않습니다. nominal pose projection·PnP·새 fit·모델·보호 점수 접근은 없습니다.

[프로토콜](PROTOCOL_BEFORE_MASK_DEPTH_PIXEL_READS.json) · [실행 false 템플릿](ROOT_ACTUAL_MASK_DEPTH_ENTRY_TEMPLATE_APPROVED_FALSE.json) · [17개 authored 검사](INVENTED_FUNCTIONAL_CHECKS.json)

향후 승인된 실행 명령:

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B diagnose_fixed12_masks_depth_queries.py --entry <새 root 승인 entry 절대경로>
```
