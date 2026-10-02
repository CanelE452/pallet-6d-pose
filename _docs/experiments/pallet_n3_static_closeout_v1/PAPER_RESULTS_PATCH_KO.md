# 국문 v3 결과 반영

원본은 보존하고 manuscript_ko_v3_static_closeout.md에 실제 표와 관련 본문을 반영했다. matching v3 LaTeX 원본은 없어 generated_tables의 동일 label별 .tex는 교체 준비 상태이다. 다른 self-training/P-only 원고는 수정하지 않았다.

- `tab:composition`: 기존 표 교체
- `tab:corners`: 기존 표 교체
- `tab:pose`: 기존 표 교체
- `tab:ablation_results`: 기존 표 교체
- `tab:ablation_deltas`: 보조표 추가
- `tab:type_results`: 기존 표 교체
- `tab:square_manual_declared`: 보조표 추가
- `tab:square_manual_in_frame`: 보조표 추가
- `tab:square_contrasts`: 보조표 추가
- `tab:occlusion_results`: 기존 표 교체
- `tab:visibility`: 기존 표 교체
- `tab:tail`: 기존 표 교체
- `tab:cost`: 기존 표 교체
- `tab:comparators`: 기존 표 교체
- `tab:backbones`: 기존 표 교체
- `tab:update_alternative`: 기존 표 교체
- `tab:pose_uncertainty`: 보조표 추가
- `tab:pose_common_success`: 보조표 추가
- `tab:ablation_uncertainty`: 보조표 추가
- `tab:corner_uncertainty`: 보조표 추가
- `tab:backbone_occlusion`: 보조표 추가

ResNet 실제10epoch/DSNT 명세, 공통128장 학생 표, 두 정사각형 모드, 직접 검수71점 상태와 참조 좌표 차이, 중앙값/P90 및 CI의 다른 해석을 본문에 반영했다. 리프터 장은 원본과 byte 단위로 같으며 이번 검증 대상이 아니다. 원래319장 학생 칸은 BLOCKED_CONTRACT; 정사각형T/R은 x; 일반화CI는 NA이다.
