# 공개 SOURCE GT 보유·활용 준비도

기존 MR6D SOURCE GT가 없다는 전제는 철회합니다. V153의 보존 영수증은 원 RGB 230개, depth 230개, 전체 마스크 303개, 가시 마스크 303개, 대상 모델 1개(총 1,067개)를 기록합니다. `scene_gt` 바인딩이 있고, 당시 intake 코드는 `scene_gt_info`·`scene_camera`와 GT 목록 인덱스별 마스크를 사용했습니다. 이번에는 그 실제 값·파일·이미지를 다시 열거나 stat/hash하지 않았습니다. 현재 보유 상태는 **과거 보존 영수증 근거**이며 현재 payload 재검증과 구분합니다.

입력 전용 manifest의 `GT_pose_mask_depth_or_label_fields_copied=0`은 GT를 배포 입력에서 제외한 계약입니다. GT가 없다는 뜻이 아닙니다. [MR6D 공식 배포 페이지](https://huggingface.co/datasets/anas-gouda/mr6d)와 고정 commit의 기존 intake가 출처입니다.

## 이미 수행된 작업

| 경로 | 기존 활용 | 권위의 한계 |
|---|---|---|
| [V154 감독 seal](../../query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_SUPERVISION_SEAL.json) | 실제 query의 대상/역할 대응과 기존 GT-instance·부위 제안을 연결 | assistant SOURCE weak이며 point-quality·물리 점 gold가 아님. 잘못된 물체를 E로 바꾸지 않음 |
| [V159 부위 정의](../../query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json) | nominal native↔부위 관례 및 전체/가시 마스크 지원 진단 | 공식 native8 keypoint 의미 GT가 아님. 마스크가 의도 부위의 가시성을 단독 증명하지 않음 |
| [V159 owner bridge](../../query_visibility_native_part_support_feasibility_20261024_v159/owner_supervision_bridge/COMPLETE.json) | query 픽셀의 TARGET/FOREIGN/BG 보조 감독 | native D/E·좌표 품질과 별개 |

따라서 다음 작업을 ‘첫 GT 활용’으로 부르면 부정확합니다. 기존 교차 대응·부위/owner 진단의 한계를 이어서 점검해야 합니다.

## 무엇이 아직 필요하나

MR6D의 obj1 Euro 목재 pallet과 프로젝트의 nominal native 관례를 그대로 공식 native8 정답으로 보지 않습니다. 저장 순열의 방향은 `dynamic[k]=fixed[perm[k]]`이며 한 번만 적용해야 합니다. 대상 인스턴스는 scene/frame/GT 목록 인덱스/obj_id와 해당 마스크로 연결해야 합니다. 같은 obj_id의 다른 pallet도 외부 물체일 수 있습니다. 원 입력 producer에는 GT가 인증한 선택 인스턴스 대응이 없습니다.

전체 마스크는 실루엣, 가시 마스크는 보이는 표면의 근거입니다. q의 마스크 양성은 다른 표면일 수 있고, 음성은 가림 외에도 틈·경계·pose/치수 오차·OOB·누락일 수 있습니다. 음성을 곧바로 E로 바꾸지 않습니다. [저자 논문](https://arxiv.org/html/2508.13775v1)의 pose/실물 치수 불확실성도 남습니다. 완벽한 mesh/정확한 3D corner 인증을 모든 weak 감독의 선결 조건으로 요구하지는 않습니다. 설명 가능한 동일 인스턴스·동일 부위 근거와 불확실성을 별도로 보존합니다.

**다음 한 가지 점검**은 기존 고정 SOURCE TRAIN 프레임에서 원 q/L/native/U를 유지한 채, 기존 GT-instance·nominal 부위 교차 대응과 depth/전체·가시 마스크 계약을 대조하는 것입니다. 먼저 기존 intake 영수증에서 해당 TRAIN의 정확한 GT/info/camera/마스크/depth whitelist를 만들고, 별도 실제 접근 범위가 승인된 뒤에만 파일을 확인합니다. held 000001이나 새 nonval 다운로드로 범위를 늘리지 않습니다. PnP·GT 최근접 순열·q 보정·재중심화는 하지 않습니다. 부위 대응, 가시 표면 지원, query-local support, native visibility 후보를 다른 열로 기록하고 모호함은 UNKNOWN으로 남깁니다. GT는 향후 SOURCE 타깃 근거일 수 있으며 배포 head의 입력으로 넣지 않습니다.

## 데이터 출처 구분

MR6D obj1은 Euro 목재 pallet이고 이 SOURCE 부분집합은 실제 정사각 pallet 범위를 제공하지 않습니다. Pallox pallet-bin 자료는 별도 대상입니다. [KU Leuven 공식 출판 메타데이터](https://iiw.kuleuven.be/onderzoek/acro/personeelsmap/publicaties/view?fromnr=1&pubsonpage=20&pubtype=&sortby=scdate)는 RGB-D·분할 마스크·물체 pose가 있는 replication data를 설명하지만, 이번 20개 파일 범위에서 Pallox 로컬 보존 여부는 **UNKNOWN**입니다. YCB-V는 기존 방법의 외부 가림 cutout donor이며 pallet/native8 정답이 아닙니다. 서로 객체 ID나 native 관례를 합치지 않습니다.

## 이번 확인 범위

기존 로컬 text 20개와 공식 primary 웹 text만 확인했습니다. 실제 GT/camera 값, RGB/depth/mask/CAD/NPZ payload의 stat·hash·read·decode, 모델·forward·학습·점수·새 annotation은 모두 0입니다. pinned HF tree의 웹 text 확인은 실패하여 현재 원격 개별 파일 존재를 새로 인증하지 않았습니다. V216 A 관찰은 이미 GT를 보지 않고 봉인되었으며 이 후속 문서로 관찰값·합의·학습 admission을 변경하지 않습니다. 현재 teacher 적격성 및 training admission은 false입니다.

정확한 20개 증거 바인딩·과거 영수증과 현재 미확인 사항은 [REVIEW.json](REVIEW.json)에 있습니다.
