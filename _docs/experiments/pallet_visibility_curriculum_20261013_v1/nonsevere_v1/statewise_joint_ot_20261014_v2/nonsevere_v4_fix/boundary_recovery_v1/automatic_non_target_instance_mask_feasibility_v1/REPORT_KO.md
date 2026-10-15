# 자동 other-instance mask 공급의 기존 자산·최소 시험안

**새 detector 다운로드 없이 시작할 수 있는 경로가 있습니다.** 기존 COCO80 YOLO26n이 다른 물체의 box 후보를 만들고, 기존 MobileSAM vit_t가 그 box마다 mask 후보를 만드는 방식입니다. 현재는 정적 source/metadata 조사만 했으며 모델 로드·영상·NN·CAL/primary 행·라벨·적합·파이프라인 변경은 0입니다. [핀 원장](SOURCE_INPUT_BINDINGS.json), [체크포인트 정적 메타데이터](STATIC_CHECKPOINT_METADATA.json), [실행 전 제안 계획](PROPOSED_EXECUTION_PLAN_V1.json).

| 확인한 자산 | 실제 확인 범위 | 결론 |
|---|---|---|
| [challenge/weights/pretrained_yolo/yolo26n.pt](../../../../../../../../challenge/weights/pretrained_yolo/yolo26n.pt) 및 [root yolo26n.pt](../../../../../../../../yolo26n.pt) | 각 5,544,453B, SHA `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`로 동일. ZIP의 data.pkl을 **pickletools opcode로만** 읽어 DetectionModel/Detect, task detect, nc80, COCO80 클래스명 전체 일치 확인. | 기존 generic 검출 후보 모델. box/class/conf를 제공하며 instance mask head는 없음. 모델의 실제 로드/API 호환·정확도는 미시험. |
| [팔레트 pose release 비교 예](../../../../../../../../challenge/yolo_pose_one_model/release/pallet-pose-yolo26n-ft/pallet_yolo26n_pose_ft.pt) | 정적 metadata: PoseModel/Pose26, task pose, nc1/name item, kpt_shape9×3. SHA `ba4ef77da0eb3b0b603713b8af8210480d0b3246f13fb9a1c7ba54cfb3cc96b7`. | generic80 detector와 다른 자산. 현31 원시 KP 생성 체크포인트를 이 파일로 새로 단정하는 것은 아님. |
| 기존 MobileSAM vit_t | 기존 manifest와 opaque checkpoint SHA `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` 재일치. | 별도 object box를 주어 새 후보 mask 생성 가능. 캐시 targetmask의 세 대안을 other-object로 바꾸는 것이 아님. |
| 설치 ultralytics8.4.60 | [YOLO task_map](../../../../../../../../../../../anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics/models/yolo/model.py) 87–106, [DetectionPredictor](../../../../../../../../../../../anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics/models/yolo/detect/predict.py)는 원영상 좌표 boxes/클래스명 반환; [Results/Boxes](../../../../../../../../../../../anaconda3/envs/pallet-yolo26/lib/python3.10/site-packages/ultralytics/engine/results.py)는 xyxy/conf/cls 제공. segment API도 존재. | current pallet-only 모델을 쓰지 않고 기존 generic detector를 별도 객체로 로드하는 공식 API 소스가 이미 있음. 외부 네트워크 fallback 금지로 계획. |
| MobileSAMv2 ObjectAware 경로 | [Inference.py](../../../../../foundation_mask_v1/vendor/MobileSAM/MobileSAMv2/Inference.py) 15–16, 30–37, 81–105에 object detector→box→mask 구조가 있으나 지명된 ObjectAwareModel.pt 두 경로 모두 없음. Prompt_guided_Mask_Decoder.pt(16,310,784B)만 존재. | source만 있고 필요한 detector checkpoint가 빠져 있어 즉시 실행할 기존 조합으로 볼 수 없음. 옛 vendored ultralytics를 현재8.4.60과 전역 교체/혼합할 이유도 없음. |

전체 checkpoint의 바이트는 SHA 계산에 읽었지만 tensor 저장 member를 추출·decode하지 않았고 `torch.load`/`pickle.loads`/NN import는 사용하지 않았습니다. 재현용 [정적 검사 코드](inspect_checkpoint_metadata_v1.py)가 별도로 있습니다. generic instance-seg 이름(`yolo*-seg.pt`, MaskRCNN 등)의 checkpoint는 repo/연결 vendor 파일명 검색에서 발견하지 못했습니다. 모든 임의 이름의 best.pt를 분류했다는 주장은 아닙니다.

COCO80에는 **chair**가 있지만 **pallet/forklift**는 없습니다. 따라서 generic 검출이 의자 등의 후보를 줄 수 있어도 모든 창고 물체를 다루지는 못합니다. 특히 ‘pallet 클래스가 없으므로 검출된 모든 box가 non-target’이라는 규칙은 잘못입니다. pallet을 chair/table 등으로 오인할 수 있고, box가 실제로 팔레트와 다른 instance인지 및 앞에 있는지는 별도 불확실성입니다.

| 경로 | source상 연산 구조 | identity·범위 한계 |
|---|---|---|
| **generic detector box → MobileSAM** | fullframe YOLO1회 + MobileSAM image encoder1회 + M개 box의 prompt/mask decoder M회. M은 반환 box 수. box를 순차 처리하면 원영상1행과 mask 대안 한 묶음만 보유 가능. | COCO 밖 물체 miss/오분류 가능. SAM 품질은 의미·depth 확률 아님. mask는 `CANDIDATE_NON_TARGET_INSTANCE/UNKNOWN`으로 시작. |
| 같은 MobileSAM encoder 공유 | [predictor.py](../../../../../foundation_mask_v1/vendor/MobileSAM/mobile_sam/predictor.py) 34–91의 set_image가 feature를 만들고 217–239의 predict_torch는 `self.features`를 재사용. 동일 영상에서 target/여러 object prompt가 encoder를 공유할 수 있음. | 기존 NPZ에는 embedding이 없으므로 **과거 targetmask encoder를 공짜로 재사용하는 것은 불가**. targetmask는 그대로 두고 새 supplier가 frame별 encoder1회를 수행하는 안. 같은 모델·영상의 상관을 독립 관측이라고 주장하지 않음. |
| **AMG fullimage proposals** | [SamAutomaticMaskGenerator](../../../../../foundation_mask_v1/vendor/MobileSAM/mobile_sam/automatic_mask_generator.py) 기본32×32=1024 point prompts, batch64라 crop0에서16 decoder batches, 각 prompt3대안=필터 전3072 slots. crop별 set_image1회; 추가crop은 encoder추가. | detector 없이 unknown 물체 후보도 만들 수 있으나 target-vs-other와 foreground/depth identity가 여전히 없음. fullimage grid는 nativeS/U 좌표를 쓰지 않지만 국소 prompt 수가 훨씬 많고 실제 비용 미측정. 첫 시험안으로 우선하지 않음. |
| 별도 generic instance-seg | API 존재, 이번에 준비된 지명 checkpoint는 못 찾음. | 새 checkpoint 확보/정적 바인딩/실제 호환 검증이 추가로 필요. 지금 다운로드·실행하지 않음. |

최소 후속 시험은 **기존에 고정한 CAL3** `wood_184309:000925`, `wood_184309:000804`, `eval_cad:1778653017736058368`을 같은 규칙으로 처리하는 standalone supplier입니다. 새 detector 출력·성능을 보고 사례를 고르지 않습니다. 세 사진에 알아볼 other object가 전혀 없으면 zero/unknown으로 남기고, 양성 공급 성능을 검증했다고 결론내리지 않습니다.

1. 실행 전에 신규 supplier 소스·작은 mock tests·고정3계획·현재 영상/모델/API/pixel binding을 봉인하고 ROOT의 실제 NN/영상 GO를 받습니다. 모델은 위 고정 로컬 경로만 쓰고 누락/불일치/API 오류는 보존 후 중단합니다.
2. 원 RGB 전체를 generic detector에 넣고 모든 반환 box/class/conf·원 순서/instance ID·0검출 행을 보관합니다. 제안 설정은 현재 API 기본 `imgsz640/conf.25/iou.7/max_det300/batch1/halfFalse`를 고정한 engineering control이며 조정·실행하지 않았습니다. GT·e 자세·오류·S/U 좌표를 입력하지 않습니다.
3. MobileSAM set_image1회 후 모든 유효 box를 box-only로 순차 decode하고 모든3대안과 quality를 저장합니다. target complement/같은target 대안차집합/O 예측원은 입력으로 쓰지 않습니다. 1 worker/1 frame, 원 targetmask는 변경하지 않습니다. 시간/RSS는 최초 실제 시험에서 측정해야 합니다.
4. 각 결과는 **다른 물체 후보**이며 자동 물리O 확정이 아닙니다. instance별 원 box/class/score/모델·영상·파일·픽셀 SHA와 target 혼동·mask merge·미검출·unknown을 함께 기록하고 실제 사진에서 검토합니다. 애매한 target overlap은 non-target 양성 근거로 사용하지 않습니다. `independent_of_target_mask`는 targetmask를 입력/차집합에 쓰지 않았다는 계보이지 통계적 독립·물체 소유 증명이 아닙니다.
5. 별도 검증된 non-target 영역만 후속 사전 고정 정책으로 VIS gate 시험에 공급합니다. 이때도 ‘알려진 non-target pixels 제외’와 ‘요청된3D corner의 진짜 foreground occlusion’은 다르며, 후자는 독립적인 앞뒤 증거가 없으면 unknown입니다. 기존 optional gate가 다른 물체 경계를 제외하면서 진짜 pallet support를 보존하는지 None control과 함께 확인합니다. 이 단계까지도 pose fit·기본 채택은 하지 않습니다.

예정 pure 계약은 1encoder/Mbox 호출·frame간 feature 재사용금지, checkpoint/API mismatch/누락, 0검출의 unknown 보존, invalid box 장부, S/U poison payload 미접근, mask 대안·source/pixel binding, 기존 VIS full-footprint 제외와 targetmask/None경로 불변입니다. 현재는 공급기 구현도 실행도 하지 않았습니다. 새 sampling/선택 모집단으로 넘어가면 별도 generation/profile 및 CAL line/tangent 검증이 필요하고, visibility 장부를 최종 guard/비용에 연결하는 설계는 또 별도입니다. **현재31 소스·모델·관측·렌더 대기는 그대로입니다.**
