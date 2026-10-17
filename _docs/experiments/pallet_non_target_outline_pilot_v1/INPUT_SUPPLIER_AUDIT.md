# 비대상 입력 공급기 감사

2026-10-17 현재 `ASSISTED` 공급기는 기존 MobileSAM과 별도로 동결한 RGB 점/박스 입력을 사용한다. AUTO는 `AUTO_INPUT_SUPPLIER_NOT_AVAILABLE`이며 실행하지 않는다. source-only 확인 뒤 동결된 3개 admissible prompt에서 실제 MobileSAM 추가 추론을 수행했다. 6행 입력 장부와 9개 대안 마스크를 보존했다. 고정 마스크를 평가/채택하는 절차는 별도 검토 gate다.

## 기존 자료와 실제 바이트 확인

다음 두 기존 감사 기록을 먼저 읽었다. 이전 공급기 제안은 `PROPOSAL_ONLY_NO_ACTUAL_GO_NO_PIPELINE_CONNECTION`, `implementation_created=false`, `actual_steps_executed=0`이었다.

- `.../boundary_recovery_v1/occluder_mask_supply_source_audit_v1/AUDIT_V1.json`
- `.../boundary_recovery_v1/automatic_non_target_instance_mask_feasibility_v1/PROPOSED_EXECUTION_PLAN_V1.json`

실제 vendor는 원 저장소의 `_docs/experiments/pallet_visibility_curriculum_20261013_v1/foundation_mask_v1/vendor/MobileSAM`이다. git HEAD는 `f706ad9c4eb7f219c00d9050e46328518ffb65d2`이며 vendor 작업 트리는 깨끗했다. `MOBILESAM_ASSET_MANIFEST.json`의 18개 파일을 모두 새로 SHA256 계산했고 모두 일치했다. 이 중 코드/라이선스 17개와 체크포인트 1개를 별도 런타임 설정에 고정한다.

| 항목 | 실제 확인값 |
|---|---|
| checkpoint | `weights/mobile_sam.pt`, 40,728,226 bytes |
| checkpoint SHA256 | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` |
| predictor SHA256 | `6dafa1d20430f6c4e0d37c53bfa80a8241619abb4264d26ca30d843991b760ff` |
| build_sam SHA256 | `bc195e97cdb20286b7312421f790cbb943659ca4bf5babb60dbebc6400cf32ed` |
| license | vendor `LICENSE`: Apache License 2.0 |
| LICENSE SHA256 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| model | 기존 registry의 `vit_t`, 고정 checkpoint |
| 사용 가능한 기존 Python | `PYTHON_ENV/bin/python` |
| installed runtime | Python 3.10, torch `2.1.1+cu118`, torchvision `0.16.1+cu118`, timm `0.9.16` |

기본 `CONDA_ROOT/bin/python`에는 timm/torchvision이 없어 사용하지 않는다. 기존 pallet-pose 환경에 vendor 경로를 추가한 import 및 `inspect.signature` 확인은 통과했다. 이 확인에서는 모델 생성/가중치 로드/NN forward를 하지 않았다. 저장소 Apache-2.0 라이선스를 확인했으며 별도의 데이터 권리나 현재 영상에서의 분할 정확도를 인증하는 의미는 아니다.

## 실제 API와 고정 규칙

사용한 로컬 원문은 vendor `mobile_sam/predictor.py`, `build_sam.py`, `modeling/sam.py`다.

1. `sam_model_registry['vit_t'](checkpoint=실제절대경로)`와 `SamPredictor(model)`을 사용한다. 새 설치, 모델 다운로드, torch hub 호출은 없다.
2. `set_image(rgb_uint8_HWC, image_format='RGB')`는 vendor ResizeLongestSide 변환 후 image encoder를 한 번 호출한다. 인정 가능한 입력이 있는 프레임마다 최대 한 번이다.
3. 같은 embedding에 `predict(point_coords=[[x,y]], point_labels=[1], box=[x0,y0,x1,y1], mask_input=None, multimask_output=True, return_logits=False)`를 한 번 호출한다. 원이미지 XY 좌표이며 박스와 점을 한 호출에 함께 준다.
4. 반환값은 원 canvas의 `bool[3,H,W]`, quality `[3]`, low-resolution logits `[3,256,256]`다. vendor 기준 원해상도 logits `> model.mask_threshold`이며 threshold는 `0.0`임을 실행 시 검사한다. 추가 threshold/팽창/연결성 필터는 없다.
5. 최대 quality의 원 인덱스를 선택하고 정확 동률은 첫 인덱스를 사용한다. quality는 인증된 신뢰확률이나 객체 소유권/깊이 인증이 아니다.
6. 세 대안, 원 인덱스, quality, low-resolution logits, 선택 인덱스와 마스크를 모두 새 private 출력에 보존한다. 원 canvas, 파일 SHA, raw binary pixel SHA, encoder/decode 시간 및 실제 forward hook 계수를 기록한다.

pixel SHA는 이진 마스크를 row-major 연속 `uint8` 0/1 배열로 변환한 원바이트의 SHA256이다. PNG는 0/255 표시 인코딩이며 그 파일 SHA는 별도로 기록한다. 향후 부모 membership에는 원좌표의 `floor(x+0.5), floor(y+0.5)` 고정 최근접 규약만 사용한다. 이미지 범위 밖의 좌표를 영상 안으로 clamp하지 않는다.

## 입력 경계와 검토 gate

신규 `supplier.py`는 별도 프로세스에서 실행한다. API 입력은 strict whitelist의 neutral RGB manifest, 동결한 점/박스 PROMPT manifest, 모델/실행환경 설정뿐이다. 각 manifest의 SHA를 CLI 인자로 고정하며 알 수 없는 필드는 거부한다. 후보/비용/GT/참조/원 target mask/`SUPPORT_DECISIONS`를 인자로 받거나 pose pipeline 모듈을 import하지 않는다. 원 RGB의 파일 SHA, 디코딩 후 RGB pixel SHA와 canvas가 prompt binding에 일치해야 한다.

Python audit hook은 정확히 허용한 RGB 파일, 두 manifest, 모델 설정, 고정 vendor 런타임 및 checkpoint, 설치환경 코드/필수 시스템 런타임 읽기, 새 출력만 허용한다. 저장소 전체와 실험 private root 전체는 읽기 트리로 허용하지 않는다. 네트워크 연결/이름 조회와 child process 실행도 거부한다. GT/target-mask/candidate/기존 support 감사 이름의 실제 open 시도 sentinel을 시작 시 거부하고 그 결과를 보존한다. 이는 협조적인 Python 프로세스의 경계이며 모든 native C/CUDA 파일 접근을 보안 sandbox로 가로채는 것은 아니다. 이 한계는 실제 접근 장부에도 명시한다.

mask 출력은 `PREDICTED_PENDING_FIXED_MASK_REVIEW`, `admitted_for_rerank=false`로 기록된다. 원 RGB와 **최대 quality로 이미 고정된 마스크**의 명백한 다른 객체 혼합을 새 자세 점수 열기 전에 한 번 검토해야 한다. 다른 SAM 대안을 순회하거나 새 결과를 보고 입력을 바꾸지 않는다. model이 검토하면 `ASSISTED_REVIEWED`와 `reviewer_kind=model`을 기록하며 인간 검수로 부르지 않는다. 검토 실패/미확정은 원 경로를 유지한다. 콘이 명확하지 않은 프레임은 `NO_ADMISSIBLE_NON_TARGET_EVIDENCE`이며 이는 가림 없음 또는 background 인증이 아니다.

동결 prompt manifest SHA256는 `36a8072c14897acfd29b04e954fb1f33366b2feb27924ffbffe4a6b36b26b030`이다. 프롬프트 제작자는 별도 모델이며, 이전에 선정된 6개 사례를 대상으로 원 RGB만 한 번씩 보았다. 사후 선택된 사례이므로 독립 블라인드 시험이 아니다. admissible 3개 입력과 no-admissible 3개 입력을 모두 원분모에 남긴다.

## AUTO를 실행하지 않는 이유

기존 source audit를 재사용한 뒤 실제 바이트와 정적 pickle opcode metadata를 다시 확인했다. 기존 `yolo26n.pt`와 `challenge/weights/pretrained_yolo/yolo26n.pt`는 각각 5,544,453 bytes, 동일 SHA256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`이며 task `detect`, class 80개다. 두 체크포인트의 class 이름은 설치된 Ultralytics COCO80 목록과 일치한다. 이 검사는 unpickle/tensor decode/model 생성/NN 실행 없이 이뤄졌다. COCO80에는 `traffic cone`이 없다. pallet이 목록에 없다는 이유로 검출 결과 전부를 비대상 물체로 분류하지 않는다.

vendor MobileSAMv2의 두 기존 ObjectAwareModel 지정 경로에는 가중치가 없다. prompt decoder 파일은 일반 detector checkpoint가 아니다. 기존 조사 범위 및 이번의 repo·연결 vendor·HuggingFace/torch cache·Downloads의 모델명 파일 검색에서 cone를 지원하는 GroundingDINO/OWL/YoloWorld/YoloE의 실제 API+가중치+라이선스 묶음은 확인되지 않았다. 이는 디스크 모든 opaque checkpoint가 없다는 증명은 아니다. 의미 식별기가 없는 SAM의 자동 제안만으로 cone AUTO를 만들지 않는다. COCO80 detector로 콘 미검출만 나열하는 추론도 실행하지 않는다.

따라서 AUTO238은 미실행이다. 새 모델 설치/다운로드/외부 API/RGB 전송은 0이다. 별도 자동 공급기 도입은 실험 결과 보고 후 하나의 후속 변경 요청으로 남긴다.

## 검산과 실제 실행 기록

pure fixture 검사는 fake predictor로 단일 encoder/decode, box+양성점 한 개, 3개 마스크 보존, max quality exact-tie first를 확인했다. 금지 source 경로 5개 및 network/subprocess sentinel도 거부했다. 실제 RGB read와 NN 호출은 이 단위검사에서 0이다. 인위적인 fixture는 영상 성능 결과가 아니다.

protocol lock 및 baseline/empty identity 검사 통과 후 ROOT의 명시적 GO를 받아 실행했다. 실제 supplier 출력은 private namespace `supplier_v3/`에 있다. actual forward hook 계수는 image encoder 3, prompt encoder 3, mask decoder 3이며, prompt decode 호출도 3이다. 3개 no-admissible 행은 NN을 호출하지 않았다. 6개 원 RGB의 resolved 경로가 read ledger에 있으며 admissible RGB의 파일/픽셀 SHA와 원 canvas를 검사했다. 모델이 출력한 3개 대안을 각 프레임마다 NPZ/PNG에 보존했다.

| image_id | 고정 최대 quality 원 인덱스 | quality (원 인덱스 0/1/2) |
|---|---:|---|
| plastic_night_01:038630 | 2 | 0.956330538 / 0.962458968 / 0.965139151 |
| wood_day_01:005286 | 1 | 0.984083951 / 0.984312654 / 0.964729786 |
| plastic_day_01:020955 | 1 | 0.979053080 / 0.979118705 / 0.970536828 |

최초 프로세스 종료는 깨끗한 성공이 아니었다. 세 시도를 모두 남겼다.

- `supplier_v1`: CLI에서 neutral package SHA를 RGB JSONL SHA 대신 넘겨, manifest 검사에서 중단했다. NN 0회.
- `supplier_v2`: timm의 optional wandb import가 출력 밖 임시폴더를 탐색하려 해 read/write boundary가 막았다. 모델 생성/NN 전에 중단했다. 이후 runtime temp/cache를 새 출력 안으로 보냈으며 데이터 read allowlist를 넓히지 않았다.
- `supplier_v3`: 승인된 encoder 3회와 decoder 3회를 모두 끝내고 9개 PNG와 3개 NPZ를 썼다. 마지막 장부 검사에서 이미 차단했던 optional runtime `socket.bind` 1회, `subprocess.Popen` 4회를 보고 실패했다. 금지 source 파일 읽기 시도는 0이며 이 socket/child process 동작은 모두 차단되었다. 성공한 외부 연결이나 child process 실행 기록은 없다.

`SUPPLIER_FAILURE.json`의 실제 완료 계수와 파일을 유지한 상태로, 이미 저장된 NPZ/PNG 및 동결된 manifest만 읽어 최종 장부를 복구했다. 상태는 **`RECOVERED_AFTER_POST_FORWARD_PACKAGING_FAILURE`**다. 복구에서 NN 재실행/RGB 재디코딩은 0이다. 모든 NPZ에서 `selected_index == argmax(quality)` 및 원 인덱스 `[0,1,2]`, 3개 원 canvas bool 마스크, 유한 logits/quality를 확인했고 9개 PNG가 NPZ 마스크와 정확히 일치했다. 원 순위/선택이나 prompt는 바꾸지 않았다. raw manifest SHA는 `e8ff23847a7b7529d94f883df4b3cabe3a7c14a8a5ccbd4de23cf12c6e35a93a`다.

측정된 supplier_v3 전체 wall time은 **2.551405289 s**다. 이 값은 checkpoint/API import와 로딩, 세 프레임 segmentation, 출력 및 최종 실패 검사를 포함하며 RGB→pose 전체 latency가 아니다. 프레임별 encoder/decoder 및 model-load 시간은 실패 직전 메모리에만 남아 저장되지 않았으므로 **NA**다. 그 시간을 채우기 위한 추론 재실행은 하지 않았다. GPU 최대 메모리도 NA다. 이후 코드는 매 행을 즉시 쓰도록 수정했으며 pure tests만 재실행했다.

실행 장부는 `SUPPLIER_ATTEMPT_HISTORY.json`, `supplier_v3/SUPPLIER_RUN.json`, `SUPPLIER_FAILURE.json`, `SOURCE_ACCESS_AUDIT.json`, `BOUNDARY_SENTINELS.json`, `NON_TARGET_INPUT_MANIFEST.jsonl`다. raw 출력의 세 예측은 검토 전 `admitted_for_rerank=false`로 유지한다. ROOT가 새 비용을 열기 전 고정된 최대 quality 마스크를 원 RGB에서 한 번 검토하여 별도 frozen 입력을 만든다.

공개 최종 supplier 소스 SHA256는 `de1790288771a342ed070d1235a6a7e99c887a587d60c89725f4e9c9ffb9617e`다. 실제 추론 시 소스는 private `supplier_v3/SUPPLIER_SOURCE_AT_INFERENCE.py`로 보존했으며 SHA256 `066e12e8bc39a5ce273ee1524af04c4e2f50b245f0b270eca2d7b768d654697b`가 실행 receipt와 일치한다. 이후 수정은 출력 장부 처리에 한정된다. 각 행을 즉시 저장하고, source 파일 접근 거부는 실패로 유지하며 이미 차단된 optional runtime network/subprocess 시도는 별도 항목으로 남긴다. 네트워크/child process 차단과 RGB read allowlist는 완화하지 않았다. 성공/실패 모두 `finally`의 `SUPPLIER_FINAL_RECEIPT.json`에 실제 hook 계수, 보유한 프레임별 시간, 전체 시간, model-load 시간과 차단 범주를 남긴다. mock failure receipt의 JSON 저장/재읽기 검사는 통과했으며 수정 후 추가 실제 NN은 0이다.

재현 시 실제 장부는 원 출력에서 그대로 검사한다. 각 `*__alternatives.npz`를 `allow_pickle=False`로 열어 `masks`, `quality`, `low_resolution_logits`, `original_indices`, `selected_index`만 읽고, 선택 인덱스가 `np.argmax(quality)`인지 확인한다. PNG의 `>0` 배열과 각 NPZ bool 마스크의 exact equality, 연속 uint8 0/1 pixel SHA 및 file SHA를 재계산한다. 복구 완료 장부의 `execution_status`와 원 `SUPPLIER_FAILURE.json`을 함께 유지한다. 누락된 시간값을 새 추론이나 파일 수정시간으로 채우지 않는다.

pure 검사는 다음 명령으로 NN 없이 재현한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHON_ENV/bin/python \
  scripts/research/pallet_non_target_outline_pilot_v1/supplier.py --self-test
```

추론의 정확한 고정 인자: `--neutral-rgb .../neutral_v1/NEUTRAL_RGB.jsonl`, `--prompts .../PROMPT_MANIFEST.jsonl`, `--model-config .../SUPPLIER_MODEL_CONFIG.json`; 각 byte SHA는 `7f6c14351430df738cbb5ade4af394cd2851e0ca50d426437b8f94904443c1f9`, `36a8072c14897acfd29b04e954fb1f33366b2feb27924ffbffe4a6b36b26b030`, `fb93ad8b2f7a04dcafd96d85d75a8291f3fb2f01a19168bf4ce71be928a481ec`다. 환경은 `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1`, 위 기존 Python이다. `...`는 private `PRIVATE_AUDIT`을 뜻한다. 이미 한 번의 허용 추론을 소비했으므로 이 작업에서는 이 명령을 다시 실행하지 않는다.
