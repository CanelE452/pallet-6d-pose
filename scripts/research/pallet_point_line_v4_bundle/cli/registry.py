"""SOURCE_REGISTRY.json / DATA_CONTRACT.json / ADAPTER_MAPPING.md from real files."""
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import torch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from cli.common import ROOT, RESULTS, STRUCTURED_V2, DECODER_PROBE, backbone_binding, read, write, sha256, git  # noqa: E402

READ_SOURCES = [
    'scripts/research/pallet_dht_structured_v2/model.py',
    'scripts/research/pallet_dht_structured_v2/data_ops.py',
    'scripts/research/pallet_dht_structured_v2/proposals.py',
    'scripts/research/pallet_dht_structured_v2/cache.py',
    'scripts/research/pallet_dht_structured_v2/infer.py',
    'scripts/research/pallet_dht_structured_v2/train.py',
    'scripts/research/pallet_dht_structured_v2/evaluation.py',
    'scripts/research/pallet_dht_structured_v2/SEMANTICS.md',
    'scripts/research/pallet_dht_decoder_probe_v1/cache.py',
    'scripts/research/pallet_dht_decoder_probe_v1/geometry.py',
    'scripts/research/pallet_dht_joint_v1/hough_block.py',
    'scripts/research/pallet_dht_joint_v1/integration.py',
    'scripts/research/pallet_dht_joint_v1/evaluate.py',
    'scripts/research/pallet_dht_joint_v1/train.py',
    'scripts/research/pallet_line_pose_v1/source_data.py',
]


def main():
    import ultralytics
    export = RESULTS / 'export'
    manifest = read(DECODER_PROBE / 'MANIFEST.json')
    cache = read(STRUCTURED_V2 / 'CACHE_COMPLETION.json')
    protocol = read(STRUCTURED_V2 / 'PROTOCOL.json')
    write(RESULTS / 'SOURCE_REGISTRY.json', dict(
        schema='pointline_v4_source_registry_1',
        repository_commit=git('rev-parse', 'HEAD'),
        repository_status_short=git('status', '--short'),
        bundle_reviewed_commit=read(KIT / 'BUNDLE_MANIFEST.json')['source_repo_commit_reviewed'],
        read_source_sha256={p: sha256(ROOT / p) for p in READ_SOURCES if (ROOT / p).is_file()},
        bundle_core_sha256={p.name: sha256(p) for p in sorted((KIT / 'pointline_v4').glob('*.py'))},
        adapter_sha256={p.name: sha256(p) for p in sorted((KIT / 'cli').glob('*.py'))},
        backbone=backbone_binding(),
        frozen_upstream=dict(
            structured_v2_cache_completion_sha256=sha256(STRUCTURED_V2 / 'CACHE_COMPLETION.json'),
            structured_v2_protocol_sha256=sha256(STRUCTURED_V2 / 'PROTOCOL.json'),
            decoder_probe_manifest_sha256=sha256(DECODER_PROBE / 'MANIFEST.json'),
            decoder_probe_extraction_sha256=sha256(DECODER_PROBE / 'EXTRACTION_COMPLETE.json'),
            split=protocol['split'], populations=cache['populations'], real_GT_read=cache['real_GT_read']),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                         device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         conda_env='pallet-yolo26'),
        observation_note='[확인] 실제 파일을 읽어 등록했다. 문서 주장만 확인한 항목은 없다.'))

    write(RESULTS / 'DATA_CONTRACT.json', dict(
        schema='pointline_v4_data_contract_1',
        populations={role: len(read(export / f'{role}.json')['records']) for role in ('train', 'calibration', 'synth_val')},
        expected=dict(train=1792, calibration=256, synth_val=512),
        real_dev_positive_deferred=319,
        split_lineage=('structured-v2 assigned calibration by SHA of '
                       f"{protocol['split']['calibration_prefix']} over the 2,048 synth_train pool; "
                       'the remaining 1,792 are train. synth_val is the upstream decoder-probe val split.'),
        held_out_final=False,
        dev_reuse_warning='real_dev positive319 and its 13 sessions have been evaluated repeatedly; this is not a holdout.',
        state='native', artificial_corruption_used_in_main_training=False,
        centroid_policy='candidate index8 copies the original prediction exactly',
        candidate_bank='structured-v2 build_proposals, 64 slots, identity at index0, GT-free',
        gt_source_synthetic='frozen structured-v2 targets (points/valid/matched); real GT is not exported yet'))

    (RESULTS / 'ADAPTER_MAPPING.md').write_text('''# ADAPTER_MAPPING — 확인한 소스 key → 새 export key

새 export 는 이 패키지가 정의한 인터페이스다. 아래 왼쪽은 **실제 파일을 읽어 확인한**
기존 key 이고 오른쪽이 새 계약이다.

| 새 `Observation` 필드 | 실제 출처 | 확인 위치 |
|---|---|---|
| `features` = (P3, P4) | `HoughFeatureFusion.forward(features)` 의 `features[0]`, `features[1]` — **블록 입력**, residual 되먹임 전 | `pallet_dht_joint_v1/hough_block.py:113-121` (`if len(features)!=3: raise ... expected P3, P4 and P5`), 채널 `channels=(64,128,256)` |
| `raw_to_feature` | `raw_to_input_affine` (structured-v2 packed, FP64) ÷ stride, offset `(-0.5,-0.5)` | `pallet_dht_structured_v2/cache.py` INPUT_SPECS `raw_to_input_affine`; offset 계약은 `pallet_dht_decoder_probe_v1/cache.py:sample_visual` (`grid=2*input/size-1`, `align_corners=False`) |
| `content_valid` | affine 역변환으로 raw 사각형 `[0,W-1]x[0,H-1]` 안에 드는 feature 중심 | `pointline_v4.adapter_helpers.content_mask_from_raw_rectangle` + 실제 raw H/W (`MANIFEST.json` `width`/`height`) |
| `baseline`, `point_valid`, `point_conf` | `baseline_points`, `point_valid`, `point_conf` | structured-v2 `INPUT_SPECS` (예측 기반 유효성, GT 아님) |
| `layouts`, `candidate_valid` | `build_proposals(...)['layouts'], ['valid']` — native(clean) 상태 | `pallet_dht_structured_v2/proposals.py`; 64슬롯·identity index0·centroid는 원본 복사 |
| `line_h`, `line_logits`, `line_valid` | `line_h`, `peak_logits`, `peak_valid` | structured-v2 `cache.pack_observation` (raw 공간 homogeneous line; 저장된 `line_fusion__line_peaks_h_raw` 와 동일성 검증됨) |
| `diagonal` | `diagonal` = `hypot(width,height)` (raw px) | structured-v2 `INPUT_SPECS` |
| supervision `points`/`supervised`/`matched` | `targets['points']` / `targets['valid']` / `targets['matched']` | structured-v2 `TARGET_SPECS`; **별도 파일**, Observation 으로 전달되지 않음 |

## 새로 수행한 것과 재사용한 것

- **새로 수행**: 동일한 검증된 predictor(`CanonicalPredictor`, checkpoint SHA
  `0960fb32…`)로 2,560장 실제 forward 를 다시 돌려 **native P3** 를 캡처했다.
  기존 캐시에는 P4 만 있었다.
- **재사용**: baseline 점·역할선·교점·합성 GT 는 frozen structured-v2 pack 그대로다.
- **엄격 parity**: 새로 캡처한 P4 를 frozen 캐시 P4 와 비교해 **2,560/2,560 bit-exact**
  (max abs diff 0.0). 따라서 새 P3 는 그 baseline 을 만든 것과 같은 forward 산물이다.
  이 결과는 `EXTRACTION_COMPLETE.json` 이 기록한 수치환경(`cudnn_allow_tf32=True`)을
  그대로 재현했을 때 얻어졌다. 더 엄격한 설정(TF32 off)에서는 max abs diff 6.2e-2 로
  bit-exact 가 아니며, 이 사실을 덮지 않는다.

## 알려진 한계

- feature 평면은 batch 처리를 위해 P3 `(64,80,80)`, P4 `(128,40,40)` 으로
  **오른쪽·아래 zero padding** 했다. affine 은 그대로 두고 `content_valid` 가
  padding 과 letterbox 여백을 배제한다 (실측: 416x640 입력에서 P4 유효 612 cell,
  기하 예상 616).
- `content_valid` 는 raw 화소가 존재하는 cell 표시일 뿐, backbone 수용영역에 이미
  들어온 padding 영향까지 제거하지 못한다. 가시성 감독도 아니다.
- feature index offset 은 `-0.5` (기존 검증된 repo sampler 계약). `AFFINE_PROBE.json`
  의 실측은 stride(1/8, 1/16)는 확인하지만 `-0.5` 와 수용영역 중심 기준 `-0.469`
  (0.03 cell 차)를 구분할 해상도가 없다. 이 상수는 네 군에 공통이라 군간 차이를
  만들 수 없다.
''', encoding='utf8')
    print('registry written')


if __name__ == '__main__':
    main()
