"""Metadata/code/stat-only inventory. Never opens weights, RGB, labels or scores."""
from pathlib import Path
import hashlib
import json

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OUT = Path(__file__).resolve().parent
V63 = P / 'query_visibility_public_real_20261022_v63'
V64 = P / 'query_visibility_multitarget_20261022_v64'
V65 = P / 'query_visibility_amodal_global_20261022_v65'
V66 = P / 'query_visibility_controlled_real_20261022_v66'
V67 = P / 'query_visibility_visible_preservation_20261023_v67'
V68 = P / 'query_visibility_sam21_frozen_20261023_v68'
V70 = P / 'query_visibility_target_ownership_20261023_v70'
V81 = P / 'query_visibility_grounded_target_20261023_v81'
V115 = P / 'query_visibility_loco_annotation_target_masks_20261023_v115'
RT = P / 'foundation_runtime_sam21_20261023'

evidence = []


def metadata(path):
    path = Path(path)
    assert path.suffix == '.json'
    assert '_v112' not in str(path).lower()
    assert not any(t in str(path).lower() for t in ['private_evaluation', 'private_primary'])
    body = path.read_bytes()
    evidence.append({'path': str(path), 'sha256': hashlib.sha256(body).hexdigest(),
                     'kind': 'metadata', 'bytes': len(body)})
    return json.loads(body)


def code_binding(path):
    path = Path(path)
    assert path.suffix == '.py' and '_v112' not in str(path).lower()
    body = path.read_bytes()
    evidence.append({'path': str(path), 'sha256': hashlib.sha256(body).hexdigest(),
                     'kind': 'code', 'bytes': len(body)})


def frozen_run(root, name):
    d = metadata(root / 'runs' / name / 'MODEL_FREEZE.json')
    return {'run': name, 'completed': d.get('completed'),
            'optimizer_updates': d.get('optimizer_updates'),
            'best_update': d.get('best_update'), 'parameters': d.get('parameters'),
            'target': d.get('target'), 'head_config': d.get('head_config'),
            'runtime_recorded': d.get('runtime'),
            'checkpoint_body_read_for_inventory': False}


def asset_stat(path, recorded_sha=None):
    path = Path(path)
    return {'path': str(path), 'exists': path.is_file(),
            'bytes_now_stat_only': path.stat().st_size if path.is_file() else None,
            'sha256_inherited_from_metadata_not_rehashed': recorded_sha,
            'body_read_or_deserialized': False}


mobile_source = metadata(V63 / 'sam_source/SEAL.json')
mobile_standalone = metadata(V63 / 'sam/PREDICTION_SEAL.json')
mobile_smoke = metadata(V63 / 'sam/CPU_LOAD_SMOKE.json')
v63runs = [frozen_run(V63, s) for s in ['SYNTH_CONTEXT_RGB_SAM_V4', 'MIXED_CONTEXT_RGB_SAM_V4']]
v64runs = [frozen_run(V64, s) for s in ['SYNTH_MULTITARGET_UNWEIGHTED', 'MIXED_MULTITARGET_UNWEIGHTED']]
v65runs = [frozen_run(V65, s) for s in ['SYNTH_GLOBAL_UNWEIGHTED', 'MIXED_GLOBAL_UNWEIGHTED']]
v65extract = metadata(V65 / 'sam_source/SEAL.json')
v65protocol = metadata(V65 / 'PROTOCOL.json')
v66runs = [frozen_run(V66, s) for s in ['CONTROLLED_ONLY', 'CONTROLLED_PLUS_PALLOX']]
v67runs = [frozen_run(V67, 'VISIBLE_BCE1')]
v70run = frozen_run(V70, 'OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2')
v70protocol = metadata(V70 / 'FIT_PROTOCOL.json')
v70decision = metadata(V70 / 'SOURCE_ADMISSION_DECISION.json')
v68 = metadata(V68 / 'RESULTS_KO.json')
v81 = metadata(V81 / 'segmentation/outputs/SUMMARY.json')
v81protocol = metadata(V81 / 'segmentation/SAM_STAGE_PROTOCOL_V2.json')
v115 = metadata(V115 / 'source_sam_candidates_v1/SUMMARY.json')
v115decision = metadata(V115 / 'ROOT_ACTUAL_FIXED_VISUAL_REVIEW_AND_DECISION.json')
runtime = metadata(RT / 'RUNTIME_SEAL.json')
for root, name in [(V63, 'model/query_head.py'), (V63, 'build_sam_source_cache.py'),
                   (V64, 'runs/MIXED_MULTITARGET_UNWEIGHTED/code/query_multitarget_head.py'),
                   (V65, 'model/query_global_amodal_head.py'),
                   (V70, 'model/ownership_head.py'), (V70, 'model/input_prototype.py'),
                   (V115, 'sam_candidate_preparation/produce_sam_candidates_once.py')]:
    code_binding(root / name)

mobile_sha = mobile_source['checkpoint_sha256']
mobile_vendor = Path('/home/minjae/Documents/github/pallet-pose/_docs/experiments/pallet_visibility_curriculum_20261013_v1/foundation_mask_v1/vendor/MobileSAM')
assets = [asset_stat(mobile_vendor / 'weights/mobile_sam.pt', mobile_sha),
          asset_stat(V63 / 'model/deployment_assets/MobileSAM/weights/mobile_sam.pt', mobile_sha),
          asset_stat(runtime['checkpoint']['path'], runtime['checkpoint']['sha256'])]

result = {
 'schema': 'READONLY_LOCAL_SAM_AND_OWNER_EXPERIMENT_INVENTORY_V1',
 'completed': True,
 'scope': {'metadata_code_and_filename_stat_only': True, 'new_model_calls': 0,
           'fit_calls': 0, 'downloads': 0, 'weight_body_reads': 0, 'RGB_reads': 0,
           'label_or_score_body_reads': 0, 'forbidden_phase_access': 0,
           'private_current_access': 0, 'PnP': 0, 'primary_denominator_changes': 0},
 'main_conclusion_KO': 'SAM 특징과 작은 CNN, SAM 가중 대상 prototype을 함께 학습한 기록이 이미 있다. SAM/owner branch 자체를 미시도 방법으로 볼 수 없다. SAM2.1은 실제 실행됐지만 검토한 완료 기록은 frozen mask proxy 또는 source 후보 조사이며 현재 native physical-corner CF 의미를 학습한 동일 조건의 검증은 아니다.',
 'experiments': [
  {'phase': 'V63', 'kind': 'frozen MobileSAM standalone and local learned residual CNN',
   'frozen_source_extraction': {k: mobile_source[k] for k in ['targets','frames','seconds','GT_masks_read','fit_updates']},
   'standalone': {k: mobile_standalone[k] for k in ['targets','image_encoder_forwards','prompt_decoder_calls','seconds']},
   'feature': 'Selected full-image MobileSAM logit sampled at original query/L 9x9; /8 clipped[-4,4], invalid zero; F16. Local SAM CNN + raw RGB CNN + P3/context CNN; learned raw-logit residual prior.',
   'target_scope': 'Exact query-pixel target-instance visible-mask membership. Not registered physical-corner external/self semantics.',
   'prompt_scope': 'Box-only, no positive query click, highest predicted quality among alternatives. Source YCB boxes were annotation-derived fixed jitter; synthetic boxes initial-query bounds.',
   'runs': v63runs},
  {'phase': 'V64', 'kind': 'two SAM-channel multi-target spatial CNN',
   'feature': 'SAM exact+r2 local channels with P3, RGB75, context and box grid, two 3x3 SAM convolutions; features only, no SAM sign veto.',
   'target_scope': 'Exact/radius2 visible and full-instance support, six paired joint states VV_EV_EE_OV_OE_OO; not native8 physical-corner semantics.',
   'runs': v64runs},
  {'phase': 'V65', 'kind': 'global RGB/SAM shape U-Net and local SAM CNN',
   'extraction': {k: v65extract[k] for k in ['targets','frames','local_exact_and_r2_replay_cells','local_replay_mismatches','GT_masks_read']},
   'feature_contract': v65protocol['input'], 'runs': v65runs},
  {'phase': 'V66/V67', 'kind': 'same global/local SAM architecture under controlled public-real and visible-BCE experiments',
   'feature': 'V65 95,508-parameter RGB/SAM/global+local architecture; fresh frozen MobileSAM on composites/public pallet sources. V67 added visible-preservation BCE.',
   'runs': v66runs + v67runs},
  {'phase': 'V68', 'kind': 'actual frozen SAM2.1 box-only baseline',
   'completed_targets': v68['completed_targets'], 'completed_queries': v68['completed_queries'],
   'native_valid': v68['native_valid'], 'native_invalid_U': v68['native_invalid_U'],
   'training_updates': v68['training_updates'], 'verdict': v68['verdict'],
   'diagnostics_pass_fail': [v68['source_diagnostic_pass'], v68['source_diagnostic_fail']],
   'semantic_limit': v68['full_goal_limit'],
   'recipe': 'Same rough box, no query-positive clicks. Highest predicted quality multimask, raw0 exact query readout; radius2 reported separately. No learned SAM2 branch here.'},
  {'phase': 'V70', 'kind': 'actual SAM-conditioned target ownership/prototype CNN',
   'feature': 'MobileSAM2ch + P3 + raw RGB75. Other same-current-target ROIs weighted by predicted SAM foreground, box validity and sigmoid(logit), leave-one-entire-ROI-out prototype; query-target cosine/product/absolute difference. Fallback context flagged, no GT mask input.',
   'target_scope': 'Known exact visible/external conditional target-support facts from SYNTH/original YCB/controlled YCB/Pallox; original full-only unknown not treated as E. These legacy eight query slots are not all registered native corners.',
   'new_backbone_or_SAM_calls_in_this_fit': v70protocol.get('new_backbone_or_SAM_calls'),
   'run': v70run, 'verdict': v70decision['verdict'],
   'broader_eligibility': v70decision['broader_eligibility'],
   'limitation': 'Predicted foreground can include occluder; other ROIs can overlap, no clean ownership guarantee. Architecture/objective/sampler changed together.'},
  {'phase': 'V81', 'kind': 'actual grounded detector-box to frozen SAM2.1 pipeline',
   'summary': {k: v81[k] for k in ['completed','PASS','targets','unique_RGB','query_slots','associated','unassociated','embedding_calls','query_predictions','fit']},
   'input_contract': v81protocol['input'],
   'semantic_limit': v81protocol['semantics'],
   'mask_recipe': 'Single detector-selected box, no points, no multimask/quality selection, no postprocessing. Unassociated target is U; zero mask is only NOT_VISIBLE_MASK.'},
  {'phase': 'V115', 'kind': 'actual public LOCO annotation-target plus competitor SAM2.1 mask pilot',
   'summary': {k: v115[k] for k in ['completed','PASS','frames','request_slots','primary_masks','missing_competitor_no_calls','set_image_calls','total_predict_calls','original_query_slots','native_mapping','human_gold','semantic_D_E_S_created','training_admission','fit']},
   'prompt_scope': 'Official annotation-entity bbox target and fixed nearest annotation-entity competitor bbox, not GT-free deployment target association.',
   'decision': v115decision['decision'], 'why': v115decision['why'],
   'limitation': 'Target/competitor masks can overlap and do not establish depth order/ownership. All semantic UNKNOWN retained. No trained joint competitor-owner branch in this pilot.'}
 ],
 'local_assets': assets,
 'runtime': {
   'MobileSAM': {'vendor': str(mobile_vendor), 'deployment_vendor': str(V63 / 'model/deployment_assets/MobileSAM'),
                'parameters_from_existing_CPU_smoke': mobile_smoke['parameters'],
                'new_import_or_smoke': False,
                'recorded_compatible_training_python': '/home/minjae/anaconda3/bin/python',
                'recorded_Torch': '2.5.1+cu121',
                'runtime_evidence': 'V64/V65/V66 MODEL_FREEZE and V63 CPU_LOAD_SMOKE; no fresh package compatibility claim'},
   'SAM2.1': {'READY_from_existing_seal': runtime['READY'], 'python': runtime['python'],
              'python_exists_now': Path(runtime['python']).exists(), 'config': runtime['config'],
              'repo_revision': runtime['repo_revision'], 'pinned_versions': runtime['pinned_versions'],
              'license_recorded': runtime['license'], 'generated_smokes_recorded': runtime['generated_smokes'],
              'new_import_or_smoke': False},
   'SAM3': {'local_ready_asset_found_in_inspected_locations': False,
            'scope': 'Project root dedicated runtime/asset names, github root SAM-named repos, torch checkpoint cache, Hugging Face cache first-level SAM names only; no exhaustive machine-wide absence claim.',
            'download_or_remote_availability_check': False}
 },
 'nonduplication_assessment_KO': [
  'SAM 중심 mask0/1 또는 local SAM CNN 추가, 글로벌 RGB+SAM U-Net, SAM 가중 LOO 대상 prototype은 이미 시험했다. 이를 새 방법의 근거로 삼으면 중복이다.',
  '남은 구체적 차이는 현재 actual initial q8/L 및 native CF D/E/S/U 라벨을 그대로 유지한 동일 source control에서 SAM2.1의 예측 target/competitor 관계를 조건으로 학습하는 것이다. 기존 완료 기록과 입력·라벨 조건이 다르며 성공은 아직 입증되지 않았다.',
  'V81은 예측 detector target association을 이미 시도했고 V115는 target+competitor mask를 이미 추출했다. 따라서 detector association 또는 competitor mask의 존재 자체도 새로움의 근거는 아니다.',
  '기존 MobileSAM learned-owner 실패는 SAM2.1 joint native conditioning의 불가능성을 증명하지 않는다. 반대로 더 큰 SAM 이름만으로 개선을 예상할 근거도 없다.',
  '모든 초기 slot/원래 input U와 OOB 분모를 보존해야 한다. SAM 미검출·mask0을 E로 바꾸거나 성공 mask만 남기면 목표와 분모가 바뀐다. GT mask/GT bbox를 배포 입력으로 주지 않는다. Query를 positive SAM click으로 넣으면 보임을 강제할 수 있으므로 기존 box-only readout과 동일한 의미가 아니다.'
 ],
 'evidence': evidence
}
path = OUT / 'INVENTORY.json'
assert not path.exists(), 'Inventory is append-only: use a new version instead of replacing it.'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'completed': True, 'output': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'evidence_files': len(evidence), 'weight_body_reads': 0, 'model_calls': 0}))
