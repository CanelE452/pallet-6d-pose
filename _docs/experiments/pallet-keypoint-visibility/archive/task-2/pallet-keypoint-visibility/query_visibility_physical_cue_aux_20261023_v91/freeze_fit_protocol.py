"""Root decisions, fixed before any V91 feature/checkpoint deserialization or fit."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

R = Path(__file__).resolve().parent
P = R.parent

def bind(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)

def write(path, value):
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

V88 = P/'query_visibility_external_ray_head_20261023_v88'
V89 = P/'query_visibility_public_cue_grid_20261023_v89'
V90 = P/'query_visibility_public_cue_inputs_20261023_v90'
T = R/'teacher_preparation'
C = T/'corpus_build_v1'

review = V90/'independent_saved_input_audit/REVIEW.json'
assert json.loads(review.read_text())['PASS'] is True
bindings = dict(
    model_code=R/'local_cue_evidence_map.py', mosaic_code=R/'spatial_mosaic.py',
    source_checkpoint=V88/'runs/YCB_EXTERNAL_RAY_V1/BEST_EXTERNAL_HEAD.pt',
    source_checkpoint_freeze=V88/'runs/YCB_EXTERNAL_RAY_V1/MODEL_FREEZE.json',
    source_checkpoint_completion=V88/'runs/YCB_EXTERNAL_RAY_V1/FIT_COMPLETE.json',
    teacher_contract=T/'TEACHER_CONTRACT_BEFORE_BUILD.json',
    teacher_seal=C/'GT_CORPUS_SEAL.json',
    teacher_units=C/'ALL77916_TILE_CUE_TRAINING_UNITS_GT_ONLY.npz',
    teacher_views=C/'ALL604_CUE_VIEW_TEACHERS_GT_ONLY.jsonl',
    teacher_schema=T/'TRAINER_GT_INTEGRATION_SCHEMA.json',
    independent_teacher_review=R/'independent_teacher_review/GT_UNITS_AND_WEIGHTS_REPLAY.json',
    independent_model_math_review=R/'independent_model_math_review/REVIEW.json',
    source_grid=V89/'inputs_metadata/INPUTS_GRID_ONLY.jsonl',
    source_frames=V89/'inputs_metadata/FRAMES_INPUTS_ONLY.jsonl',
    source_pack_manifest=V90/'input_pack/PACK_MANIFEST.json',
    source_pack_seal=V90/'input_pack/INPUT_PACK_SEAL.json',
    source_pack_QA=V90/'input_pack/INPUT_PACK_QA.json',
    source_pack_independent_QA=review,
)
protocol = dict(
    schema='V91_SOURCE_ONLY_PHYSICAL_CUE_REGION_AUXILIARY_FIT_V1',
    created_UTC=datetime.now(timezone.utc).isoformat(),
    status='CHOICES_FIXED_BEFORE_V91_SOURCE_FEATURE_OR_CHECKPOINT_DESERIALIZATION',
    bindings={k:bind(v) for k,v in bindings.items()},
    run_directory=str(R/'runs/PHYSICAL_CUE_AUXILIARY_V1'),
    parameter_count=25381,
    forward_fields=['p3_roi','local_rgb96','local_valid96'],
    architecture='Local 9x9 evidence map. Only V88 p3_encoder/local_encoder copied; new fusion default random, final evidence zero. No source fusion/classifier/target/context copied.',
    target=dict(
        codes={'0':'CONDITIONAL_PHYSICAL_CUE_OBSERVED_D','1':'CONDITIONAL_PHYSICAL_CUE_EXTERNALLY_OCCLUDED_E','-1':'UNKNOWN_TYPED'},
        definition='V85/V84 original physical structural cue at unchanged reference/uncertainty, before/after conditional AI observation. D is not wood-material membership; E is not foreign-depth single-ray proxy. These are not human native8 gold.',
        native8_mapping='UNKNOWN',
        loss_operator='For each admitted FULL cue/view/tile, validity-normalized bilinear interpolation at nine fixed uncertainty-region locations; mean nine LOGITS then ONE BCE. Not nine material pixel labels. U/MISS/PARTIAL/unsupported terms zero gradient.',
        loss_admission='Original teacher loss_admitted AND actual region_available; never change q/L or choose a best tile. Nominal admitted terms remain in the sampler; report unavailable count and effective mass. No additional all-supported gate.',
        uncertainty_interpretation='Coarse region evidence; 9x9 cell spacing is L/9, so no precise corner localization claim.',
        synthetic_YCB_role='Only pretrained shared encoders from V88, itself based on synthetic V75 and YCB V87. New supervised labels are public Pallox physical cues only; do not merge clear-ray proxy or synthetic material pixels into D gold.',
        instance_association='No target instance or native-slot identity input. Physical cue association/initializer cue MISS remains separate and unproven.'),
    data=dict(
        input_frames=100,input_tiles=12900,cue_view_rows=604,original_cues=302,tile_cue_units=77916,
        split='Original TRAIN_LAB44 frames/5676 tiles vs VAL_HALL56/7224. One physical high-sided bin and environment holdout; sourceVAL reused development, not independent object generalization.',
        known_raw_D_E_U=[261,37,306],known_duplicate_weighted_D_E=[257,37],
        train_known_views=128,val_known_views=170,
        input_construction='All fixed V89 tiles from raw RGB/frozen YOLO full P3, no label/GT reference/native8/context. Original explicit L128/256. V90 all100/12900 inputs audited.',
        labels_previously_seen='Counts, source images, conditional cue observations and GT geometry audit are known. V91 feature tensors and V88 checkpoint have not yet been deserialized for this fit. This is prospective fitting, not a blind annotation study.',
        private_provenance='User confirms current223 personally captured and unrelated to KU Leuven Pallox. Current223 underlying pixels/arrays/annotations remain unread in V77..V91.'),
    sampling=dict(batch_queries=96,per_class=48,class_order=[0,1],replacement=True,seed=20261023,
        within_class_probability='Normalize original teacher loss_weight; base cue/view mass1, each scale total.5 across all FULL tiles, seed duplicate members weight.5 in existing two groups.',
        unavailable='Keep nominal sampled terms; unavailable terms zero gradient. BCE mean over available sampled terms. Report loss of effective mass separately.',
        augmentation='No new online augmentation or query relocation.'),
    optimizer=dict(name='AdamW',transferred_encoder_lr=1e-4,new_fusion_output_lr=3e-4,weight_decay=.001,betas=[.9,.999],updates=2000,clip_grad_norm=5.,schedule='constant'),
    validation=dict(
        map_construction='Predict all129 fixed tiles of each of all56 VAL frames. Construct native480x640 map using fixed tent weights and validity-normalized bilinear logits. Complete every frame before the separate view-scoring reader is called. The full GT term loss-bank, including VAL coordinates, is loaded earlier, but only TRAIN terms enter training; no GT argument or label/region/FULL based tile selection enters mosaic construction.',
        region='At completed native map sample same fixed nine uncertainty locations and mean LOGITS; r-only is a separate diagnostic.',
        uncertainty_or_no_support='Factual no support/outside remains U, never forced D/E. Teacher U not treated as correct model U.'),
    selection=dict(every_updates=100,criterion='Minimum equal mean of duplicate-weighted D and E known VAL region BCE; strict lower, earliest tie; only100..2000 eligible, update0 diagnostic.',threshold=.5,
        unavailable_region_NLL='FIXED_ZERO_LOGIT_BCE_LOG2_DIAGNOSTIC_PENALTY',
        calibration='Fixed zero-logit threshold only; balanced-training scores are not calibrated deployment likelihood. No source/current threshold search.'),
    source_gate_before_results=dict(
        requirements={k:{'D_recall':'>=0.8','E_recall':'>=0.8','E_precision':'>=0.5','factual_U_rate':'<=0.1'} for k in ['region','reference']},
        paired_external_requirement='VAL original22 D-to-E pairs: region beforeD and afterE both correct rate>=.7',
        total_gates=9,
        scope='All170 known VAL cue views, not only favorable tiles. U counted in recall denominator; all336 VAL views and166 teacherU reported separately. Stable63 D/D pairs descriptive.',
        pass_consequence='All9 permits preparation of separately frozen current223 native-query-centre transfer diagnostic only. Does not prove native8 mapping/instance association/self occlusion or make deployment/goalsolved claim.',
        fail_consequence='No current223 reads/evaluation. Preserve failed fit, diagnose source output, no gate/threshold change or hidden second fit.'),
    reporting='TRAIN/VAL region and r-only 2x3 CM rowsD/E colsD/E/U, classP/R/U, paired D/E and D/D, teacherU counts/distribution; save all12900 tilemaps/all100 native mosaics/all604 scores including U. AI/onebin limits explicit.',
    runtime=dict(python='/home/minjae/anaconda3/bin/python',torch='2.5.1+cu121',device='cuda:0',CPU_threads=1,deterministic=True,TF32=False,exact_GPU_backward_bit_claim=False),
    execution='Generated tests and independent trainer/protocol review before root approved entry. Actual source GPU preflight has0 optimizer updates; one2000-update fit only. Preserve failures/code/log/partials, never silently restart after first update.',
    current223_reads_allowed=False,PnP_allowed=False,fully_solved=False,goal_status='active')
write(R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json',protocol)
write(R/'FIT_PROTOCOL_BINDING.json',dict(protocol=bind(R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'),root_choices_script=bind(__file__)))
print(json.dumps(bind(R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json')))
