"""Append-only finalization after fixed-six geometry audit, before full50 evidence."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

out=ROOT/'PAIRED_REAL_LABEL_PROTOCOL.json'
assert not out.exists(),'Frozen protocol already exists'
p=json.loads((ROOT/'PAIRED_REAL_LABEL_PROTOCOL_DRAFT.json').read_text())
p['schema']='V66_PAIRED_PUBLIC_REAL_LABEL_PROTOCOL_FROZEN_SYMMETRIC_V2'
p['status']='FROZEN_FOR_FULL50_SOURCE_EVIDENCE_PREPARATION_CONDITIONAL_TEACHER; MODEL_FIT_HOLD'
p['freeze_basis']='Parent approved conditional-teacher preparation after actual fixed-six native overlays and independent mesh raster agreement. No performance/model/threshold selection and no current223 data used.'
p['scope']='Photographic same-camera/static-bin A/B conditional paired evidence, separate from controlled YCB paste truth. Fixed50 source pairs only; no model fitting authorized by this contract.'
p['pair_admission_prerequisites']=[
 'Both native RGB/depth/SAM have expected480x640 dimensions; source mm camera-Z interpretation is used. Missing/nonpositive/nonfinite depth yields no pixel evidence.',
 'Supplied K and4x4object-to-camera pose matrices exactly equal between A/B; same session, source split, and static camera/bin are conditional source assumptions.',
 'Supplied mesh raster: all vertices strictly in front, near_plane_crossing_triangles=0. Reject pair to UNKNOWN if geometric/metadata gates fail; no pose fitting or alignment repair.',
 'Source mesh scale/object origin and supplied pose convention are the independently audited native renderer recipe; renderer correctness does not validate mesh truth.',
 'RGB-depth registration is empirically supported by fixed-six inspection/residuals, NOT established by official dataset calibration metadata. This permits conditional TEACHER evidence only, never GOLD or registration-certified claims.',
 'Same physical instance and photographic pair assumptions apply; cross-object, semantic-self or human corner labels are not created.'
]
p['candidate_constants']['primary_pair_closer_gap_mm']=50
p['constants']=p.pop('candidate_constants')
p['pixel_facts']=p.pop('candidate_pixel_facts')
p['pixel_facts']['full_positive']='V_A OR V_B, shared by A/B, under conditional same-camera/static-bin assumptions; retained even if the other frame has UNKNOWN visibility.'
p['pixel_facts']['full_negative']='DISABLED. Always Boolean zeros in this version, including renderer background and SAM0. Therefore no O hard labels are produced.'
p['pixel_facts']['visible_negative_A']='E_A only. No source SAM-negative completeness assumption. B uses E_B.'
p['pixel_facts']['visible_negative_B']='E_B only. All other non-V_B pixels remain visible-UNKNOWN.'
p['pixel_facts']['conflicts']='Assert V_A&E_A=false, V_B&E_B=false, E_A&E_B=false. Metadata/geometric admission failure makes all pixels UNKNOWN, not O.'
p['pixel_facts']['known_class_A']='V_A OR E_A, distinct from shared full_positive. For B use V_B OR E_B.'
p['full_source_preparation_gate']={
 'manifest':'public_pallet/EXACT_POSE_PAIR_MANIFEST.json, fixed50 pairs;22TRAIN_LAB and28VAL_HALL',
 'split':'All lab sessions TRAIN; all hall sessions VAL from immutable source split lock. A/B always same split.',
 'allowed':'Generate and audit frozen conditional-teacher masks/certainty for all fixed50 pairs, preserving every diagnostic. No quality threshold optimization, per-pair residual selection, relabeling, model fit or current-evaluation access.',
 'hard_pair_checks':['same session/split','native shapes480x640','supplied K/T exact equality','all mesh vertices infront','no near-plane crossing'],
 'descriptive_only':['per-pair common-teacher20mm stable-depth fraction','mesh/depth signed/absolute residual distributions','valid-depth coverage','symmetric50/100/200mm gap counts','strict25pixel Ecore and reverse speckle counts'],
 'no_numeric_residual_admission_threshold':'Residuals and coverage describe empirical support; they do not create a registration certificate or choose/drop a pair.',
 'failures':'Preserve and report. No repair by fitting, alignment or mask/threshold changes. Any pair with a hard-check failure supplies UNKNOWN facts.',
 'fit_admission':'HOLD until independent dataset/artifact review and a separate frozen training/source-evaluation protocol.'
}
p['initial_audit']['held_out_status']='Fixed-six lab/hall source fixtures were inspected for provenance/geometry before full evidence preparation. Disclose this sourceVAL preview. No model/label thresholds chosen by those statistics.'
p['initial_audit']['no_model_training']=True
p['initial_audit']['geometry_replay']='Two independent native mesh raster implementations: all six finite support masks and depths exactly equal, no crossing triangles. This checks implementation agreement only.'
p['evidence_tiers']=[
 'CONTROLLED_YCB_ORIGINAL_OBSERVED_PLUS_OPAQUE_PASTE (separate frozen protocol)',
 'CONDITIONAL_PAIRED_SAM_MESH_DEPTH_TEACHER_VISIBLE',
 'CONDITIONAL_PAIRED_SYMMETRIC_FOREGROUND_DEPTH_TEACHER_EXTERNAL_SUPPORT',
 'CONDITIONAL_PAIRED_STRICT25PIXEL_EXTERNAL_SUPPORT_CORE',
 'MESH_OUTSIDE_PROXY_NOT_FULL_NEGATIVE',
 'UNKNOWN_MISSING_UNSUPPORTED_OR_CONFLICTING_EVIDENCE']
p['partial_label_storage']={
 'canonical_classes':{'UNKNOWN':-1,'VISIBLE':0,'EXTERNAL_SUPPORT':1,'OUTSIDE':2},
 'external_storage_encoding':'If intake uses0=U,1=V,2=E, name that encoding explicitly and convert before joint-state training/evaluation.',
 'required_per_frame':['visible_positive','visible_negative','visible_known','full_positive','full_negative','full_known','full_unknown','exact_state'],
 'joint_order':[['V','V'],['E','V'],['E','E'],['O','V'],['O','E'],['O','O']],
 'U_training_policy':'UNKNOWN is partial evidence, never a seventh class or an O/E hard negative. If later trained, use compatible joint-state likelihood and a separately frozen loss contract.',
 'pure_helper':'contract/paired_truth.py; constructs native erode2 teachers, symmetric evidence and correlated exact/r2 compatible sets.',
 'full_negative_enabled':False,
 'semantic_self':'UNKNOWN; geometry proxy, if used, remains separately named and supplied-pose-only.'
}
p['official_geometry_reference']={
 'url':'https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/',
 'verified_utc_date':'2026-10-22',
 'supported':'Native pixel centers integer; RealSense depth is camera Z rather than Euclidean range; RGB/depth sensors have distinct intrinsic/extrinsic geometry and require explicit stream alignment.',
 'not_supported':'This general manufacturer document does not certify this dataset underwent RGB-depth alignment or supply missing dataset calibration metadata.'
}
p['finalization']='FROZEN before full50 paired-teacher generation. Preserve DRAFT and V1 diagnostic results. Any future negative/full geometry extension requires a separately versioned protocol, not mutation of this file.'
for rel in ['source_audit/INDEPENDENT_FIXED_SIX_GEOMETRY.json','contract/paired_truth.py','contract/PAIRED_TRUTH_MOCK_PROOF.json','public_pallet/EXACT_POSE_PAIR_MANIFEST.json','public_pallet/FIXED_SIX_PAIR_SELECTION.json']:
 p['bindings'].append(bind(ROOT/rel))
for b in p['bindings']:
 assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256'],b['path']
out.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(bind(out),indent=2))
