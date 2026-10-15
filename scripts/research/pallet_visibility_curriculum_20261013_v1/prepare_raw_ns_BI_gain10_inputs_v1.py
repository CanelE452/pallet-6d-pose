"""Extract the user's posthoc gain10 prediction-only packets; no GT/image IO."""
from pathlib import Path
import hashlib
import json

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
C=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1'
B=C/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
BASE=C/'posefix_sub_comparison_v1'
PAIR=B/'raw_yolo_vs_N3_subpix_TEST49_paired_v1'
OUTPUT=B/'raw_ns_BI_gain10_input_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rel(p):return str(p.relative_to(ROOT))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run():
    if OUTPUT.exists():raise FileExistsError('Preserve every extraction attempt')
    seal=read(PAIR/'SEAL.json')
    for name,digest in seal['assets'].items():
        if sha(PAIR/name)!=digest:raise ValueError('Frozen pair input changed')
    pairs=read(PAIR/'PAIR_METRICS.json');ids=pairs['groups']['gain']
    assert len(ids)==len(set(ids))==10 and pairs['group_counts']['loss']==0
    packet=read(BASE/'BASELINE_INPUTS.json');protocol=read(BASE/'PROTOCOL.json')
    assert packet['prediction_only']is True
    source_inputs=C/'PREDICTION_INPUTS.json'
    assert sha(source_inputs)==protocol['source_bindings'][str(source_inputs)]
    original=read(source_inputs);original_lookup={f['id']:f for f in original['frames']}
    frames=[];lookup={f['frame']['id']:f['frame']for f in packet['frames']}
    arms={'RAW_YOLO':'BASE','N3_SUBPIX':'N3_SUBPIX'}
    for fid in ids:
        f=lookup[fid]
        assert f==original_lookup[fid]and f['partition']=='TEST'
        common={key:f[key]for key in ('id','session','image','image_sha256','raw_hw','K','xyz','selected_index')}
        common['candidate_metadata']=f['candidate_metadata']
        common['arms']={}
        for name,key in arms.items():
            support=f['prediction_support'][key]if isinstance(f['prediction_support'],dict)else f['prediction_support']
            q=f['points'][key];conf=f['candidate_metadata']['keypoints_conf']
            assert len(q)==len(conf)==len(support)==9
            common['arms'][name]=dict(native_points=q,keypoints_conf=conf,prediction_support=support,
                coarse_pose=f['initial_poses'][key],source_point_key=key,
                confidence_source='Original selected YOLO detection confidence reused; no separately calibrated N3 confidence',
                coarse_source='Saved arm own initial_pose from original prediction-only PREDICTION_INPUTS; not E/GT')
        frames.append(common)
    OUTPUT.mkdir()
    inference=dict(schema='POSTHOC_GAIN10_RAW_AND_N3_PREDICTION_ONLY_INPUTS_V1',ordered_ids=ids,arms=list(arms),frames=frames,
        posthoc_user_selected_comparison=True,full_population_generalization_evidence=False,
        numeric_reference_metrics_labels_or_GT_fields_included=False,
        state_and_BI_postprocessing_must_be_recomputed_separately_for_each_arm=True,
        native_point_archives_are_before_new_state_gate=True)
    write(OUTPUT/'INPUTS.json',inference)
    bindings={rel(p):sha(p)for p in (source_inputs,BASE/'BASELINE_INPUTS.json',BASE/'PROTOCOL.json',BASE/'BASELINE_PROVENANCE.json',PAIR/'PAIR_METRICS.json',PAIR/'SEAL.json')}
    write(OUTPUT/'INPUT_SEAL.json',dict(schema='SEALED_POSTHOC_GAIN10_PREDICTION_ONLY_INPUTS_V1',rows=10,ordered_ids=ids,arms=list(arms),
        inputs_sha256=sha(OUTPUT/'INPUTS.json'),source_sha256={rel(Path(__file__).resolve()):sha(Path(__file__).resolve())},
        input_sha256=bindings,reference_or_image_file_opens=0,old_sources_predictions_or_scores_changed=False))
    # This file is for the postprediction scorer only. It is deliberately not
    # embedded into INPUTS or its worker records; no reference numeric data read.
    write(OUTPUT/'SCORING_REFERENCE_MANIFEST.json',dict(schema='SCORER_ONLY_FIXED_TEST49_REFERENCE_BINDING_FOR_GAIN10_V1',
        ordered_ids=ids,reference_binding=pairs['reference_binding'],reference_document_was_not_opened=True,
        truth_lookup="document['rows'][id]['target']['truth']",prediction_seal_required_before_reference_IO=True,
        inference_INPUTS_sha256=sha(OUTPUT/'INPUTS.json'),input_seal_sha256=sha(OUTPUT/'INPUT_SEAL.json'),
        old_metric='scripts.research.pallet_visibility_curriculum_20261013_v1.scoring.pose_metric',
        thresholds='translation_cm <5 AND rotation_deg <5; fixed legacy physical symmetry metric'))
    write(OUTPUT/'SCORING_INPUT_SEAL.json',dict(manifest_sha256=sha(OUTPUT/'SCORING_REFERENCE_MANIFEST.json'),
        inference_input_seal_sha256=sha(OUTPUT/'INPUT_SEAL.json'),reference_numeric_decodes=0))
    return dict(rows=10,input_seal_sha256=sha(OUTPUT/'INPUT_SEAL.json'),inputs_sha256=sha(OUTPUT/'INPUTS.json'),
        scoring_manifest_sha256=sha(OUTPUT/'SCORING_REFERENCE_MANIFEST.json'))


if __name__=='__main__':print(run())
