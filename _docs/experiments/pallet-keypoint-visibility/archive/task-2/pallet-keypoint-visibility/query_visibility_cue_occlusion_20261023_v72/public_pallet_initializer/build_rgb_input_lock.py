"""Freeze the existing 48 public R2B RGB inputs; no model or pixel decode."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import ast
import hashlib
import json
import struct

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def write_json(name, value):
    p = OUT / name
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return p


def main():
    assert BASE.name == 'pallet-keypoint-visibility'
    feasibility_path = BASE / 'query_visibility_reference_jitter_20261023_v71/source_audit/NEXT_PUBLIC_PALLET_INITIALIZER_FEASIBILITY.json'
    feasibility = json.loads(feasibility_path.read_text())
    for k in ['model', 'code', 'source_provenance']:
        b = feasibility[k]
        assert sha(b['path']) == b['sha256']
        assert Path(b['path']).stat().st_size == b['bytes']
    tree = ast.parse(Path(feasibility['code']['path']).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FrozenYoloFeatures')
    predict = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
    call = next(n for n in ast.walk(predict) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'predict')
    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
    expected = dict(conf='0.001', imgsz='640', rect='True', augment='False', half='False', device='self.device', verbose='False', save='False', stream='False')
    assert kwargs == expected

    rows, sources = [], []
    for clip, expected_n in [('r2b_storage_v3', 32), ('r2b_galileo2_v1', 16)]:
        directory = BASE / 'public_real_sources_20261023' / clip
        manifest = directory / 'RGB_FIXED_MANIFEST.jsonl'
        source_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        assert len(source_rows) == expected_n
        sources.extend([binding(manifest), binding(directory / 'INTAKE_SEAL.json')])
        for r in source_rows:
            p = Path(r['image'])
            data = p.read_bytes()
            assert hashlib.sha256(data).hexdigest() == r['image_sha256']
            assert data[:8] == b'\x89PNG\r\n\x1a\n'
            w, h = struct.unpack('>II', data[16:24])
            assert (h, w) == (r['height'], r['width'])
            assert r['encoding'] == 'rgb8'
            rows.append(dict(
                id=f"{clip}:{r['sensor']}:{r['sensor_slot']:02d}",
                image=str(p.resolve()), image_sha256=r['image_sha256'],
                RGB_pixels_sha256=r['RGB_pixels_sha256'], raw_hw=[h, w],
                source_clip=clip, sensor=r['sensor'], topic=r['topic'],
                sensor_slot=r['sensor_slot'], frame_index=r['index_in_sorted_topic'],
                timestamp_ns=r['bag_timestamp_ns'], encoding=r['encoding']))
    assert len(rows) == 48 and len({r['id'] for r in rows}) == 48
    assert len({r['image_sha256'] for r in rows}) == 48
    assert Counter((r['source_clip'], r['sensor']) for r in rows) == {
        ('r2b_storage_v3', 'D455_COLOR'): 16,
        ('r2b_storage_v3', 'HAWK_LEFT'): 16,
        ('r2b_galileo2_v1', 'D455_COLOR'): 16}
    assert not any(any(k in r for k in ['q', 'query', 'box', 'mask', 'GT', 'cue', 'label', 'pose', 'depth']) for r in rows)
    inputs = OUT / 'INPUTS_RGB_ONLY.jsonl'
    assert not inputs.exists()
    inputs.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))
    protocol = dict(
        schema='V72_PUBLIC_PALLET_INITIALIZER_RGB_ONLY_PROTOCOL_V1',
        status='FROZEN_BEFORE_ROOT_MODEL_FORWARD',
        objective='Run the existing synthetic-trained frozen YOLO26 initializer on all existing fixed public R2B RGB frames; this is source input-proposal measurement, not semantic visibility training or goal completion.',
        input_manifest=binding(inputs), number_of_RGB_frames=48,
        order='Original storage manifest 32 rows followed by original Galileo manifest 16 rows; no image-content, cue, mask, score, or GT based frame selection.',
        source_counts={'r2b_storage_v3:D455_COLOR':16, 'r2b_storage_v3:HAWK_LEFT':16, 'r2b_galileo2_v1:D455_COLOR':16},
        initializer_model=feasibility['model'], initializer_code=feasibility['code'],
        declared_source_provenance=feasibility['source_provenance'],
        provenance_limit='The prior feasibility and smoke protocol identify the existing frozen synthetic-trained checkpoint. This lock verifies file identities; it does not recreate or independently certify historical model training.',
        runtime=feasibility['runtime'],
        predictor=dict(class_name='FrozenYoloFeatures', method='predict',
            call_arguments={'already_padded':False, 'cpu_features':True},
            decoded_color='OpenCV BGR input from each original RGB PNG; no crop/recenter before predict.',
            border={'pixels':100, 'mode':'cv2.BORDER_REFLECT_101', 'applications':1},
            explicit_ultralytics_kwargs={'conf':0.001,'imgsz':640,'rect':True,'augment':False,'half':False,'verbose':False,'save':False,'stream':False},
            batching='one original frame per predict call',
            runtime_defaults='Use unchanged installed Ultralytics defaults for kwargs not specified in FrozenYoloFeatures.predict; runner records actual version/resolved args without tuning.',
            output_coordinates='All nine keypoints and box_xyxy in original native image coordinates after subtracting the added 100px border; preserve returned values, including out-of-image coordinates.',
            selection='selected_index = int(numpy.argmax(candidate_scores)); exact ties select the first returned candidate; selected_index=null if there are no returned candidates.',
            candidates='Retain every candidate returned by the unchanged Ultralytics predict call at conf=.001, in returned order. This does not mean pre-NMS raw head logits.',
            no_extra_filter=True),
        prohibited_selection=['No cue/GT-based detection choice or nearest-reference choice.', 'Do not drop zero-detection, low-confidence, out-of-image, or qualitatively poor frames.', 'No GT/cue-derived query/box, recentering, matching, threshold selection, or model selection.'],
        required_saved_output_fields=['input id/hash', 'all candidate indices', 'all candidate scores', 'all candidate native box_xyxy', 'all candidate raw9 keypoints_xy', 'all candidate raw9 keypoints_conf', 'selected_index or null', 'input_shape', 'canvas_shape', 'added_border', 'explicit no_detection status', 'runtime and checkpoint/state hashes'],
        zero_detection='Save the frame with empty candidates, null selected_index, and no_detection=true. No replacement or retry selected by cue evidence.',
        low_confidence='All returned detections are retained at the fixed .001 call floor; no second confidence cutoff defines success or eligibility.',
        optional_P3_cache={'permitted':True,'required':False,'dtype':'float16','shape':'FrozenYoloFeatures unchanged padded P3: channel_count x 80 x 80','budget_uncompressed_bytes':100000000,'rule':'Save all 48 or explicitly report unavailable/budget exception; never subset by cue/model performance. Cache contains input features only; no learned head/fit.'},
        state_invariants={'model_eval':True,'requires_grad':False,'optimizer_steps':0,'parameters_and_buffers_before_after_hash_match':True,'checkpoint_file_hash_before_after_match':True,'runner_must_record_state_digest_method_and_device':True},
        source_family={'coarse_group':'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED','split':'UNASSIGNED_GROUP_HOLD','training_only_possible_after_separate_admission':True,'independent_validation_claim':False,'all48_same_coarse_group':True},
        labels={'existing_AI_cue_exposure_known':True,'cue_rows_or_labels_in_model_inputs':False,'native8_mapping':'UNKNOWN','no_new_visibility_GT_from_initializer':True,'GT_admission':'HOLD; this proposal pass does not admit labels or train'},
        exclusions={'current223_inputs_GT_scores_features':0,'new_fit':0,'PnP_or_pose_optimization':0,'new_source_download':0,'this_input_builder_model_imports_or_forward':0},
        execution_authority='Root runs the actual frozen forward separately after binding this protocol/input lock and the runner code; this builder does not execute a model.')
    pp = write_json('PROTOCOL.json', protocol)
    review = write_json('PREFORWARD_INPUT_REVIEW.json',dict(
        schema='V72_PUBLIC_RGB_PREFORWARD_INPUT_REVIEW',PASS=True,
        frames=48,unique_file_hashes=48,all_PNG_SHA_and_IHDR_shapes_match=True,
        RGB_pixel_decode_calls=0,model_imports_forward_fit_PnP=0,current223_access=0,
        original_manifest_order_preserved=True,source_stream_counts=protocol['source_counts'],
        predictor_AST_kwargs=kwargs,predictor_lines=[predict.lineno,predict.end_lineno],
        code_behavior_review=['Frozen model is SHA-gated, eval, and requires_grad false.', 'Predict adds reflect101 100px once and subtracts it from every returned point and box.', 'Nine native camera-facing query points are deployment predictions, not ground-truth semantic native8.', 'Every returned candidate preserved; argmax first tie and no-detection null are input-only.', 'No call to branch refinement, restore_refinement, PnP, optimizer or visibility model is authorized by this protocol.'],
        limitation='File hash/IHDR metadata audit only; no new RGB visual review or empirical model-output validation. Root will bind/run the actual runner, then saved-output audit can check this contract.'))
    lock = dict(schema='V72_PUBLIC_PALLET_INITIALIZER_PREFORWARD_INPUT_LOCK_V1',
        status='INPUTS_PROTOCOL_READY_NO_FORWARD_BY_INTAKE_AGENT',
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        bindings=[binding(inputs),binding(pp),binding(review),binding(Path(__file__)),binding(feasibility_path),
                  *[binding(feasibility[k]['path']) for k in ['model','code','source_provenance']],*sources],
        family_audit=binding(BASE/'public_real_sources_20261023/r2b_galileo2_v1/CAPTURE_FAMILY_AUDIT.json'),
        additional_execution_gate='Root must bind actual runner code/runtime and this INPUT_LOCK.json before forward. No cue/GT input allowed.',
        no_forward_here=True,no_current223_read=True,no_fit_PnP=True)
    lp = write_json('INPUT_LOCK.json',lock)
    print(json.dumps({'PASS':True,'frames':48,'inputs':binding(inputs),'protocol':binding(pp),'lock':binding(lp)},ensure_ascii=False))


if __name__ == '__main__':
    main()
