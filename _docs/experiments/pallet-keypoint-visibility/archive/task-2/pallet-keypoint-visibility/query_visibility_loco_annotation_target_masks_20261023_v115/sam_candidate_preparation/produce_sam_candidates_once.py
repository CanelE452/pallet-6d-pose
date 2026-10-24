"""V115 one-shot SAM candidates. Import is inert; root entry precedes payload/model I/O."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from candidate_contract import (require, validate_entry_structure, validate_cohort,
                                single_native_mask, query_sample, mask_diagnostics, pair_diagnostics,
                                output_request)

R = Path(__file__).resolve().parent
V = R.parent
P = V.parent
RT = P / 'foundation_runtime_sam21_20261023'
REPO = RT / 'sam2_repo'
PYTHON = RT / 'venv/bin/python'
OUT = V / 'source_sam_candidates_v1'
CONFIG = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
CHECKPOINT = RT / 'checkpoints/sam2.1_hiera_base_plus.pt'
CHECKPOINT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
V81 = P / 'query_visibility_grounded_target_20261023_v81/segmentation'
STATIC = {
    'code': Path(__file__).resolve(), 'contract': R / 'candidate_contract.py',
    'generated': R / 'GENERATED_CANDIDATE_TESTS.json', 'protocol': R / 'PRODUCER_PROTOCOL.json',
    'root_policy': V / 'ROOT_SAM_CANDIDATE_POLICY_BEFORE_PIXELS.json',
    'metadata_receipt': V / 'METADATA_PREPARATION_RECEIPT.json',
    'metadata_seal': V / 'METADATA_PREPARATION_SEAL.json',
    'request_plan': V / 'SAM_REQUEST_PLAN40_METADATA_ONLY.jsonl',
    'frame_inputs': V / 'FRAME20_ORIGINAL_QUERY_INPUTS_ONLY.jsonl',
    'query_inputs': V / 'QUERY160_ORIGINAL_INPUTS_ONLY.jsonl',
    'target_entities': V / 'TARGET20_EXISTING_ANNOTATION_ENTITIES_GT_ONLY.jsonl',
    'witnesses': V / 'EXPLICIT_SAME_ANN_ID_WEAK_VISIBLE_WITNESSES_GT_ONLY.jsonl',
    'runtime': RT / 'RUNTIME_SEAL.json', 'checkpoint': CHECKPOINT,
    'sam_build': REPO / 'sam2/build_sam.py',
    'sam_predictor': REPO / 'sam2/sam2_image_predictor.py',
    'sam_transforms': REPO / 'sam2/utils/transforms.py', 'sam_config': REPO / 'sam2' / CONFIG,
    'V81_API_precedent_code': V81 / 'run_sam_from_saved_associations.py',
    'V81_API_precedent_contract': V81 / 'sam_stage_contract.py',
    'V81_official_API_receipt': V81 / 'OFFICIAL_SAM_MASK_API_CONTRACT.json',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def write(path, obj):
    with open(path, 'x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def write_rows(path, rr):
    with open(path, 'x') as f:
        for row in rr:
            f.write(json.dumps(row, allow_nan=False) + '\n')


def entry_guard(entry_path):
    """Read entry then metadata only. No source array or checkpoint bytes here."""
    e = read(entry_path)
    validate_entry_structure(e, STATIC, OUT, sys.executable, PYTHON)
    require(not OUT.exists(), 'Output must be absent; preserve failures and never silently restart')
    for name, path in STATIC.items():
        if name != 'checkpoint':
            require(sha(path) == e['bindings'][name]['sha256'], 'Changed metadata/code binding: ' + name)
    require(e['bindings']['checkpoint']['sha256'] == CHECKPOINT_SHA, 'Exact official checkpoint identity')
    protocol = read(STATIC['protocol'])
    require(protocol['completed'] is True and protocol['root_policy'] == e['bindings']['root_policy'], 'Frozen prospective policy')
    for name, expected in protocol['bindings'].items():
        require(e['bindings'][name] == expected, 'Protocol/entry mismatch: ' + name)
    proof = read(STATIC['generated'])
    require(proof['PASS'] is True and proof['actual_source_payload_reads'] == 0 and proof['model_calls'] == 0, 'Generated-only PASS')
    require(proof['bindings']['runner'] == e['bindings']['code']
            and proof['bindings']['contract'] == e['bindings']['contract'], 'Proof binds exact code')
    rt = read(STATIC['runtime'])
    require(rt['READY'] is True and rt['repo_revision'] == REVISION and rt['config'] == CONFIG, 'Official frozen runtime')
    require(rt['checkpoint']['path'] == str(CHECKPOINT)
            and rt['checkpoint']['sha256'] == CHECKPOINT_SHA, 'Runtime checkpoint lineage')
    api = read(STATIC['V81_official_API_receipt'])
    require(api['predictor'] == e['bindings']['sam_predictor'] and api['return_logits'] is False
            and api['multimask_output'] is False, 'V81 official single binary F32 API precedent')
    return e


def saved_metadata(e):
    frames = lines(STATIC['frame_inputs']); slots = lines(STATIC['query_inputs'])
    requests = lines(STATIC['request_plan']); entities = lines(STATIC['target_entities'])
    factual = validate_cohort(frames, slots, requests, entities)
    return frames, slots, requests, entities, factual


def state_digest(model):
    h = hashlib.sha256()
    for key, t in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(entry_path):
    e = entry_guard(entry_path)
    frames, slots, requests, entities, factual = saved_metadata(e)
    OUT.mkdir(); (OUT / 'masks').mkdir(); (OUT / 'repeat_diagnostics').mkdir()
    write(OUT / 'RUN_STARTED.json', dict(entry=bind(entry_path), source_only=True,
          expected_primary_masks=33, expected_repeat_calls=1, current223=False, V112=False, fit=False, PnP=0))
    start = time.monotonic()
    primary_calls = repeat_calls = set_image_calls = 0
    try:
        # Only a valid explicit root run reaches checkpoint bytes/imports/source arrays.
        require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'Official checkpoint byte SHA')
        import torch
        import sam2.build_sam as builder
        import sam2.sam2_image_predictor as predictor_module
        require(torch.__version__ == '2.5.1+cu121' and np.__version__ == '1.26.4', 'Pinned Torch/NumPy')
        require(torch.cuda.is_available(), 'CUDA required')
        require(Path(builder.__file__).resolve() == STATIC['sam_build']
                and Path(predictor_module.__file__).resolve() == STATIC['sam_predictor'], 'Official import paths')
        torch.manual_seed(20261023); np.random.seed(20261023); torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        def no_compile(*args, **kwargs):
            raise RuntimeError('torch.compile forbidden')
        torch.compile = no_compile
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = builder.build_sam2(CONFIG, str(CHECKPOINT), device='cuda', mode='eval',
                apply_postprocessing=False, hydra_overrides_extra=['++model.compile_image_encoder=false'])
        model.requires_grad_(False)
        require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Frozen eval')
        require({str(p.dtype) for p in model.parameters()} == {'torch.float32'}, 'FP32 weights')
        predictor = predictor_module.SAM2ImagePredictor(model, mask_threshold=0., max_hole_area=0., max_sprinkle_area=0.)
        before_state = state_digest(model)
        output_rows = []; query_rows = []; frame_rows = []; pair_rows = []; repeat_receipt = None
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i, f in enumerate(frames):
                rgb_binding = f['decoded_RGB_input']
                require(sha(rgb_binding['path']) == rgb_binding['sha256'], 'Canonical NPY file changed')
                image = np.load(rgb_binding['path'], allow_pickle=False)
                require(image.dtype == np.uint8 and image.shape == tuple(f['raw_hw']) + (3,), 'Native RGB uint8 HWC')
                image = np.array(image, dtype=np.uint8, order='C', copy=True)
                require(hashlib.sha256(image.tobytes()).hexdigest() == f['RGB_pixels_sha256'], 'Exact initializer RGB pixel SHA')
                t0 = time.monotonic(); predictor.set_image(image); set_image_calls += 1
                frame_rows.append(dict(frame_index=i, input_id=f['input_id'], split=f['split'],
                    canonical_decoded_RGB=rgb_binding, verified_RGB_pixel_sha256=f['RGB_pixels_sha256'],
                    original_JPEG_sha256=f['image_sha256'], raw_hw=f['raw_hw'],
                    set_image_call_index=set_image_calls-1, set_image_seconds=time.monotonic()-t0))
                masks_for_frame = {}
                for req in requests[2*i:2*i+2]:
                    mask = quality = mb = diag = None
                    if req['call_eligible_metadata']:
                        box = np.asarray(req['box_xyxy_raw'], dtype=np.float32)
                        t0 = time.monotonic()
                        masks, qualities, _ = predictor.predict(point_coords=None, point_labels=None,
                            box=box, multimask_output=False, return_logits=False)
                        primary_calls += 1
                        mask, quality = single_native_mask(masks, qualities, f['raw_hw'])
                        mp = OUT / 'masks' / ('%02d_%s_ann%d.npz' % (i, req['role'], req['ann_id']))
                        np.savez_compressed(mp, visible_mask=mask); mb = bind(mp)
                        diag = dict(mask_diagnostics(mask, req['box_xyxy_raw']),
                                    mask_payload_bytes=mp.stat().st_size, predict_seconds=time.monotonic()-t0)
                        # First TARGET is predetermined by cohort order, never replaced by repeat.
                        if repeat_calls == 0 and req['role'] == 'TARGET':
                            rep_masks, rep_quality, _ = predictor.predict(point_coords=None, point_labels=None,
                                box=box, multimask_output=False, return_logits=False)
                            repeat_calls += 1
                            rep, rq = single_native_mask(rep_masks, rep_quality, f['raw_hw'])
                            rp = OUT / 'repeat_diagnostics/FIRST_TARGET_REPEAT.npz'
                            np.savez_compressed(rp, visible_mask=rep)
                            repeat_receipt = dict(request_id=req['request_id'], primary_mask=mb,
                                repeat_mask=bind(rp), primary_quality=quality, repeat_quality=rq,
                                mask_bitexact=bool(np.array_equal(mask, rep)), quality_abs_difference=abs(quality-rq),
                                same_embedding=True, replaces_primary=False)
                            write(OUT / 'repeat_diagnostics/REPEAT_CHECK.json', repeat_receipt)
                            require(repeat_receipt['mask_bitexact'] and abs(quality-rq) <= 1e-6, 'Fixed first-target repeat failed')
                    masks_for_frame[req['role']] = mask
                    output_rows.append(output_request(req, mb, quality, diag,
                        None if mask is None else not bool(mask.any()), e['bindings']['root_policy'], bind(entry_path)))
                target, competitor = masks_for_frame['TARGET'], masks_for_frame['COMPETITOR']
                require(target is not None, 'All20 preexisting target requests retained')
                pair_rows.append(dict(frame_index=i, input_id=f['input_id'],
                    **pair_diagnostics(target, competitor), initializer_target_equivalence='UNKNOWN'))
                for s in slots[8*i:8*i+8]:
                    ts = query_sample(target, s, f['raw_hw']); cs = query_sample(competitor, s, f['raw_hw'])
                    query_rows.append(dict(s, target_candidate=ts, competitor_candidate=cs,
                        both_candidate_masks_true_at_q=(None if ts['mask_value'] is None or cs['mask_value'] is None
                                                        else ts['mask_value'] and cs['mask_value']),
                        semantic_D_E_S_created=False, native_expected_state='UNKNOWN', human_gold=False,
                        training_admission=False))
                print(json.dumps(dict(processed_frames=i+1, primary_calls=primary_calls,
                                      repeat_calls=repeat_calls, elapsed_seconds=time.monotonic()-start)), flush=True)
        after_state = state_digest(model)
        require(before_state == after_state, 'Model state changed')
        require(set_image_calls == 20 and primary_calls == 33 and repeat_calls == 1, 'Exact 20/33+1 calls')
        require(len(output_rows) == 40 and len(query_rows) == 160 and len(frame_rows) == 20, 'Complete output retention')
        require(all(all(r[k] == s[k] for k in s) for r, s in zip(query_rows, slots)), 'Original q/L/folds/status unchanged')
        for name, obj in [('OUTPUT_REQUESTS40.jsonl', output_rows), ('QUERY_READOUTS160.jsonl', query_rows),
                          ('DECODED_FRAME_LINEAGE20.jsonl', frame_rows), ('TARGET_COMPETITOR_DIAGNOSTICS20.jsonl', pair_rows)]:
            write_rows(OUT / name, obj)
        summary = dict(completed=True, PASS=True, frames=20, request_slots=40, primary_masks=33,
            missing_competitor_no_calls=7, repeat_predict_calls=1, total_predict_calls=34, set_image_calls=20,
            original_query_slots=160, original_factual_counts=factual, model_state_before=before_state,
            model_state_after=after_state, model_unchanged=True, first_target_repeat=repeat_receipt,
            empty_primary_masks=sum(r['empty_mask'] is True for r in output_rows),
            weights='FP32', autocast='CUDA BF16', TF32=False, compile=False, postprocessing=False,
            mask_threshold=0., multimask_output=False, return_logits=False, point_prompts=False,
            quality_selection=False, nearest_readout='numpy.rint_ties_to_even; original factual validity preserved',
            seconds=time.monotonic()-start, GPU_peak_bytes=torch.cuda.max_memory_allocated(),
            native_mapping='UNKNOWN', human_gold=False, semantic_D_E_S_created=False,
            source_only=True, training_admission=False, current223=False, V112=False, fit=False, PnP=0)
        write(OUT / 'SUMMARY.json', summary)
        paths = [OUT / n for n in ('OUTPUT_REQUESTS40.jsonl', 'QUERY_READOUTS160.jsonl',
                 'DECODED_FRAME_LINEAGE20.jsonl', 'TARGET_COMPETITOR_DIAGNOSTICS20.jsonl', 'SUMMARY.json',
                 'RUN_STARTED.json', 'repeat_diagnostics/FIRST_TARGET_REPEAT.npz', 'repeat_diagnostics/REPEAT_CHECK.json')]
        write(OUT / 'SAM_CANDIDATE_OUTPUT_SEAL.json', dict(completed=True, PASS=True,
            root_entry=bind(entry_path), bindings=e['bindings'], output_requests=bind(OUT / 'OUTPUT_REQUESTS40.jsonl'),
            query_readouts=bind(OUT / 'QUERY_READOUTS160.jsonl'), summary=bind(OUT / 'SUMMARY.json'),
            outputs=[bind(p) for p in paths]+[r['mask'] for r in output_rows if r['mask'] is not None],
            source_only=True, training_admission=False, current223=False, V112=False, fit=False, PnP=0))
    except BaseException as ex:
        write(OUT / 'FAILED.json', dict(type=type(ex).__name__, error=str(ex), traceback=traceback.format_exc(),
              primary_calls=primary_calls, repeat_calls=repeat_calls, set_image_calls=set_image_calls,
              no_automatic_retry=True, source_only=True, current223=False, V112=False))
        raise


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['validate', 'run']); ap.add_argument('--entry', required=True)
    args = ap.parse_args()
    if args.mode == 'validate':
        e = entry_guard(args.entry); frames, slots, requests, _, factual = saved_metadata(e)
        print(json.dumps(dict(PASS=True, frames=len(frames), slots=len(slots), requests=len(requests),
                             factual=factual, source_payload_reads=0, checkpoint_byte_reads=0, model_imports=0)))
    else:
        run(args.entry)


if __name__ == '__main__':
    main()
