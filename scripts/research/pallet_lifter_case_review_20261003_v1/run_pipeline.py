"""Frozen offline inference preflight/run/resume; imports no hardware entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from audit import HERE, ROOT, ISO, OUT, CAPTURE, REFERENCE_COMMIT, dump, sha, modules


ASSETS = HERE / 'assets'
ASSET_NAMES = {'yolo_r0': 'yolo_r0.pt', 'n3_checkpoint': 'n3_seed1.pt'}


class CaptureAuditProxy:
    """Record prediction-only candidate identity without adding another forward."""
    def __init__(self, extractor):
        self.extractor = extractor
        self.captured = None

    def predict(self, image):
        self.captured = self.extractor.predict(image)
        return self.captured

    def close(self):
        self.extractor.close()

    def selected_object_record(self):
        c = self.captured
        selected = c['selected_index']
        chosen = None if selected is None else c['candidates'][selected]
        return {'candidate_count':len(c['candidates']),'selected_index':selected,
                'selected_box_xyxy':None if chosen is None else chosen['box_xyxy'].tolist(),
                'selected_score':None if chosen is None else chosen['score'],
                'selection_rule':'maximum model confidence, frozen, no reference input',
                'target_match_status':'not_human_reviewed'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stage_verified_assets(lr, asset_root):
    records = []
    for name, rel, digest, size in lr.LEGACY_BINDINGS:
        destination = ISO / rel
        if name in ASSET_NAMES and not destination.exists():
            supplied = asset_root / ASSET_NAMES[name]
            if supplied.exists():
                if supplied.stat().st_size != size or sha(supplied) != digest:
                    raise RuntimeError('Fixed checkpoint hash/size mismatch: ' + name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(supplied, destination)
        exists = destination.is_file()
        observed = sha(destination) if exists else None
        actual_size = destination.stat().st_size if exists else None
        good = exists and observed == digest and actual_size == size
        records.append({'name': name, 'relative_path': rel.as_posix(), 'expected_sha256': digest,
                        'expected_bytes': size, 'observed_sha256': observed, 'observed_bytes': actual_size,
                        'status': 'VERIFIED_COMPLETE' if good else 'BLOCKED_CONTRACT'})
    return records


def preflight(asset_root):
    lf, lr = modules()
    records = stage_verified_assets(lr, asset_root)
    missing = [r['name'] for r in records if r['status'] != 'VERIFIED_COMPLETE']
    plan = read(OUT/'LIFTER_EVALUATION_PLAN.json')
    inputmap = read(OUT/'LIFTER_INPUT_AND_TIME_MAP.json')
    plan_hash = sha(OUT/'LIFTER_EVALUATION_PLAN.json')
    if plan['stored_frame_count'] != 8910 or len(plan['frames']) != 8910:
        raise RuntimeError('Full denominator changed')
    for s in inputmap['sessions']:
        sid = s['session_id']
        for f in s['files']:
            p = CAPTURE/f['filename']
            if sha(p) != f['sha256'] or p.stat().st_size != f['bytes']:
                raise RuntimeError('Original capture changed: ' + sid + '/' + f['role'])
    if not missing:
        lr.verify_legacy_bindings()  # original guard remains unchanged
        lr.fixed_selection_contract()
    record = {'schema_version': 'lifter_inference_preflight_v1', 'status': 'BLOCKED_CONTRACT' if missing else 'READY_TO_REVIEW',
              'source_commit': REFERENCE_COMMIT, 'plan_sha256': plan_hash, 'bindings': records,
              'missing_bindings': missing, 'full_stored_frame_count': 8910,
              'requested_methods': ['YOLO_R0','YOLO_R0_N3_seed1'], 'optimizer_updates':0,
              'training_runs':0, 'independent_accuracy_claimed':False}
    dump(OUT/'INFERENCE_PREFLIGHT.json', record)
    return record, lf, lr, plan


def cache_equivalence(lr, lf, plan, stack, asset_root):
    receipt = read(ISO/'_docs/experiments/pallet_n3_completion_v3/LIFTER_YOLO_R0_N3_SEED1_COMPLETE.json')
    paths = {'raw': asset_root/'legacy_raw.json', 'plan': asset_root/'legacy_plan.json'}
    source = {}
    for key,p in paths.items():
        b = receipt[key]
        if not p.exists():
            return {}, {'status':'BLOCKED_CONTRACT','ran':False,'cache_reused_frames':0,
                        'reason':'Hash-bound legacy raw/plan unavailable; all frames will be fresh computation'}
        if sha(p) != b['sha256'] or p.stat().st_size != b['bytes']:
            raise RuntimeError('Legacy cache input mismatch: ' + key)
        source[key] = read(p)
    for sid in lf.SESSION_IDS:
        for role,p in lf.session_paths(CAPTURE,sid).items():
            size,digest = lf.EXPECTED_FILES[sid][role]
            if not p.exists() or p.stat().st_size != size or sha(p) != digest:
                return {}, {'status':'BLOCKED_CONTRACT','ran':False,'cache_reused_frames':0,
                            'reason':'Legacy capture binding changed, including separately excluded fifth session; preserve old cache, fresh full pass'}
    os.environ['PALLET_LIFTER_CAPTURE_ROOT'] = str(CAPTURE)
    lr.validate_plan(source['plan'],verify_files=True,capture_root=CAPTURE)
    lr.validate_raw(source['raw'],source['plan'],verify_files=True)
    cached = {(sid,int(f['video_index'])):f for sid,session in source['raw']['sessions'].items() for f in session['frames']}
    checks = []
    all_equal = True
    pixel_proof = True
    for sid in lf.USABLE_SESSION_IDS:
        old_indices = sorted(index for session,index in cached if session == sid)
        # Five deterministic chronological cache positions; no output-based choice.
        chosen = {old_indices[int(round(i*(len(old_indices)-1)/4))] for i in range(5)}
        camera = lr.camera_contract(read(lf.session_paths(CAPTURE,sid)['meta']))
        for frame in lf.iter_offline_frames(sid,CAPTURE,verify=True):
            if frame.video_index not in chosen:
                continue
            candidate = lr.infer_shared_frame(frame.image_bgr,frame.metadata(),camera,stack)
            previous = cached[(sid,frame.video_index)]
            digest = hashlib.sha256(frame.image_bgr.tobytes()).hexdigest()
            pixel_match = previous.get('decoded_bgr_sha256') == digest
            pixel_proof &= pixel_match
            equal = candidate['methods'] == previous['methods'] and candidate['n3'] == previous['n3']
            all_equal &= equal
            checks.append({'session_id':sid,'saved_frame_index':frame.video_index,'outputs_exactly_equal':equal,
                           'cache_has_matching_decoded_pixel_hash':pixel_match,'decoded_bgr_sha256':digest})
    reusable = all_equal and pixel_proof and len(checks)==20
    # Old raw schema did not require decoded pixel hashes. Output equality alone is not pixel evidence.
    return (cached if reusable else {}), {'status':'VERIFIED_COMPLETE' if reusable else 'BLOCKED_CONTRACT',
        'ran':True,'checks':checks,'outputs_equal':all_equal,'decoded_pixel_proof':pixel_proof,
        'reason':None if reusable else 'Exact output or decoded-pixel contract not established; preserve legacy cache and compute new pass',
        'cache_reused_frames':len(cached) if reusable else 0}


def normalize(frame, pred, ordinal, source):
    methods = {}
    for old,new in [('R0','Base'),('N3_seed1','N3')]:
        m = pred['methods'][old]
        pts = m['points_xy']
        mask = [] if pts is None else [all(v is not None and math.isfinite(v) for v in p) for p in pts[:8]]
        methods[new] = {'pose_state':'fresh' if m['fresh'] else 'no_pose','detection_present':m['detected'],
                        'keypoints_mask':mask,'pnp_failed':m['detected'] and not m['available'],
                        'object_id':None,'object_match':None, 'selected_index':m['selected_index'],
                        'selected_object':pred.get('selected_object_audit'),
                        'keypoints':None if pts is None else pts[:8],
                        'pose':{'x_m':m['pos_x_m'],'z_m':m['pos_z_m'],'yaw_deg':m['yaw_deg']}}
    return {'frame_id':frame['frame_id'],'session_id':frame['session_id'],'stored_index':frame['saved_frame_index'],
            'sensor_timestamp_ms':frame['camera_sensor_timestamp_ms'],'camera_frame_number':frame['camera_frame_number'],
            'decoded_bgr_sha256':frame['decoded_bgr_sha256'],
            'inference_execution_id':('full_v1:'+str(ordinal) if source=='new_frozen_inference' else 'legacy-cache:'+frame['frame_id']),
            'inference_execution_id_source':('current_execution' if source=='new_frozen_inference' else 'derived_cache_record_identity'),
            'new_inference_executed':source=='new_frozen_inference',
            'compute_source':source,'methods':methods,'raw_shared_prediction':pred}


def run(asset_root):
    started = time.perf_counter()
    status,lf,lr,plan = preflight(asset_root)
    if status['status'] == 'BLOCKED_CONTRACT':
        print(json.dumps({'status':'BLOCKED_CONTRACT','missing_bindings':status['missing_bindings'],
                          'inferred_frames':0,'gpu_inference_seconds':0},ensure_ascii=False))
        return 2
    path = OUT/'raw_predictions/ALL_STORED_FRAMES.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    identity = {'plan_sha256':sha(OUT/'LIFTER_EVALUATION_PLAN.json'),'source_commit':REFERENCE_COMMIT,
                'fixed_bindings':lr.public_legacy_bindings(),'pass_budget_s':3600,'independent_reference':False}
    dump(OUT/'raw_predictions/RUN_IDENTITY.json',identity,frozen=True)
    existing = []
    if path.exists():
        with path.open(encoding='utf-8') as f:
            existing = [json.loads(line) for line in f]
        for i,row in enumerate(existing):
            frame = plan['frames'][i]
            expected = {'frame_id':frame['frame_id'],'session_id':frame['session_id'],
                        'stored_index':frame['saved_frame_index'],'sensor_timestamp_ms':frame['camera_sensor_timestamp_ms'],
                        'camera_frame_number':frame['camera_frame_number'],'decoded_bgr_sha256':frame['decoded_bgr_sha256']}
            if any(row.get(k)!=v for k,v in expected.items()) or set(row.get('methods',{}))!={'Base','N3'}:
                raise RuntimeError('Resume raw prefix identity changed')
    if len(existing)>8910:
        raise RuntimeError('Raw prefix longer than frozen full plan')
    if existing and not (OUT/'INFERENCE_RUN_STATUS.json').exists():
        raise RuntimeError('Partial predictions have no original budget ledger; refuse budget reset')
    previous_cost = read(OUT/'INFERENCE_RUN_STATUS.json').get('gpu_inference_seconds',0) if (OUT/'INFERENCE_RUN_STATUS.json').exists() else 0
    if len(existing)==8910:
        previous_status=read(OUT/'INFERENCE_RUN_STATUS.json')
        previous_status.update({'status':'VERIFIED_COMPLETE','inferred_frames':8910,'newly_computed_frames':0,
                                'resume_reused_complete_pass':True})
        dump(OUT/'INFERENCE_RUN_STATUS.json',previous_status)
        print('VERIFIED_COMPLETE existing full pass verified; 0 new model forwards')
        return 0
    if previous_cost >= 3600:
        print('BLOCKED_CONTRACT original 60 minute budget exhausted; partial results retained')
        return 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if gpu:
        dump(OUT/'GPU_WAIT_STATUS.json',{'status':'BLOCKED_CONTRACT','reason':'foreign GPU compute process; preserved',
                                       'timestamp':datetime.now(timezone.utc).isoformat()})
        print('BLOCKED_CONTRACT foreign GPU compute process preserved')
        return 2
    lock = OUT/'GPU_0.lock'
    try:
        with lock.open('x',encoding='utf-8') as f:
            json.dump({'pid':os.getpid(),'role':'CLI_A','device':0,'created_at':datetime.now(timezone.utc).isoformat()},f)
    except FileExistsError:
        print('BLOCKED_CONTRACT GPU lock exists; inspect owner, never terminate another process')
        return 2
    stack = None
    newly = 0
    compute = 0
    try:
        cv2.setNumThreads(1)
        import torch
        torch.set_num_threads(1)
        deadline_path=OUT/'raw_predictions/BUDGET_DEADLINE.json'
        if deadline_path.exists():
            deadline=read(deadline_path)['deadline_unix_s']
        else:
            deadline=time.time()+3600
            dump(deadline_path,{'schema_version':'lifter_gpu_budget_deadline_v1','deadline_unix_s':deadline,
                               'budget_s':3600,'policy':'conservative wall-clock deadline across partial resumes; never reset'},frozen=True)
        if time.time()>=deadline:
            raise TimeoutError('Original single-pass wall deadline exhausted; partial results retained')
        gpu_start=time.perf_counter()
        dump(OUT/'INFERENCE_RUN_STATUS.json',{'status':'RUNNING','inferred_frames':len(existing),
                                            'gpu_inference_seconds':previous_cost,'deadline_unix_s':deadline})
        stack = lr.load_legacy_stack('cuda')
        stack.extractor = CaptureAuditProxy(stack.extractor)
        if existing:
            cache = {}
        else:
            cache,check = cache_equivalence(lr,lf,plan,stack,asset_root)
            dump(OUT/'CACHE_EQUIVALENCE_20.json',check)
        ordinal = 0
        with path.open('a',encoding='utf-8',newline='\n') as out:
            for sid in lf.USABLE_SESSION_IDS:
                camera = lr.camera_contract(read(lf.session_paths(CAPTURE,sid)['meta']))
                for f in lf.iter_offline_frames(sid,CAPTURE,verify=True):
                    frame = plan['frames'][ordinal]
                    if f.session_id != frame['session_id'] or f.video_index != frame['saved_frame_index']:
                        raise RuntimeError('Frame order changed')
                    digest = hashlib.sha256(f.image_bgr.tobytes()).hexdigest()
                    if digest != frame['decoded_bgr_sha256']:
                        raise RuntimeError('Decoded pixels changed; stop reuse/inference')
                    if ordinal < len(existing):
                        ordinal += 1
                        continue
                    if time.time()>=deadline or previous_cost + time.perf_counter()-gpu_start >= 3600:
                        raise TimeoutError('Original single-pass 60 minute budget reached')
                    key = (sid,f.video_index)
                    if key in cache:
                        raw = cache[key]
                        src = 'verified_legacy_cache'
                    else:
                        raw = lr.infer_shared_frame(f.image_bgr,f.metadata(),camera,stack)
                        raw['selected_object_audit'] = stack.extractor.selected_object_record()
                        compute += 1
                        src = 'new_frozen_inference'
                    move = raw['n3']['movement']['max_move_px']
                    if move is not None and move > 0.01*math.hypot(480,640) + 1e-5:
                        raise RuntimeError('Original-image 1 percent movement cap violated')
                    if not raw['base_forward_shared']:
                        raise RuntimeError('Base forward must be shared')
                    row = normalize(frame,raw,ordinal,src)
                    out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    out.flush()
                    ordinal += 1
                    newly += 1
                    if ordinal % 250 == 0:
                        print('FULL_PASS',ordinal,'/',8910,flush=True)
        status_value = 'VERIFIED_COMPLETE' if ordinal == 8910 else 'BLOCKED_CONTRACT'
    except Exception as exc:
        status_value = 'BLOCKED_CONTRACT'
        dump(OUT/'INFERENCE_FAILURE.json',{'status':status_value,'error_type':type(exc).__name__,'reason':str(exc),
                                         'failed_frame_id':locals().get('frame',{}).get('frame_id'),
                                         'partial_raw_retained':True})
        raise
    finally:
        elapsed = previous_cost + (time.perf_counter()-gpu_start if 'gpu_start' in locals() else 0)
        dump(OUT/'INFERENCE_RUN_STATUS.json',{'status':locals().get('status_value','BLOCKED_CONTRACT'),
            'inferred_frames':len(existing)+newly,'newly_computed_frames':compute,
            'gpu_inference_seconds':elapsed,'invocation_wall_seconds':time.perf_counter()-started,
            'optimizer_updates':0,'training_runs':0,'full_pass_limit':1,'gpu_pass_budget_s':3600})
        if stack is not None:
            stack.close()
        lock.unlink()
    return 0 if status_value == 'VERIFIED_COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','run'])
    parser.add_argument('--asset-root',type=Path,default=ASSETS)
    args = parser.parse_args()
    if args.action == 'run':
        return run(args.asset_root)
    status,_,_,_ = preflight(args.asset_root)
    print(json.dumps({'status':status['status'],'missing_bindings':status['missing_bindings']},ensure_ascii=False))
    return 0 if status['status'] == 'READY_TO_REVIEW' else 2


if __name__ == '__main__':
    raise SystemExit(main())
