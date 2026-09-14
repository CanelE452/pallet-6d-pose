"""CPU-only, immutable preflight receipts for the authorized replay comparison.

No training, evaluation forward, legacy writes, or automatic GPU/driver changes.
--out allows a fresh recovery audit without overwriting the first blocked record.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image
import torch
import ultralytics
from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8DetectionLoss
from contracts import ARM_WEIGHTS, backward_coefficients, exposure_plan

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments/pallet_transfer_replay_control_v1'
AL = ROOT / '_docs/experiments/pallet_active_learning_v1/retrospective_v1'
R0 = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
R0_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
REPLAY = ROOT / 'data/pallet/results/pallet_paper_contribution_screen_v1/C_geometry_preserving_da/dataset/BINDINGS.json'
SOURCE = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
NEG = ROOT / 'challenge/real_gt_v2/manifests/DEV_NEG2689.json'
REFERENCE = 'd0d416eb21fea1a2a932d61ab94ded98edbe638d'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == data, f'Immutable receipt differs: {path}; use --out for a recovery audit'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(data)


def local(path):
    return str(Path(path).relative_to(ROOT))


def verify_row(row):
    image, label = Path(row['image']), Path(row['label'])
    assert sha(image) == row['image_sha256'], image
    assert sha(label) == row['label_sha256'], label
    with Image.open(image) as im:
        im.verify()
    return dict(image=local(image), image_sha256=row['image_sha256'],
                label=local(label), label_sha256=row['label_sha256'])


def data_audit(out):
    old, split, selection = (read(AL / name) for name in ('PROTOCOL_LOCK.json', 'SPLIT.json', 'SELECTION_LOCK.json'))
    assert sha(R0) == R0_SHA
    for path, expected in old['source_bindings'].items():
        assert sha(ROOT / path) == expected, path
    pool, evaluation = split['pool'], split['evaluation']
    ids = selection['selections']['random']
    assert len(ids) == len(set(ids)) == 30 and len(pool) == 174 and len(evaluation) == 145
    by_id = {r['frame_id']: r for r in pool}
    selected = [by_id[fid] for fid in ids]
    assert not {r['image_sha256'] for r in pool} & {r['image_sha256'] for r in evaluation}
    assert not {r['capture_session'] for r in pool} & {r['capture_session'] for r in evaluation}
    assert len({r['capture_session'] for r in evaluation}) == 4
    target_bindings = []
    for row in [*pool, *evaluation]:
        assert sha(ROOT / row['image_path']) == row['image_sha256']
        # Hash-only preservation audit: no evaluation coordinates parsed.
        target_bindings.append(dict(frame_id=row['frame_id'], image=row['image_path'],
            image_sha256=row['image_sha256'], label=row['label_path'],
            label_sha256=sha(ROOT / row['label_path'])))
    replay_rows = read(REPLAY)['synthetic']
    assert len(replay_rows) == 1440
    with ThreadPoolExecutor(max_workers=4) as workers:
        replay = list(workers.map(verify_row, replay_rows))
    replay_hashes = {r['image_sha256'] for r in replay}
    manifest = read(SOURCE)
    assert manifest['schema'] == 'pallet_line_pose_source_v1'
    heldout = [manifest['records'][i] for i in manifest['partitions']['heldout']]
    assert all(r['partition'] == 'heldout' for r in heldout)
    eligible = [r for r in heldout if r['image_sha256'] not in replay_hashes]
    assert len({r['image_sha256'] for r in eligible}) == len(eligible)
    def sort_key(row):
        return hashlib.sha256(('replay-retention-v1\n' + row['image_sha256']).encode()).hexdigest()
    retained = sorted(eligible, key=lambda r: (sort_key(r), r['id']))[:512]
    assert retained
    with ThreadPoolExecutor(max_workers=4) as workers:
        checked = list(workers.map(verify_row, retained))
    for item, row in zip(checked, retained):
        with Image.open(ROOT / item['image']) as im:
            assert list(im.size[::-1]) == row['prepared_shape_hw']
        item.update(id=row['id'], scenario_id=row['scenario_id'], source=row['source'],
                    raw_shape_hw=row['raw_shape_hw'], prepared_shape_hw=row['prepared_shape_hw'],
                    reflect_pad_px=row['reflect_pad_px'], selection_sort_sha256=sort_key(row))
    negatives = read(NEG)['items']
    assert len(negatives) == 2689
    with ThreadPoolExecutor(max_workers=4) as workers:
        negative_hashes = list(workers.map(lambda r: sha(ROOT / r['image']), negatives))
    assert not set(negative_hashes) & {r['image_sha256'] for r in [*pool, *evaluation]}
    write(out / 'SOURCE_RETENTION_SPLIT.json', dict(status='MEMBERSHIP_LOCKED_BEFORE_NEW_RESULTS',
        manifest=local(SOURCE), manifest_sha256=sha(SOURCE), heldout_candidates=len(heldout),
        excluded_replay_file_hash_overlap=len(heldout)-len(eligible), eligible=len(eligible), actual_n=len(checked),
        ordering='SHA256(UTF8(replay-retention-v1 + newline + image_sha256)), ascending; ID tie-break',
        records=checked, groups=dict(Counter(r['scenario_id'] for r in checked)),
        scope='Source-retention development probe; historical R0 validation exposure is not excluded',
        pixels='Actual prepared images and original YOLO labels verified; use established 100px coordinate transform, never frozen P3/P4 cache or double padding',
        raw_coordinate_evaluator_adapter='NOT_IMPLEMENTED_OR_EXECUTED'))
    write(out / 'DATA_AUDIT.json', dict(status='STATIC_MEMBERSHIP_AND_FILES_PASS_NOT_LOADER_GATE',
        target_pool=174, target_train=30, target_evaluation=145, negative_evaluation=2689,
        target_evaluation_sessions=dict(Counter(r['capture_session'] for r in evaluation)),
        selected_ids=ids, target_bindings=target_bindings, synthetic_replay=replay,
        negative_manifest_sha256=sha(NEG), negative_image_sha256s=negative_hashes,
        source_retention_n=len(checked), source_replay_file_hash_overlap=0,
        pool_eval_hash_overlap=0, pool_eval_capture_session_overlap=0,
        target_negative_hash_overlap=0, training_label_export='NOT_RUN',
        other_target_label_coordinate_reads=0, evaluation_label_coordinate_reads=0,
        raw_image_integrity_checked='PIL verify for replay and selected retention; SHA for all target and negative images',
        augmentation_allowlist_and_tensor_parity='PENDING_ACTUAL_LOADER_TEST',
        GT='Existing mixed manual/geometric provenance; no new labels or independent sensor reference'))
    return old, len(checked)


def model_audit(out):
    torch.set_num_threads(4)
    checkpoint = torch.load(R0, map_location='cpu')
    model = (checkpoint.get('ema') or checkpoint['model']).float()
    model.requires_grad_(True); model.train(); model.model[-1].dfl.requires_grad_(False)
    bn = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert module.track_running_stats
            assert all(v is not None for v in (module.running_mean, module.running_var, module.num_batches_tracked))
            assert torch.isfinite(module.running_mean).all() and torch.isfinite(module.running_var).all()
            module.eval()
            assert module.weight.requires_grad and module.bias.requires_grad
            bn.append(name)
    from ultralytics.cfg import get_cfg
    model.args = get_cfg(overrides=read(AL / 'PROTOCOL_LOCK.json')['training_hyp'])
    criterion = model.init_criterion()
    assert isinstance(criterion, E2ELoss)
    assert isinstance(criterion.one2many, PoseLoss26) and isinstance(criterion.one2one, PoseLoss26)
    source = {c.__name__: inspect.getsource(c) for c in (E2ELoss, PoseLoss26, v8DetectionLoss)}
    assert 'return loss * batch_size' in source['PoseLoss26']
    assert 'target_scores_sum' in source['v8DetectionLoss']
    write(out / 'LOSS_SCALE_AUDIT.json', dict(status='SOURCE_AND_CPU_MODEL_CONFIGURATION_INSPECTED',
        torch_version=torch.__version__, ultralytics_version=ultralytics.__version__,
        criterion=type(criterion).__name__, branches=[type(criterion.one2many).__name__, type(criterion.one2one).__name__],
        loss_source_path=inspect.getfile(PoseLoss26), loss_source_sha256=sha(inspect.getfile(PoseLoss26)),
        inspected_class_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in source.items()},
        C8='sum of returned weighted branch loss vector; PoseLoss26 returns loss * batch_size',
        batch_standardization='ell=C8/8; backward=32*J; NOT a sum of independent per-image losses',
        internal_normalization='Detection cls/box/DFL use target score sum; keypoint loss uses foreground and visibility masks; visibility != 0 is supervised',
        backward_coefficients={a: backward_coefficients(a) for a in ARM_WEIGHTS},
        E2E_schedule='One criterion.update() per 30 optimizer updates / virtual epoch, 10 calls; independent of microbatch count',
        E2E_initial=dict(o2m=criterion.o2m, o2o=criterion.o2o),
        BN_modules=bn, BN_count=len(bn), BN_buffers_present_finite=True,
        BN_affine_requires_grad=True, BN_actual_gradient_and_running_buffer_parity='NOT_RUN',
        fixed_parameter_names=[n for n, p in model.named_parameters() if not p.requires_grad],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        actual_model_gradient_identities='NOT_RUN', optimizer_updates=0, forward_calls=0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=DOC)
    args = ap.parse_args(); out = args.out.resolve()
    assert out == DOC or DOC in out.parents, 'New experiment receipts only'
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'main'
    subprocess.run(['git', 'merge-base', '--is-ancestor', REFERENCE, 'HEAD'], cwd=ROOT, check=True)
    paths = [R0, AL/'PROTOCOL_LOCK.json', AL/'SPLIT.json', AL/'SELECTION_LOCK.json', REPLAY, SOURCE, NEG,
             ROOT/'scripts/research/pallet_active_learning_v1/simulation.py',
             ROOT/'scripts/research/pallet_active_learning_v1/simulation_evaluate.py',
             ROOT/'scripts/research/pallet_active_learning_v1/simulation_report.py',
             ROOT/'scripts/research/pallet_paper_contribution_screen_v1/track_c/train.py',
             ROOT/'scripts/research/pallet_line_pose_v1/source_data.py']
    paths += sorted((ROOT/'_docs/paper/final').rglob('*'))
    paths = [p for p in paths if p.is_file()]
    before = {local(p): sha(p) for p in paths}
    write(out/'SOURCE_BINDING.json', dict(reference_main=REFERENCE,
        execution_parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        originals=before, new_code={local(p): sha(p) for p in sorted(HERE.glob('*.py'))}))
    print('Source bindings recorded; checking raw datasets', flush=True)
    old, retained_n = data_audit(out)
    print('Dataset membership/file audit complete', flush=True)
    model_audit(out)
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_contracts.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    write(out/'REGRESSION_TESTS.json', dict(status='CPU_HELPERS_PASS_ACTUAL_MODEL_GATE_PENDING',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        toy_gradient_tests=True, missing_detection_fixed_PCK_denominator=True,
        actual_YOLO_gradient_tests=False, actual_loader_tensor_parity=False,
        checkpoint_roundtrip=False, smoke_optimizer_updates=0, main_optimizer_updates=0))
    write(out/'PROTOCOL_LOCK.json', dict(status='DESIGN_AND_MEMBERSHIP_LOCKED_TRAINING_GATE_NOT_READY',
        reference_main=REFERENCE, arms=ARM_WEIGHTS, seeds=[1,2,3],
        execution_order=[f'{a}_seed{s}' for s in (1,2,3) for a in ARM_WEIGHTS],
        fits=12, updates_per_fit=300, total_updates=3600, smoke_update_cap=16,
        planned_exposures={a: exposure_plan(a) for a in ARM_WEIGHTS},
        microbatch=8, common_backward_scale=32, stock_hyp=old['training_hyp'],
        optimizer='Reuse legacy bias/norm/other SGD Nesterov grouping; no auto scaling',
        schedule='Same epoch-index cosine; 30 warmup updates; clip norm10 before single optimizer.step',
        BN='Freeze only R0 running statistics, train affine; no target/negative BN calibration',
        FP32=True, AMP=False, EMA=False, checkpoint='step300 last only',
        criterion='stock E2ELoss(PoseLoss26), update once per epoch; never pseudo TRUE_IGNORE',
        RNG='Independent domain+stream datasets/buffers, workers0; crypto seed; cyclic shuffle without replacement; preserve model RNG around loader',
        parity='Same seed all arms require exact target_base image/box/keypoint/visibility tensor SHA per update',
        augmentation='Legacy HYP all arms, mosaic0 from epoch7; audit every auxiliary image against domain allowlist',
        primary='ALL_GT_PCK10, full supervised GT denominator, highest-score top1 IoU>=.5, finite error<=10px; misses score0',
        secondary=['ALL_GT_PCK5/20','AP50/AP50-95','match rate','negative AUROC/FPR95 and fixed threshold FP',
                   'all13model common-matched and separate-matched kp median/P90/frame mean/gross20',
                   'canonical 6D translation/yaw/rotation/IoU3D/ADDsym with coverage/failures','padding-only top1'],
        inference='Canonical target padding100/confidence floor.001/imgsz640 and stock PnP unchanged; raw new-weight forwards for all13 models',
        source_retention_n=retained_n,
        contrasts={'C_main':['REPLAY','T8_QUARTER'],'C_pract':['REPLAY','T8_FULL'],
                   'C_budget':['REPLAY','T32_COMPUTE'],'C_scale':['T8_QUARTER','T8_FULL']},
        uncertainty='Seed-mean statistic; common paired frame/session draw across every model; re-sum PCK numerator/denominator per draw; fixed single R0',
        bootstrap=dict(resamples=10000, seed=20260914, interval='95% percentile, marginal not familywise',
                       target_sessions=4, source_cluster='scenario_id', target_leave_one_session_out=4),
        evidence='Historical DEV145, NEG2689, source-retention development probe; no independent confirmation',
        pending=['isolated loader with Mosaic provenance','actual-model gradient/BN/save-load tests',
                 'resumable runner','new primary/source evaluator adapter','paired statistical evaluator','GPU availability'],
        automatic_training_authorized_by_this_CPU_audit=False))
    assert all(sha(ROOT / p) == h for p, h in before.items())
    write(out/'CPU_PREFLIGHT_AUDIT.json', dict(status='PARTIAL_PREFLIGHT_COMPLETE',
        preserved_bound_files=len(before), bound_sources_and_paper_final_exact=True,
        fits_completed=0, main_optimizer_updates=0, smoke_optimizer_updates=0,
        actual_model_forward_calls=0, training_gate_passed=False))
    print('CPU_PREFLIGHT_COMPLETE; NO_TRAINING; ACTUAL_MODEL_AND_LOADER_GATES_PENDING', flush=True)


if __name__ == '__main__':
    main()
