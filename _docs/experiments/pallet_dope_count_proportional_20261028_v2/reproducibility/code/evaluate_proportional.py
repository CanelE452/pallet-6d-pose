"""Seed-aware evaluation of the count-proportional R2 study.

This file imports no neural runtime and opens no input on import. Evaluation
uses the frozen original metric and decoder sources, exact primary372 caches,
and only the separately authorized WOOD80 RGB exception. No raw reference
annotation is opened. Training-seed SD and conditional point-error SD are
different statistics and are deliberately stored under different names.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
KINDS = ('PLASTIC', 'WOOD', 'GREEN')
CONDITIONS = ('A', 'B', 'C', 'D', 'E')
TRAINING_SEEDS = (42, 43, 44)
QUALITY = ('PCK5', 'PCK10', 'PCK20', 'coverage', 'conditional_precision10')
BOOTSTRAP_SEED, REPLICATES = 20261026, 10000
EXPECTED = {'PLASTIC': (128, 985), 'WOOD': (125, 963), 'GREEN': (119, 943)}
REAL_COUNTS = {'PLASTIC': 20, 'WOOD': 550, 'GREEN': 550}
SYN_COUNTS = {'PLASTIC': (4, 10, 20, 40, 80), 'WOOD': (110, 275, 550, 1100, 2200),
              'GREEN': (110, 275, 550, 1100, 2200)}
BASE = Path('/home/minjae/Documents/pallet-research')
EXPECTED_ROOT = BASE / 'dope-r2-count-proportional-per-type-20261028-v2'
OLD_COMMON = BASE / 'dope-r2-ratio-per-type-20261027-v1/code/ratio_eval_common.py'
OLD_COMMON_SHA = '3f16bcfe9dba1510c81984bb54113b18b2c43728158cf543465d106c03a495ad'
GPU_LOCK = BASE / 'dope-thesis-delivery-20261029-v1/GPU_QUEUE.lock'
NOTE = ('Count-proportional image sampling; original-view primary372 evaluation; '
        'PLASTIC128/985, WOOD125/963, GREEN119/943. Missing predictions fail PCK. '
        'Observed native-pixel error SD is population SD (ddof=0). Training-seed '
        'metric SD is sample SD (ddof=1), seeds42/43/44; three R2 '
        'sampling/augmentation seeds conditional on the same original R1 '
        'weights, Adam state, and fixed pseudo-label/subset sets, not three '
        'complete self-training pipeline repetitions. '
        'Frame bootstrap uses10000 draws, seed20261026, percentile95, no adjustment. '
        'Seed-aggregate frame CIs condition on the three trained seeds; they do '
        'not quantify uncertainty from new training seeds. Shared synthetic '
        'control has a different R1 history and is contextual. GREEN manual600 '
        'is separate. WOOD125 mixes native resolutions and added80 shares '
        'recordings with adaptation data; it is not an independent-recording test.')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean(value):
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def stable(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frozen_common():
    require(sha(OLD_COMMON) == OLD_COMMON_SHA, 'Frozen prior evaluator changed')
    name = 'proportional_frozen_metric_context'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, OLD_COMMON)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(self.root == EXPECTED_ROOT, 'Use the designated new experiment root')
        self.bindings = {}
        self.forward_runtime = False
        self.raw_rgb_reads = 0
        self.authorized80_reads = 0
        allow_path = self.root / 'preflight/EVAL_ALLOWLIST.json'
        require(allow_path.is_file() and not allow_path.is_symlink(), 'Exact evaluation allowlist missing')
        self.eval_allow = json.loads(allow_path.read_text())
        self.bindings[str(allow_path)] = sha(allow_path)
        self.allow = self.eval_allow['input_files']
        self.protected = set(self.eval_allow.get('protected_image_sha256', []))
        self.approved80 = self.eval_allow['approved_wood80_frames']
        self.approved80_paths = {r['image_path']: r['image_sha256'] for r in self.approved80.values()}
        require(len(self.approved80) == len(self.approved80_paths) == 80, 'Exact WOOD80 exception required')
        for path, item in self.allow.items():
            require(str(Path(path).resolve()) == path and isinstance(item.get('category'), str),
                    'Malformed exact evaluation allowlist')
        for name in ('INPUTS.json', 'FIXED_PLAN.json'):
            path = self.root / ('preflight/' + name) if name == 'INPUTS.json' else self.root / name
            require(path.is_file(), 'Root preparation missing: ' + str(path))
            self.bindings[str(path)] = sha(path)
        # Apply the root seal before any permitted reference cache is read.
        # This gate also applies to CPU-only score/report invocations.
        self.go = verify_go(self)
        self.authorization80_verified = False
        proof = self.eval_allow['wood80_authorization']
        self.hash(proof['authorization_path'], proof['authorization_sha256'])
        manifest = self.read(proof['manifest_path'])
        self.hash(proof['manifest_path'], proof['manifest_sha256'])
        require(len(manifest['records']) == 80 and
            {r['id'] for r in manifest['records']} == set(self.approved80), 'WOOD80 authorization ID mismatch')
        for row in manifest['records']:
            declared = self.approved80[row['id']]
            require(all(row[k] == declared[k] for k in
                ('image_path', 'image_sha256', 'width', 'height', 'object_type')),
                'WOOD80 authorization metadata differs')
            require((row['width'], row['height']) == (640, 480) and
                row['object_type'] == 'wood_small_80x59x14', 'Only the authorized WOOD640 exception is valid')
        self.authorization80_verified = True
        self.bindings[str(OLD_COMMON)] = OLD_COMMON_SHA
        frozen_common()

    def path(self, path):
        raw = Path(path)
        require(not raw.is_symlink(), 'Input/output symlink forbidden')
        path = raw.resolve()
        require('PLASTIC184' not in path.name.upper(), 'PLASTIC184 is outside this study')
        if not path.is_relative_to(self.root):
            require(str(path) in self.allow, 'External input not exactly allowlisted: ' + str(path))
            require('/final/' not in str(path).lower() or str(path) in self.approved80_paths,
                    'Unapproved protected FINAL path')
        return path

    def hash(self, path, expected=None, raw_dev=False):
        path = self.path(path)
        if not path.is_relative_to(self.root):
            item = self.allow[str(path)]
            if str(path) in self.approved80_paths:
                require(self.authorization80_verified and self.forward_runtime and raw_dev,
                        'WOOD80 RGB requires exact authorization and active inference')
            else:
                require(item['category'] in ('metadata', 'protocol', 'checkpoint_receipt',
                    'source_code', 'checkpoint', 'cached_scores', 'cached_predictions',
                    'dev_two_view_predictions', 'dev_rgb'), 'Forbidden input category')
                require(item['category'] != 'dev_rgb' or raw_dev and self.forward_runtime,
                        'DEV RGB requires inference')
        digest = sha(path)
        if not path.is_relative_to(self.root):
            require(digest == self.allow[str(path)]['sha256'], 'Frozen external input changed: ' + str(path))
            if digest in self.protected:
                require(str(path) in self.approved80_paths and self.authorization80_verified,
                        'Protected hash outside the exact WOOD80 exception')
        if str(path) in self.approved80_paths:
            require(digest == self.approved80_paths[str(path)], 'Authorized WOOD80 RGB changed')
        if expected is not None:
            require(digest == expected, 'Expected SHA differs: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == digest,
                'Input changed within evaluation')
        self.bindings[str(path)] = digest
        return digest

    def read(self, path):
        path = self.path(path)
        if not path.is_relative_to(self.root):
            require(self.allow[str(path)]['category'] in ('metadata', 'protocol',
                'checkpoint_receipt', 'cached_scores', 'cached_predictions', 'dev_two_view_predictions'),
                'Unpermitted JSON body')
        digest = self.hash(path)
        value = json.loads(path.read_text())
        self.hash(path, digest)
        return value

    def module(self, path, name):
        path = self.path(path)
        require(path.is_relative_to(self.root) or self.allow[str(path)]['category'] == 'source_code',
                'Unpermitted executable source')
        self.hash(path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def own(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), 'Output outside new experiment')
        return path

    def write(self, path, value):
        path = self.own(path)
        content = json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        self.write_text(path, content)

    def write_text(self, path, content):
        path = self.own(path)
        if path.is_file():
            require(path.read_text() == content, 'Preserve existing different artifact: ' + str(path))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(content)

    def append(self, path, value):
        path = self.own(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(clean(value), ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def csv(self, path, rows):
        import io
        fields = list(dict.fromkeys(k for row in rows for k in row))
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(clean(rows))
        self.write_text(path, buffer.getvalue())

    def unchanged(self):
        for path, digest in self.bindings.items():
            require(sha(path) == digest, 'Input/source changed: ' + path)


def group_id(kind, condition, training_seed):
    require(kind in KINDS and condition in CONDITIONS and training_seed in TRAINING_SEEDS,
            'Unknown model coordinate')
    return f'{kind}__{condition}__seed{training_seed}__R2'


def validate_training_statistics(receipt, kind, condition):
    """Validate global calls separately from Adam's parameter-local steps.

    Real-only batches have no affinity loss. With set_to_none=True the final
    affinity head m6_1.* receives no gradient on those batches, so standard
    Adam correctly leaves only those parameter-local step counters unchanged.
    """
    stats = receipt['stats']
    require(stats['complete'] is True and stats['updates_completed'] == stats['updates_required'] == 828 and
        stats['global_optimizer_steps'] == 1656 and stats['effective_batch_size'] == 16 and
        stats['committed_synthetic_draws'] + stats['committed_real_draws'] == 13248 and
        stats['synthetic_unique_pool'] == SYN_COUNTS[kind][CONDITIONS.index(condition)] and
        stats['real_unique_pool'] == REAL_COUNTS[kind] and
        stats['parameter_step_validation_passed'] is True, 'Proportional training budget/count mismatch')
    history = stats['history']
    require(len(history) == 828 and [r['update'] for r in history] == list(range(1, 829)),
        'Global optimizer update history is incomplete or duplicated')
    require(all(type(r['n_synthetic']) is int and type(r['n_real']) is int and
        min(r['n_synthetic'], r['n_real']) >= 0 and r['n_synthetic'] + r['n_real'] == 16 for r in history),
        'Batch source counts differ from the fixed global batch')
    require(sum(r['n_synthetic'] for r in history) == stats['committed_synthetic_draws'] and
        sum(r['n_real'] for r in history) == stats['committed_real_draws'], 'Exposure counters/history mismatch')
    gradients = stats['per_parameter_gradient_updates']
    initial = stats['optimizer_parameter_initial_steps']
    final = stats['optimizer_parameter_steps']
    require(isinstance(gradients, dict) and gradients and set(gradients) == set(initial) == set(final),
        'Per-parameter gradient/Adam counter catalogues differ')
    require(any(name.startswith('m6_1.') for name in gradients), 'Frozen final affinity head is missing')
    affinity_updates = sum(r['n_synthetic'] > 0 for r in history)
    for name, count in gradients.items():
        expected = affinity_updates if name.startswith('m6_1.') else 828
        require(type(count) is int and count == expected and initial[name] == 828 and
            final[name] == initial[name] + count, 'Parameter-local Adam/gradient history mismatch: ' + name)
    values = sorted(set(final.values()))
    require(stats['optimizer_parameter_step_values'] == values and receipt['optimizer_steps'] == values,
        'Reported parameter-local Adam step values differ from the exact counters')


def trial(io, kind, condition, training_seed):
    folder = io.root / 'runs' / kind / condition / f'seed_{training_seed}'
    cp, receipt_path = folder / 'R2.pth', folder / 'TRAIN_COMPLETE.json'
    receipt = io.read(receipt_path)
    require(receipt.get('complete') is True and receipt['kind'] == kind and
        receipt['condition'] == condition and receipt.get('seed', receipt.get('training_seed')) == training_seed,
        'Training receipt identity/completion mismatch')
    require(receipt.get('round', 2) == 2, 'Only final R2 checkpoints are eligible')
    if 'checkpoint_path' in receipt:
        require(Path(receipt['checkpoint_path']).resolve() == cp, 'Training checkpoint path mismatch')
    protocol_path = folder / 'PROTOCOL.json'
    protocol = io.read(protocol_path)
    require(receipt['protocol_sha256'] == io.hash(protocol_path) and
        receipt['fixed_plan_sha256'] == io.hash(io.root / 'FIXED_PLAN.json') and
        protocol['kind'] == kind and protocol['condition'] == condition and protocol['seed'] == training_seed,
        'Training protocol/plan binding mismatch')
    require(receipt['real_unique_count'] == REAL_COUNTS[kind] and
        receipt['synthetic_unique_count'] == SYN_COUNTS[kind][CONDITIONS.index(condition)] and
        receipt['evaluation_used_for_selection'] is False,
        'Training count/continuation or selection contract mismatch')
    validate_training_statistics(receipt, kind, condition)
    digest = io.hash(cp, receipt['checkpoint_sha256'])
    token = group_id(kind, condition, training_seed)
    return dict(group=token, kind=kind, condition=condition, training_seed=training_seed,
        round=2, model=token, available=True, status='available_count_proportional',
        checkpoint=str(cp), checkpoint_sha256=digest, receipt_path=str(receipt_path),
        receipt_sha256=io.hash(receipt_path), predictions=None)


def verify_go(io):
    path = io.root / 'preflight/EVALUATION_GO.json'
    go = io.read(path)
    require(go.get('passed') is True and go.get('evaluation_authorized') is True and
        go.get('canonical_FP32_authorized') is True, 'Root EVALUATION_GO required')
    require(go['eval_allowlist_sha256'] == io.hash(io.root / 'preflight/EVAL_ALLOWLIST.json'),
            'Evaluation allowlist GO binding mismatch')
    require(go['evaluator_sha256'] == io.hash(__file__), 'Evaluator source not root-reviewed/frozen')
    frozen = go.get('frozen_files')
    require(isinstance(frozen, dict) and frozen, 'Root frozen_files seal is mandatory')
    required = {str(io.root / rel) for rel in (
        'FIXED_PLAN.json', 'preflight/INPUTS.json', 'SUBSETS.json',
        'preflight/TRAINING_GO.json', 'code/evaluate_proportional.py',
        'tests/test_evaluate_proportional.py')}
    require(required.issubset(frozen), 'Root frozen_files is missing required protocol/source/test bindings')
    for raw, digest in frozen.items():
        sealed_path = Path(raw)
        require(str(sealed_path.resolve()) == raw and sealed_path.is_relative_to(io.root) and
            sealed_path != path and isinstance(digest, str) and len(digest) == 64,
            'Malformed frozen_files entry or circular seal')
        io.hash(sealed_path, digest)
    return go


def inference_go(io):
    return verify_go(io)


def prediction_path(io, group):
    return io.root / 'evaluation/predictions' / (group['group'] + '.json')


def validate_prediction(saved, data, group, frames):
    require(saved.get('complete') is True and saved['group'] == group['group'] and
        saved['checkpoint_sha256'] == group['checkpoint_sha256'] and
        saved['receipt_sha256'] == group['receipt_sha256'] and
        saved['manifest_sha256'] == data['manifest_sha256'] and
        saved['runtime'] == frozen_common().CANONICAL_RUNTIME and saved['two_view'] is False and
        saved['evaluator_sha256'] == sha(__file__) and saved['frozen_metric_context_sha256'] == OLD_COMMON_SHA,
        'Prediction cache identity/runtime mismatch')
    rows = saved['records']
    require(len(rows) == len(frames) and [r['id'] for r in rows] == [f['id'] for f in frames],
            'Prediction population/order mismatch')
    for row, frame in zip(rows, frames):
        require(row['image_sha256'] == frame['image_sha256'] and
            row['checkpoint_sha256'] == group['checkpoint_sha256'], 'Prediction input binding mismatch')
        pred = row['original']
        require(len(pred['points']) == len(pred['valid']) == len(pred['confidence']) == 9,
                'Nine semantic prediction slots required')
    return rows


def runtime(io):
    inference_go(io)
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Canonical CUBLAS setting missing')
    import numpy as np
    import cv2
    import torch
    require(torch.cuda.is_available(), 'CUDA FP32 unavailable')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    C = frozen_common()
    adapter = io.module(C.ADAPTER_SOURCE, 'proportional_frozen_adapter')
    model = io.module(C.MODEL_SOURCE, 'proportional_frozen_model')
    require(adapter.THRESHOLD == .3, 'Canonical decoder threshold differs')
    io.forward_runtime = True
    return dict(np=np, cv2=cv2, torch=torch, adapter=adapter, model=model)


def infer_one(io, data, group, rt):
    C = frozen_common()
    path = prediction_path(io, group)
    frames = [f for f in data['frames'] if C.kind_of(f) == group['kind']]
    if path.is_file():
        validate_prediction(io.read(path), data, group, frames)
        return
    journal = path.with_suffix('.jsonl')
    require(not journal.exists(), 'Partial inference journal preserved; root review required')
    np, cv2, torch, adapter = (rt[k] for k in ('np', 'cv2', 'torch', 'adapter'))
    state = torch.load(io.path(group['checkpoint']), map_location='cpu', weights_only=True)
    network = rt['model'].DopeNetwork(pretrained=False, stop_at_stage=6)
    network.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    network.float().requires_grad_(False).eval().to('cuda')
    rows = []
    for frame in frames:
        image_path = io.path(frame['image_path'])
        io.hash(image_path, frame['image_sha256'], raw_dev=True)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        io.raw_rgb_reads += 1
        io.authorized80_reads += int(str(image_path) in io.approved80_paths)
        require(image is not None and image.shape[:2] == (frame['height'], frame['width']),
                'Native image dimensions differ')
        prepared = adapter.prepare(image, source_pre_padded=False)
        event = dict(group=group['group'], id=frame['id'], view='original', image_count=1,
            checkpoint_sha256=group['checkpoint_sha256'])
        io.append(io.root / 'logs/EVALUATION_FORWARDS.jsonl', dict(event, event='attempt', utc=utc()))
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            beliefs, _ = network(prepared['tensor'].unsqueeze(0).to(device='cuda', dtype=torch.float32))
        maps = beliefs[-1][0].detach().cpu().numpy()
        require(maps.dtype == np.float32 and maps.shape[0] == 9 and np.isfinite(maps).all(),
                'Noncanonical belief maps')
        pred = adapter.decode(maps, prepared)
        row = dict(id=frame['id'], kind=group['kind'], image_sha256=frame['image_sha256'],
            checkpoint_sha256=group['checkpoint_sha256'], original=clean(dict(points=pred['points_original'],
            valid=pred['valid'], confidence=pred['confidence'], input_shape=prepared['input_shape'])))
        io.append(journal, row)
        io.append(io.root / 'logs/EVALUATION_FORWARDS.jsonl', dict(event, event='complete', utc=utc()))
        rows.append(row)
    io.hash(group['checkpoint'], group['checkpoint_sha256'])
    saved = dict(complete=True, group=group['group'], kind=group['kind'], condition=group['condition'],
        training_seed=group['training_seed'], checkpoint_sha256=group['checkpoint_sha256'],
        receipt_sha256=group['receipt_sha256'], manifest_sha256=data['manifest_sha256'],
        runtime=C.CANONICAL_RUNTIME, two_view=False, records=rows,
        evaluator_sha256=io.hash(__file__), frozen_metric_context_sha256=OLD_COMMON_SHA,
        actual_image_forwards=len(rows), raw_reference_files_opened=0,
        unauthorized_protected_FINAL_reads=0, journal_sha256=io.hash(journal))
    validate_prediction(saved, data, group, frames)
    io.write(path, saved)
    del network, state, beliefs
    torch.cuda.empty_cache()


def attach_predictions(io, data, group):
    C = frozen_common()
    frames = [f for f in data['frames'] if C.kind_of(f) == group['kind']]
    path = prediction_path(io, group)
    rows = validate_prediction(io.read(path), data, group, frames)
    group['predictions'] = {r['id']: dict(r['original'], id=r['id'], image_sha256=r['image_sha256'],
        checkpoint_sha256=r['checkpoint_sha256']) for r in rows}
    return io.hash(path)


def score_one(io, data, group):
    C = frozen_common()
    cache_sha = attach_predictions(io, data, group)
    summary, frames = C.score_group(data, group)
    for row in (summary, *frames):
        row['training_seed'] = group['training_seed']
    summary.update(real_unique=REAL_COUNTS[group['kind']],
        synthetic_unique=SYN_COUNTS[group['kind']][CONDITIONS.index(group['condition'])],
        synthetic_percent=(20, 50, 100, 200, 400)[CONDITIONS.index(group['condition'])])
    manual = C.score_group(data, group, manual=True)[0] if group['kind'] == 'GREEN' else None
    if manual is not None:
        manual['training_seed'] = group['training_seed']
    output = dict(complete=True, group=group['group'], checkpoint_sha256=group['checkpoint_sha256'],
        receipt_sha256=group['receipt_sha256'], prediction_sha256=cache_sha,
        manifest_sha256=data['manifest_sha256'], summary=summary, frames=frames, manual_summary=manual)
    io.write(io.root / 'evaluation/per_model' / (group['group'] + '.json'), output)
    return output


def metric_vectors(rows, metric, manual=False):
    import numpy as np
    require(metric in QUALITY, 'Unknown metric')
    prefix = 'manual_' if manual else ''
    numerator = 'n_observed' if metric == 'coverage' else 'correct10' if metric == 'conditional_precision10' else 'correct' + metric[3:]
    denominator = 'n_observed' if metric == 'conditional_precision10' else 'n_reference'
    return (np.asarray([r[prefix + numerator] for r in rows], dtype=float),
            np.asarray([r[prefix + denominator] for r in rows], dtype=float))


def metric_draws(rows, metric, weights, manual=False):
    import numpy as np
    numerator, denominator = metric_vectors(rows, metric, manual)
    total = denominator.sum()
    den = weights @ denominator
    valid = den > 0
    draws = np.full(len(weights), np.nan, dtype=float)
    draws[valid] = (weights @ numerator)[valid] / den[valid]
    return (float(numerator.sum() / total) if total > 0 else None), draws


def summarize_difference(candidate, comparator):
    """Average seed-level metric ratios, never pool their detected counts."""
    import numpy as np
    require(len(candidate) == len(comparator) > 0, 'Paired seed count mismatch')
    points = [a[0] - b[0] for a, b in zip(candidate, comparator) if a[0] is not None and b[0] is not None]
    differences = np.asarray([a[1] - b[1] for a, b in zip(candidate, comparator)])
    require(differences.ndim == 2, 'Bootstrap draw shape differs')
    valid = np.isfinite(differences).all(axis=0)
    result = dict(point_difference=float(np.mean(points)) if len(points) == len(candidate) else None,
        ci95_low=None, ci95_high=None, undefined_replicates=int((~valid).sum()),
        decision='NA_undefined_denominator_no_draws_dropped')
    if valid.all() and result['point_difference'] is not None:
        lo, hi = np.quantile(differences.mean(axis=0), [.025, .975], method='linear')
        result.update(ci95_low=float(lo), ci95_high=float(hi),
            decision='candidate_higher' if lo > 0 else 'candidate_lower' if hi < 0 else 'includes_zero')
    return result


def seed_aggregate(summaries):
    import numpy as np
    grouped = {}
    for row in summaries:
        grouped.setdefault((row['kind'], row['condition'], row['panel']), []).append(row)
    output = []
    metrics = (*QUALITY, 'conditional_pixel_error_mean', 'conditional_pixel_error_median')
    for (kind, condition, panel), rows in sorted(grouped.items()):
        require(len(rows) == 3 and {r['training_seed'] for r in rows} == set(TRAINING_SEEDS),
                'Every aggregate requires exactly the three preregistered seeds')
        rows.sort(key=lambda r: r['training_seed'])
        item = dict(kind=kind, condition=condition, panel=panel, n_training_seeds=3,
            training_seeds='42;43;44', training_seed_SD_ddof=1,
            n_DEV=rows[0]['n_DEV'], n_reference_points=rows[0]['n_reference_points'])
        for metric in metrics:
            values = [r[metric] for r in rows]
            finite = all(v is not None and math.isfinite(float(v)) for v in values)
            item[metric + '_seed_mean'] = float(np.mean(values)) if finite else None
            item[metric + '_training_seed_sample_SD'] = float(np.std(values, ddof=1)) if finite else None
        output.append(item)
    return output


def bootstrap_reports(model_outputs, controls, replicates=REPLICATES):
    import numpy as np
    lookup = {}
    for saved in model_outputs:
        row = saved['summary']
        key = (row['kind'], row['condition'], row['training_seed'])
        require(key not in lookup, 'Duplicate model seed key')
        lookup[key] = saved['frames']
    require(len(lookup) == 45, 'Report requires all45 models; no best-seed selection')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_seed, aggregate, receipts = [], [], []
    contrasts = [(c, 'C') for c in CONDITIONS if c != 'C'] + [(c, 'control_R2') for c in CONDITIONS]
    for kind in KINDS:
        n = EXPECTED[kind][0]
        weights = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
        digest = hashlib.sha256(weights.astype('<i8', copy=False).tobytes()).hexdigest()
        ids = [r['id'] for r in lookup[kind, 'C', 42]]
        require(len(ids) == n and len(set(ids)) == n, 'Wrong evaluation frame count')
        for c in CONDITIONS:
            for seed in TRAINING_SEEDS:
                require([r['id'] for r in lookup[kind, c, seed]] == ids, 'Unpaired model frame order')
        require([r['id'] for r in controls[kind]] == ids, 'Unpaired control frame order')
        receipts.append(dict(kind=kind, n_frames=n, weights_sha256=digest, ordered_ids_sha256=stable(ids)))
        for manual in ((False, True) if kind == 'GREEN' else (False,)):
            panel = 'GREEN_manual600' if manual else 'all_annotated'
            values = {}
            for metric in QUALITY:
                values[metric, 'control_R2', None] = metric_draws(controls[kind], metric, weights, manual)
                for c in CONDITIONS:
                    for seed in TRAINING_SEEDS:
                        values[metric, c, seed] = metric_draws(lookup[kind, c, seed], metric, weights, manual)
            for candidate, comparator in contrasts:
                for metric in QUALITY:
                    aa, bb = [], []
                    for seed in TRAINING_SEEDS:
                        a = values[metric, candidate, seed]
                        b = values[metric, comparator, None if comparator == 'control_R2' else seed]
                        aa.append(a)
                        bb.append(b)
                        base = dict(kind=kind, candidate=candidate, comparator=comparator, metric=metric,
                            panel=panel, training_seed=seed, difference_direction='candidate_minus_comparator',
                            n_frame_units=n, replicates=replicates, bootstrap_seed=BOOTSTRAP_SEED,
                            weights_sha256=digest, multiplicity_correction='none', resampling_unit='paired_frame',
                            comparison_scope='contextual_different_R1' if comparator == 'control_R2' else 'matched_training_seed')
                        per_seed.append(dict(base, **summarize_difference([a], [b])))
                    aggregate.append(dict(kind=kind, candidate=candidate, comparator=comparator, metric=metric,
                        panel=panel, n_training_seeds=3, training_seeds='42;43;44',
                        difference_direction='mean_seed_metric_candidate_minus_comparator',
                        statistic='arithmetic_mean_of_three_seed_level_metric_ratios',
                        uncertainty_scope='paired_frames_conditional_on_fixed_three_trained_seeds',
                        n_frame_units=n, replicates=replicates, bootstrap_seed=BOOTSTRAP_SEED,
                        weights_sha256=digest, multiplicity_correction='none', resampling_unit='paired_frame',
                        comparison_scope='contextual_different_R1' if comparator == 'control_R2' else 'matched_training_seeds',
                        **summarize_difference(aa, bb)))
    return per_seed, aggregate, receipts


def make_markdown(aggregates, cis):
    def n(v, scale=1):
        return 'NA' if v is None else f'{v * scale:.3f}'
    lines = ['# Count-proportional R2 results', '', NOTE, '',
        '| Kind | Condition | PCK10 mean ± training-seed SD % | Coverage mean ± SD % | Conditional precision10 mean ± SD % |',
        '|---|---|---:|---:|---:|']
    for r in aggregates:
        if r['panel'] != 'all_annotated':
            continue
        vals = [r['kind'], r['condition']]
        for metric in ('PCK10', 'coverage', 'conditional_precision10'):
            vals.append(n(r[metric + '_seed_mean'], 100) + ' ± ' + n(r[metric + '_training_seed_sample_SD'], 100))
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines += ['', '| Kind | Candidate | Comparator | Mean-seed PCK10 difference pp | Conditional frame95% CI pp | Decision |',
        '|---|---|---|---:|---|---|']
    for r in cis:
        if r['panel'] == 'all_annotated' and r['metric'] == 'PCK10':
            lines.append('| ' + ' | '.join([r['kind'], r['candidate'], r['comparator'], n(r['point_difference'], 100),
                '[' + n(r['ci95_low'], 100) + ', ' + n(r['ci95_high'], 100) + ']', r['decision']]) + ' |')
    return '\n'.join(lines) + '\n'


def report(io, data):
    C = frozen_common()
    complete_path = io.root / 'evaluation/final/COMPLETE.json'
    if complete_path.is_file():
        saved = io.read(complete_path)
        require(saved.get('complete') is True and saved['models'] == 45 and
            saved['manifest_sha256'] == data['manifest_sha256'] and
            saved['source_sha256'] == io.hash(__file__), 'Existing final report binding mismatch')
        for path, digest in saved['output_sha256'].items():
            io.hash(path, digest)
        for path, digest in saved['input_sha256'].items():
            # Exact80 raw RGB is checked as immutable bytes without reopening
            # its image content; score/report never reads raw annotations.
            require(sha(path) == digest, 'Final report input changed: ' + path)
        return
    outputs, summaries, manuals, controls, control_summaries = [], [], [], {}, []
    for kind in KINDS:
        for condition in CONDITIONS:
            for seed in TRAINING_SEEDS:
                group = trial(io, kind, condition, seed)
                saved = score_one(io, data, group)
                outputs.append(saved)
                summaries.append(saved['summary'])
                if saved['manual_summary'] is not None:
                    manuals.append(saved['manual_summary'])
        control = C.cached_group(io, data, kind, 'control_R2')
        summary, rows = C.score_group(data, control)
        controls[kind] = rows
        control_summaries.append(summary)
    aggregates = seed_aggregate(summaries + manuals)
    seed_ci, aggregate_ci, sampling = bootstrap_reports(outputs, controls)
    folder = io.root / 'evaluation/final'
    io.csv(folder / 'MODEL_SUMMARY.csv', summaries)
    io.csv(folder / 'GREEN_MANUAL_MODEL_SUMMARY.csv', manuals)
    io.csv(folder / 'SEED_AGGREGATE_SUMMARY.csv', [r for r in aggregates if r['panel'] == 'all_annotated'])
    io.csv(folder / 'GREEN_MANUAL_SEED_AGGREGATE_SUMMARY.csv', [r for r in aggregates if r['panel'] != 'all_annotated'])
    io.csv(folder / 'PAIRED_CI_PER_SEED.csv', [r for r in seed_ci if r['panel'] == 'all_annotated'])
    io.csv(folder / 'GREEN_MANUAL_PAIRED_CI_PER_SEED.csv', [r for r in seed_ci if r['panel'] != 'all_annotated'])
    io.csv(folder / 'PAIRED_CI_SEED_MEAN.csv', [r for r in aggregate_ci if r['panel'] == 'all_annotated'])
    io.csv(folder / 'GREEN_MANUAL_PAIRED_CI_SEED_MEAN.csv', [r for r in aggregate_ci if r['panel'] != 'all_annotated'])
    io.csv(folder / 'CONTEXT_CONTROL_SUMMARY.csv', control_summaries)
    io.csv(folder / 'FRAME_ERRORS.csv', [{k: v for k, v in r.items() if k not in
        ('point_errors', 'errors_px', 'manual_errors_px')} for item in outputs for r in item['frames']])
    io.csv(folder / 'NATIVE_RESOLUTIONS.csv', [dict(id=f['id'], kind=C.kind_of(f), width=f['width'],
        height=f['height'], session=f['session'], authorization_scope=f['authorization_scope']) for f in data['frames']])
    io.write(folder / 'BOOTSTRAP_RECEIPT.json', dict(complete=True, replicates=REPLICATES,
        seed=BOOTSTRAP_SEED, percentile_method='linear', sampling=sampling,
        primary_per_seed_rows=405, primary_seed_mean_rows=135, manual_per_seed_rows=135,
        manual_seed_mean_rows=45, multiplicity_correction='none', no_undefined_draws_discarded=True,
        training_seeds=list(TRAINING_SEEDS), training_seed_SD_ddof=1,
        seed_mean_CI_scope='paired_frames_conditional_on_fixed_three_trained_seeds'))
    io.write_text(folder / 'RESULTS.md', make_markdown(aggregates, aggregate_ci))
    io.unchanged()
    payloads = sorted(p for p in folder.iterdir() if p.name != 'COMPLETE.json')
    io.write(folder / 'COMPLETE.json', dict(complete=True, models=45, seed_aggregate_groups=15,
        manual_seed_groups=15, manual_seed_aggregate_groups=5, primary_population='PRIMARY372',
        manifest_sha256=data['manifest_sha256'], source_sha256=io.hash(__file__),
        output_sha256={str(p): sha(p) for p in payloads}, input_sha256=dict(io.bindings),
        raw_reference_files_opened=0, unauthorized_protected_FINAL_reads=0, note=NOTE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('infer', 'score', 'report', 'all'), required=True)
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--seed', type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()
    require(args.phase != 'report' or not any((args.kind, args.condition, args.seed)),
            'Final reporting requires all45 models, without a selected condition/seed')
    io = Inputs(args.root)
    C = frozen_common()
    data = C.load_data(io)
    coordinates = [(k, c, s) for k in KINDS for c in CONDITIONS for s in TRAINING_SEEDS
        if (args.kind is None or k == args.kind) and (args.condition is None or c == args.condition)
        and (args.seed is None or s == args.seed)]
    if args.phase in ('infer', 'all'):
        require(GPU_LOCK.is_file() and not GPU_LOCK.is_symlink(), 'Global GPU lock missing')
        with GPU_LOCK.open('r') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            rt = runtime(io)
            for k, c, s in coordinates:
                infer_one(io, data, trial(io, k, c, s), rt)
    if args.phase in ('score', 'all'):
        for k, c, s in coordinates:
            score_one(io, data, trial(io, k, c, s))
    if args.phase == 'report' or args.phase == 'all' and not any((args.kind, args.condition, args.seed)):
        report(io, data)
    io.unchanged()
    print('PROPORTIONAL_EVALUATION_COMPLETE', args.phase, len(coordinates), flush=True)


if __name__ == '__main__':
    main()
