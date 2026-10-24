"""One source-only YCB foreign-ray adaptation from synthetic source pretraining.

No current evaluation reader or pose optimization is present. The reviewed V87
target is used unchanged; legacy synthetic multi-ray fractions are not relabeled
as this different single-ray target. All instances and unknown slots are kept.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'
RUN = ROOT / 'runs/YCB_EXTERNAL_RAY_V1'
FIELDS = ('p3_roi', 'local_rgb96', 'local_valid96', 'target_rgb128', 'target_valid128', 'context42')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def check(item):
    assert sha(item['path']) == item['sha256'], item['path']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def protocol():
    p = read(PROTOCOL)
    check(read(ROOT / 'PROTOCOL_BINDING.json')['protocol'])
    for b in p['bindings'].values():
        check(b)
    assert p['parameter_count'] == 41237 and p['optimizer']['updates'] == 2000
    assert p['sampling']['batch_queries'] == 96 and p['sampling']['per_class'] == 48
    return p


def model_module(p):
    s = importlib.util.spec_from_file_location('v88_external', p['bindings']['model_code']['path'])
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def admission(y, qvalid, roi, local, target, r, q, side):
    offset = (r - q) / side[:, None, None]
    square = np.isfinite(offset).all(-1) & (offset >= -.5).all(-1) & (offset < .5).all(-1)
    available = roi & local & target[:, None]
    return np.isin(y, [0, 1]) & qvalid & available & square, square, offset, available


def objective(logits, labels, admitted):
    known = admitted.bool() & ((labels == 0) | (labels == 1))
    return F.binary_cross_entropy_with_logits(logits[known], labels[known].to(logits.dtype)) if known.any() else logits.sum() * 0


def sample(pools, rng):
    return np.concatenate([rng.choice(pools[c], 48, replace=True) for c in (0, 1)])


def generated():
    p = protocol()
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    m = model_module(p)
    model = m.ExternalRayHead()
    assert tuple(inspect.signature(model.forward).parameters) == FIELDS
    assert sum(x.numel() for x in model.parameters()) == 41237
    # Generated parameter tensors only: no checkpoint/feature/GT arrays loaded.
    assert not any('localizer' in k for k in model.state_dict())
    torch.nn.init.zeros_(model.classifier[2].weight)
    torch.nn.init.zeros_(model.classifier[2].bias)
    vals = dict(p3_roi=torch.randn(1, 8, 65, 9, 9), local_rgb96=torch.randint(0, 256, (1, 8, 3, 96, 96), dtype=torch.uint8), local_valid96=torch.ones(1, 8, 1, 96, 96, dtype=torch.bool), target_rgb128=torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8), target_valid128=torch.ones(1, 1, 128, 128, dtype=torch.bool), context42=torch.randn(1, 8, 42))
    vals['p3_roi'][:, :, 64] = 1
    whole = model(**vals)
    flat = {k: (v.repeat_interleave(8, 0) if k in ('target_rgb128', 'target_valid128') else v.reshape(8, *v.shape[2:])) for k, v in vals.items()}
    assert torch.equal(whole['external_logit'].flatten(), model(**flat)['external_logit'])
    # Nonzero output checks equivalence without relying on zero initialization.
    torch.nn.init.normal_(model.classifier[2].weight, std=.1)
    assert torch.allclose(model(**vals)['external_logit'].flatten(), model(**flat)['external_logit'], rtol=1e-6, atol=1e-7)
    try:
        model(**flat, reference_GT=torch.zeros(8, 2))
    except TypeError:
        pass
    else:
        raise AssertionError('GT forward must reject')
    logits = torch.randn(7, requires_grad=True)
    labels = torch.tensor([1, 0, -1, 1, 0, -1, 1])
    mask = torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
    objective(logits, labels, mask).backward()
    assert logits.grad[:2].abs().min() > 0 and not logits.grad[2:].any()
    y = np.array([[1, 0, -1, 1, 0, 1, 1, 0]])
    q = np.zeros((1, 8, 2)); r = q.copy()
    r[0, 3] = [.5, 0]; r[0, 4] = [-.5, 0]; r[0, 5] = [0, np.nan]
    qv = np.ones((1, 8), bool); qv[0, 6] = False
    roi = np.ones_like(qv); roi[0, 7] = False
    admitted, square, _, _ = admission(y, qv, roi, np.ones_like(qv), np.ones(1, bool), r, q, np.ones(1))
    assert admitted.tolist() == [[True, True, False, False, True, False, False, False]] and square[0, 4] and not square[0, 3]
    pools = {0: np.arange(7), 1: np.arange(7, 12)}
    rng = np.random.default_rng(20261023)
    for _ in range(64):
        ix = sample(pools, rng)
        assert len(ix) == 96 and np.sum(ix < 7) == 48 and np.sum(ix >= 7) == 48
    write(ROOT / 'GENERATED_TESTS.json', dict(PASS=True, protocol=bind(PROTOCOL), trainer=bind(__file__), model_code=p['bindings']['model_code'], six_tensor_forward=True, GT_forward_rejected=True, U_invalid_miss_loss_gradients_zero=True, half_open_edges_checked=True, whole_flat_nonzero_agreement=True, balanced_sampler_batches=64, source_arrays_checkpoint_reads=0, optimizer_updates=0, current_PnP=0))
    print('GENERATED_PASS', flush=True)


def freeze():
    p = protocol(); generated = read(ROOT / 'GENERATED_TESTS.json')
    assert generated['PASS'] and generated['trainer'] == bind(__file__)
    review = read(ROOT / 'independent_preflight/INDEPENDENT_PREFIT_REVIEW.json')
    assert review['PASS'] and review['blockers'] == []
    assert review['protocol'] == bind(PROTOCOL) and review['trainer'] == bind(__file__)
    write(ROOT / 'TRAINER_ENTRYLOCK.json', dict(protocol=bind(PROTOCOL), trainer=bind(__file__), generated=bind(ROOT / 'GENERATED_TESTS.json'), independent_review=bind(ROOT / 'independent_preflight/INDEPENDENT_PREFIT_REVIEW.json'), run_directory=str(RUN), restart_allowed_after_first_update=False, current_PnP=0))
    print('ENTRY_FROZEN', flush=True)


def load_source(p, device):
    source = read(p['bindings']['source_input_bindings']['path'])
    pack = next(x for x in source['packs'] if x['source'] == 'YCB_PUBLIC_REAL')
    for k in ('manifest', 'seal', 'lineage'):
        check(pack[k])
    manifest = read(pack['manifest']['path'])
    assert manifest['completed'] and manifest['frame_count'] == 4200
    lines = [json.loads(x) for x in Path(pack['lineage']['path']).read_text().splitlines()]
    fields = ('p3_roi', 'rgb96', 'crop_valid96', 'target_rgb128', 'target_valid128', 'context42', 'query8_xy', 'query_center_valid', 'ROI_anyvalid', 'L')
    parts = {k: [] for k in fields}
    for chunk in manifest['chunks']:
        check(chunk['file'])
        with np.load(chunk['file']['path'], allow_pickle=False) as z:
            for k in fields:
                parts[k].append(z[k])
    data = {k: np.ascontiguousarray(np.concatenate(v)) for k, v in parts.items()}
    del parts
    with np.load(p['bindings']['teacher_GT']['path'], allow_pickle=False) as z:
        gt = {k: z[k] for k in z.files}
    assert len(lines) == 4200 and data['p3_roi'].shape == (4200, 8, 65, 9, 9)
    assert all(x['id'] == gt['id'][i] and x['split'] == gt['split'][i] for i, x in enumerate(lines))
    assert np.all(gt['human_semantic_gold8'] == -1)
    assert np.isin(gt['label8'], [-1, 0, 1]).all()
    assert np.all((gt['label8'] == 1) == (gt['reason8'] == 'EXTERNAL_OTHER_FIRST_HIT_PROXY'))
    assert np.all((gt['label8'] == 0) == (gt['reason8'] == 'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY'))
    local = data['crop_valid96'].reshape(4200, 8, -1).any(-1)
    target = data['target_valid128'].reshape(4200, -1).any(-1)
    admitted, square, offset, available = admission(gt['label8'], data['query_center_valid'], data['ROI_anyvalid'], local, target, gt['reference_xy8'], data['query8_xy'], data['L'])
    meta = dict(y=gt['label8'].reshape(-1).astype(np.int64), primary=admitted.reshape(-1), square=square.reshape(-1), offset=offset.reshape(-1, 2), qvalid=data['query_center_valid'].reshape(-1), input_available=available.reshape(-1), train=np.repeat(gt['split'] == 'train', 8), val=np.repeat(gt['split'] == 'val', 8), self_visible=gt['convex_self_visible_proxy8_separate'].reshape(-1), reason=gt['reason8'].reshape(-1), ids=gt['id'], split=gt['split'])
    meta['pools'] = {c: np.flatnonzero(meta['primary'] & meta['train'] & (meta['y'] == c)) for c in (0, 1)}
    meta['valpools'] = {c: np.flatnonzero(meta['primary'] & meta['val'] & (meta['y'] == c)) for c in (0, 1)}
    assert all(len(x) > 0 for x in [*meta['pools'].values(), *meta['valpools'].values()])
    gpu = dict(p3_roi=torch.from_numpy(data['p3_roi'].reshape(-1, 65, 9, 9)).to(device), local_rgb96=torch.from_numpy(data['rgb96'].reshape(-1, 3, 96, 96)).to(device), local_valid96=torch.from_numpy(data['crop_valid96'].reshape(-1, 1, 96, 96)).to(device), target_rgb128=torch.from_numpy(data['target_rgb128']).to(device), target_valid128=torch.from_numpy(data['target_valid128']).to(device), context42=torch.from_numpy(data['context42'].reshape(-1, 42)).to(device))
    del data
    return gpu, torch.as_tensor(meta['y'], device=device), meta


def forward(model, gpu, indices):
    ix = torch.as_tensor(indices, dtype=torch.long, device=gpu['p3_roi'].device)
    frame = ix // 8
    return model(**{k: v[frame if k in ('target_rgb128', 'target_valid128') else ix] for k, v in gpu.items()})


def predict(model, gpu, indices, labels):
    model.eval(); logits = []; available = []; nll = []
    with torch.inference_mode():
        for start in range(0, len(indices), 128):
            ix = indices[start:start + 128]
            result = forward(model, gpu, ix)
            logits.append(result['external_logit'].cpu().numpy())
            available.append(result['input_available'].cpu().numpy())
            nll.append(F.binary_cross_entropy_with_logits(result['external_logit'], labels[torch.as_tensor(ix, device=labels.device)].clamp(0, 1).float(), reduction='none').cpu().numpy())
    return dict(logits=np.concatenate(logits), available=np.concatenate(available), nll=np.concatenate(nll))


def metric(pred, indices, meta):
    where = np.full(33600, -1, np.int64); where[indices] = np.arange(len(indices))
    bce = {str(c): float(pred['nll'][where[meta['valpools'][c]]].astype(np.float64).mean()) for c in (0, 1)}
    return float(np.mean(list(bce.values()))), bce


def confusion(pred, mask, meta):
    ix = np.flatnonzero(mask)
    factual = meta['qvalid'][ix] & pred['available'][ix]
    predicted = np.where(factual, (pred['logits'][ix] >= 0).astype(np.int64), 2)
    cm = np.zeros((2, 3), np.int64); np.add.at(cm, (meta['y'][ix], predicted), 1)
    ratio = lambda n, d: float(n / d) if d else None
    return dict(N=len(ix), confusion_rows_GT_CLEAR_EXTERNAL_cols_pred_CLEAR_EXTERNAL_U=cm.tolist(), clear_precision=ratio(cm[0, 0], cm[:, 0].sum()), clear_recall=ratio(cm[0, 0], cm[0].sum()), external_precision=ratio(cm[1, 1], cm[:, 1].sum()), external_recall=ratio(cm[1, 1], cm[1].sum()), factual_U_rate=ratio(cm[:, 2].sum(), cm.sum()))


def report(pred, meta):
    result = {}
    for part in ('train', 'val'):
        base = meta[part]; known = base & np.isin(meta['y'], [0, 1])
        result[part] = dict(all_slots=int(base.sum()), teacher_known=int(known.sum()), teacher_U=int(np.sum(base & (meta['y'] == -1))), admitted_loss=int(np.sum(base & meta['primary'])), known_reference_outside_qcrop=int(np.sum(known & ~meta['square'])), all_known_fixed_gate_metrics=confusion(pred, known, meta), conditional_admitted_metrics=confusion(pred, base & meta['primary'], meta), all_known_by_convex_self={str(c): confusion(pred, known & (meta['self_visible'] == c), meta) for c in (0, 1)})
    return result


def scores(path, pred, meta):
    np.savez_compressed(path, id=meta['ids'], split=meta['split'], external_logit=pred['logits'].reshape(4200, 8), input_available=pred['available'].reshape(4200, 8), query_center_valid=meta['qvalid'].reshape(4200, 8), loss_admitted_GT_only=meta['primary'].reshape(4200, 8), reference_inside_qcrop_GT_only=meta['square'].reshape(4200, 8), NLL_FP32=pred['nll'].reshape(4200, 8))


def run():
    p = protocol(); entry = read(ROOT / 'TRAINER_ENTRYLOCK.json')
    assert entry['trainer'] == bind(__file__) and entry['protocol'] == bind(PROTOCOL)
    check(entry['generated']); check(entry['independent_review'])
    assert not RUN.exists(), 'Preserve any earlier fit; no automatic restart'
    assert Path(sys.executable).resolve() == Path(p['runtime']['python']).resolve() and str(torch.__version__) == p['runtime']['torch'] and torch.cuda.is_available()
    RUN.mkdir(parents=True); (RUN / 'code_snapshot').mkdir()
    for b in [entry['trainer'], entry['protocol'], p['bindings']['model_code'], p['bindings']['base_architecture'], p['bindings']['source_input_bindings']]:
        shutil.copy2(b['path'], RUN / 'code_snapshot' / Path(b['path']).name)
    write(RUN / 'RUN_STARTED.json', dict(protocol=bind(PROTOCOL), entry=bind(ROOT / 'TRAINER_ENTRYLOCK.json'), optimizer_updates=0, current_PnP=0))
    torch.set_num_threads(1); torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    np.random.seed(20261023); random.seed(20261023)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    gpu, labels, meta = load_source(p, 'cuda:0')
    module = model_module(p); model = module.ExternalRayHead().cuda()
    rejected = module.initialize_from_source_pretrain(model, torch.load(p['bindings']['source_pretrain']['path'], map_location='cuda', weights_only=True))
    assert sorted(rejected) == ['classifier.2.bias', 'classifier.2.weight', 'reference_localizer.bias', 'reference_localizer.weight']
    assert not model.classifier[2].weight.any() and not model.classifier[2].bias.any()
    all_indices = np.arange(33600); val_indices = np.flatnonzero(meta['primary'] & meta['val'])
    pool_counts = {str(c): dict(train=len(meta['pools'][c]), val=len(meta['valpools'][c])) for c in (0, 1)}
    ix = sample(meta['pools'], np.random.default_rng(20261023))
    loss = objective(forward(model, gpu, ix)['external_logit'], labels[torch.as_tensor(ix, device='cuda')], torch.ones(96, dtype=torch.bool, device='cuda'))
    assert torch.isfinite(loss); loss.backward(); model.zero_grad(set_to_none=True)
    write(RUN / 'SOURCE_GPU_PREFLIGHT.json', dict(PASS=True, optimizer_updates=0, pool_counts=pool_counts, loss=float(loss.detach()), peak_GPU_bytes=torch.cuda.max_memory_allocated(), rejected_pretrain_output_keys=rejected, current_PnP=0))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, RUN / 'UNTRAINED_UPDATE0.pt')
    baseline = predict(model, gpu, all_indices, labels)
    assert np.array_equal(baseline['available'], meta['input_available'])
    scores(RUN / 'BASELINE0_SOURCE_SCORES.npz', baseline, meta)
    write(RUN / 'BASELINE0_RESULTS.json', dict(selection_eligible=False, metrics=report(baseline, meta)))
    del baseline
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001, betas=(.9, .999))
    rng = np.random.default_rng(20261023); best = float('inf'); best_step = None; history = []
    for step in range(1, 2001):
        ix = sample(meta['pools'], rng); model.train(); optimizer.zero_grad(set_to_none=True)
        loss = objective(forward(model, gpu, ix)['external_logit'], labels[torch.as_tensor(ix, device='cuda')], torch.ones(96, dtype=torch.bool, device='cuda'))
        assert torch.isfinite(loss); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); assert torch.isfinite(grad)
        optimizer.step()
        if step == 1:
            write(RUN / 'FIRST_UPDATE.json', dict(update=1, loss=float(loss.detach()), restart_allowed=False, current_PnP=0))
            print('FIRST_UPDATE_COMMITTED', flush=True)
        if step % 100 == 0:
            pred = predict(model, gpu, val_indices, labels); score, per_class = metric(pred, val_indices, meta)
            saved = score < best
            if saved:
                best = score; best_step = step
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, RUN / 'BEST_EXTERNAL_HEAD.pt')
            row = dict(update=step, train_loss=float(loss.detach()), heldout_macroBCE=score, per_class_BCE=per_class, saved=saved, best_update=best_step, seconds=time.monotonic() - started)
            history.append(row); (RUN / 'HISTORY.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(RUN / 'BEST_EXTERNAL_HEAD.pt', map_location='cuda', weights_only=True)); model.eval().requires_grad_(False)
    write(RUN / 'MODEL_FREEZE.json', dict(completed=True, optimizer_updates=2000, best_update=best_step, selection_macroBCE=best, checkpoint=bind(RUN / 'BEST_EXTERNAL_HEAD.pt'), model_code=p['bindings']['model_code'], base_architecture=p['bindings']['base_architecture'], protocol=bind(PROTOCOL), trainer=bind(__file__), external_probability_threshold=.5, semantic_gold=False, current_for_fit_selection=False, PnP=0, fully_solved=False))
    pred = predict(model, gpu, all_indices, labels)
    assert np.array_equal(pred['available'], meta['input_available'])
    scores(RUN / 'SELECTED_SOURCE_SCORES.npz', pred, meta)
    results = report(pred, meta); gate = results['val']['all_known_fixed_gate_metrics']
    gates = dict(external_recall_ge_0p8=gate['external_recall'] is not None and gate['external_recall'] >= .8, clear_recall_ge_0p8=gate['clear_recall'] is not None and gate['clear_recall'] >= .8, external_precision_ge_0p5=gate['external_precision'] is not None and gate['external_precision'] >= .5, factual_U_rate_le_0p1=gate['factual_U_rate'] is not None and gate['factual_U_rate'] <= .1)
    write(RUN / 'SOURCE_RESULTS.json', dict(completed=True, model_freeze=bind(RUN / 'MODEL_FREEZE.json'), source_metrics=results, pool_counts=pool_counts, predeclared_source_gate=gates, all_four_pass=all(gates.values()), current_evaluation_admission=False, meaning='Foreign-first-hit single-ray proxy only; not physical corner visibility. Source gate passing would permit a separately frozen public physical-cue evaluation, not automatic current223 evaluation.'))
    write(RUN / 'FIT_COMPLETE.json', dict(completed=True, optimizer_updates=2000, best_update=best_step, model_freeze=bind(RUN / 'MODEL_FREEZE.json'), history=bind(RUN / 'HISTORY.json'), scores=bind(RUN / 'SELECTED_SOURCE_SCORES.npz'), source_results=bind(RUN / 'SOURCE_RESULTS.json'), seconds=time.monotonic() - started, peak_GPU_bytes=torch.cuda.max_memory_allocated(), current_PnP=0))
    print(json.dumps(dict(COMPLETE=bind(RUN / 'FIT_COMPLETE.json'), best_update=best_step, best_macroBCE=best)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['generated', 'freeze', 'run'])
    arguments = parser.parse_args(); {'generated': generated, 'freeze': freeze, 'run': run}[arguments.stage]()
