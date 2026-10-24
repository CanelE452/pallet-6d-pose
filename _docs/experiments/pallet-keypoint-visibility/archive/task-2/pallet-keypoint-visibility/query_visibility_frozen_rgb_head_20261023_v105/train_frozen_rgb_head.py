"""One prospective C8 fit on public/synthetic source features only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
import numpy as np
import torch

V = Path(__file__).resolve().parent
P = V.parent
DESIGN = V / 'ROOT_C8_DESIGN_BEFORE_FEATURE_FORWARD.json'
DESIGN_SHA = '2ee8c6db885399a9148ab8dc3c7186b7769d722b60d5d381f982b6521599bc7e'
PREV = P / 'query_visibility_external_only_bounded_prior_20261023_v102'
ADAPTER = PREV / 'source_fit_preparation/external_data_adapter.py'
ADAPTER_SHA = 'c83d0a9edfc15be6a380a9872624a513d8d526b644022ac91b3bc5168ea00c30'
RUN = V / 'runs/C8_FROZEN_IMAGENET_RGB_TWO_SCALE'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def adapter():
    assert sha(ADAPTER) == ADAPTER_SHA
    spec = importlib.util.spec_from_file_location('c8_external_data', ADAPTER)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class FrozenRGBHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(1026, 64), torch.nn.ReLU(),
                                        torch.nn.Linear(64, 1))
        torch.nn.init.zeros_(self.head[-1].weight)
        torch.nn.init.zeros_(self.head[-1].bias)

    def forward(self, features):
        return self.head(features).squeeze(-1)


def report(d, z, available, meta, design):
    old, old_checks, _ = d.old_helper().report(z[:d.NOLD], available[:d.NOLD], meta['old_meta'])
    new = {}
    for split in ('TRAIN', 'VALIDATION'):
        mask = (meta['domain'] == 'V99_SYNTH') & (meta['split'] == split)
        new[split] = dict(all_rows=d.external_cm(z, available, meta, mask),
                          by_variant={v: d.external_cm(z, available, meta, mask & (meta['variant'] == v))
                                      for v in d.STAGES},
                          selected_pairs=d.new_pair_metrics(z, available, meta,
                              meta['split'][meta['new_pairs'][:, 0]] == split))
    names = {'old_LOCO_before_D_recall_min': 'LOCO_before_D_recall',
             'old_PALLOX_before_D_recall_min': 'PALLOX_before_D_recall',
             'old_natural_before_E_recall_min': 'natural_before_E_recall',
             'old_sham_D_recall_min': 'sham_D_recall',
             'old_cover_E_recall_min': 'cover_E_recall',
             'old_accepted_VAL_pair_all3_min': 'accepted_VAL_pair_all3_correct_rate',
             'old_known_VAL_U_rate_max': 'known_VAL_U_rate'}
    gates = {k: dict(value=old_checks[n]['value'], limit=design['source_gates'][k])
             for k, n in names.items()}
    pred = d.decode(z, available)
    vm = d.val_masks(meta)
    for key, cohort, target in [('new_BEFORE_D_recall_min', 'new_BEFORE_D', 0),
                                 ('new_SHAM_D_recall_min', 'new_SHAM_D', 0),
                                 ('new_COVER_E_recall_min', 'new_COVER_E', 1)]:
        mask = vm[cohort]
        gates[key] = dict(value=float((pred[mask] == target).mean()),
                          limit=design['source_gates'][key], N=int(mask.sum()))
    key = 'new_selected_VAL_pair_all3_min'
    gates[key] = dict(value=new['VALIDATION']['selected_pairs']['all3_correct_rate'],
                      limit=design['source_gates'][key])
    mask = (meta['domain'] == 'V99_SYNTH') & (meta['split'] == 'VALIDATION') & meta['known']
    key = 'new_known_VAL_U_rate_max'
    gates[key] = dict(value=float((~available[mask]).mean()), limit=design['source_gates'][key],
                      N=int(mask.sum()))
    assert set(gates) == set(design['source_gates']) - {'empty_required_cohort'}
    for key, value in gates.items():
        value['PASS'] = value['value'] is not None and (value['value'] <= value['limit']
            if key.endswith('_max') else value['value'] >= value['limit'])
    return dict(old_exact_V96=old, new_external=new), gates


def run(entry_path):
    assert not RUN.exists(), 'A source fit is never silently restarted.'
    entry = read(entry_path)
    assert entry['approved_source_fit'] and entry['current_allowed'] is False
    assert entry['PnP_allowed'] is False and entry['restart_after_update'] is False
    assert sha(DESIGN) == DESIGN_SHA == entry['design']['sha256']
    assert sha(__file__) == entry['trainer']['sha256']
    for key in ('feature_pack', 'feature_receipt', 'source_bindings', 'trainer'):
        assert sha(entry[key]['path']) == entry[key]['sha256'], key
    feature_receipt = read(entry['feature_receipt']['path'])
    assert feature_receipt['completed'] and feature_receipt['PASS']
    assert feature_receipt['output'] == entry['feature_pack']
    assert feature_receipt['model_state_unchanged'] and feature_receipt['first_batch_repeat_exact']
    d = adapter()
    design = read(DESIGN)
    bindings = read(entry['source_bindings']['path'])
    meta = d.load_metadata(bindings)
    with np.load(entry['feature_pack']['path'], allow_pickle=False) as pack:
        features = np.array(pack['features'], copy=True)
        units = np.array(pack['occurrence_unit'], copy=True)
        available = np.array(pack['query_available'], copy=True)
    assert features.shape == (3652, 1026) and features.dtype == np.float32
    assert np.isfinite(features).all()
    assert units.shape == available.shape == (33984,) and available.dtype == bool
    assert units.min() == 0 and units.max() == 3651
    assert np.array_equal(available, [r['input_facts']['operator_query_available'] for r in meta['lineage']])
    assert meta['known'].sum() == 1518
    assert not meta['known'][meta['known_geometric_S']].any()
    assert torch.__version__ == '2.1.1+cu118' and torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(206)
    torch.cuda.manual_seed_all(206)
    model = FrozenRGBHead().cuda()
    assert sum(p.numel() for p in model.parameters()) == 65793
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    x = torch.as_tensor(features, device='cuda')
    unit_gpu = torch.as_tensor(units, dtype=torch.long, device='cuda')
    a_gpu = torch.as_tensor(available, device='cuda')
    targets = torch.as_tensor(meta['external_target'], dtype=torch.float32, device='cuda')
    pools = d.sampling_pools(meta)
    generator = torch.Generator(device='cpu').manual_seed(205)
    RUN.mkdir(parents=True)
    write(RUN / 'RUN_STARTED.json', dict(entry=bind(entry_path), optimizer_updates=0))
    updates = 0
    try:
        best, selected, state = float('inf'), None, None
        draw_hash = hashlib.sha256()
        start = time.monotonic()

        def checkpoint(step):
            return dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        update=step, design=bind(DESIGN), trainer=bind(__file__), entry=bind(entry_path))

        @torch.no_grad()
        def evaluate(step):
            nonlocal best, selected, state
            model.eval()
            z = model(x)[unit_gpu].cpu().numpy()
            z = np.where(available, z, 0).astype(np.float32)
            value, parts = d.selector(z, available, meta)
            with (RUN / 'HISTORY.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=step, criterion_eight_VAL_cohorts_BCE=value,
                                        VAL_cohorts=parts), allow_nan=False) + '\n')
            if value < best:
                best, selected = value, step
                state = checkpoint(step)
                torch.save(state, RUN / 'BEST_FROZEN_RGB_HEAD.pt')
            return z

        torch.save(checkpoint(0), RUN / 'INITIAL_CHECKPOINT.pt')
        initial = evaluate(0)
        assert np.count_nonzero(initial) == 0
        for step in range(1, 2001):
            ix, triples, audit = d.draw(pools, generator)
            assert meta['known'][ix].all() and (meta['split'][ix] == 'TRAIN').all()
            assert (meta['external_target'][ix] == 0).sum() == 32
            assert (meta['external_target'][ix] == 1).sum() == 32
            draw_hash.update(ix.astype('<i8').tobytes())
            draw_hash.update(triples.astype('<i8').tobytes())
            model.train()
            opt.zero_grad(set_to_none=True)
            idx = torch.as_tensor(ix, dtype=torch.long, device='cuda')
            z = model(x[unit_gpu[idx]])
            a = a_gpu[idx]
            target = targets[idx]
            assert a.any()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(z[a], target[a])
            pair = torch.as_tensor(triples, dtype=torch.long, device='cuda')
            cover, sham = pair[:, 1], pair[:, 2]
            pair_ok = a[cover] & a[sham]
            assert (target[cover] == 1).all() and (target[sham] == 0).all()
            hinge = (torch.relu(2 - (z[cover[pair_ok]] - z[sham[pair_ok]])).mean()
                     if pair_ok.any() else z.sum() * 0)
            loss = bce + .25 * hinge
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            assert torch.isfinite(norm)
            opt.step()
            updates = step
            with (RUN / 'TRAIN_UPDATES.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=step, occurrence_indices=ix.tolist(),
                    input_unit_indices=units[ix].tolist(), matched_triplet_batch_positions=triples.tolist(),
                    loss=float(loss.detach()), BCE=float(bce.detach()), matched_hinge=float(hinge.detach()),
                    gradient_norm_before_clip=float(norm)), allow_nan=False) + '\n')
            if step % 100 == 0:
                evaluate(step)
        torch.save(checkpoint(2000), RUN / 'FINAL_CHECKPOINT.pt')
        model.load_state_dict(state['state_dict'])
        with torch.no_grad():
            z = model(x)[unit_gpu].cpu().numpy()
        z = np.where(available, z, 0).astype(np.float32)
        value, parts = d.selector(z, available, meta)
        assert value == best
        metrics, gates = report(d, z, available, meta, design)
        passed = all(g['PASS'] for g in gates.values())
        expected_draw = read(PREV / 'source_fit_preparation/FIT_PROTOCOL_BEFORE_SOURCE_FORWARD.json')['frozen_2000_draws_sha256']
        assert draw_hash.hexdigest() == expected_draw
        line = meta['lineage']
        query = np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan, np.nan] for r in line])
        scale = np.array([r['L'] if r['L'] is not None else np.nan for r in line])
        np.savez_compressed(RUN / 'SELECTED_ALL33984_SOURCE_SCORES.npz', query_logit=z,
            query_available=available, prediction=d.decode(z, available), input_unit_index=units,
            raw_teacher_state=meta['typed_state'], external_target=meta['external_target'],
            split=meta['split'], domain=meta['domain'], source=meta['source'], variant=meta['variant'],
            query_xy_raw=query, L_raw=scale, old_accepted_triplets=meta['old_pairs'],
            new_selected_triplets=meta['new_pairs'], global_occurrence_index=np.arange(d.N))
        write(RUN / 'SOURCE_RESULTS.json', dict(completed=True, selected_update=selected,
            criterion_eight_VAL_cohorts_BCE=best, VAL_cohorts=parts, metrics=metrics,
            external_counts=meta['external_counts'], learned_self_classification=False,
            source_development_exposure=True, all2000_draws_sha256=draw_hash.hexdigest()))
        write(RUN / 'SOURCE_GATE_RESULTS.json', dict(completed=True, all_gates_pass=passed,
            gates=gates, current_inference_authorized=False))
        write(RUN / 'MODEL_FREEZE.json', dict(completed=True, selected_update=selected,
            checkpoint=bind(RUN / 'BEST_FROZEN_RGB_HEAD.pt'), encoder_weights=design['pretrained'],
            feature_receipt=entry['feature_receipt'], feature_pack=entry['feature_pack'],
            trainable_parameters=65793, encoder_trainable_parameters=0, threshold_logit=0,
            design=bind(DESIGN), trainer=bind(__file__), current_used_for_fit_or_selection=False,
            current_inference_authorized=False, PnP=0))
        write(RUN / 'FIT_COMPLETE.json', dict(completed=True, optimizer_updates=2000,
            selected_update=selected, criterion_eight_VAL_cohorts_BCE=best,
            all_source_gates_pass=passed, seconds=time.monotonic() - start,
            source_results=bind(RUN / 'SOURCE_RESULTS.json'), gates=bind(RUN / 'SOURCE_GATE_RESULTS.json'),
            scores=bind(RUN / 'SELECTED_ALL33984_SOURCE_SCORES.npz'),
            model_freeze=bind(RUN / 'MODEL_FREEZE.json'), current_or_PnP=0))
        print(json.dumps(read(RUN / 'FIT_COMPLETE.json')), flush=True)
    except BaseException:
        write(RUN / 'FAILED_EXECUTION.json', dict(optimizer_updates_completed=updates,
            traceback=traceback.format_exc(), restart_automatically=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', type=Path, required=True)
    run(parser.parse_args().entry)
