"""Diagnose first32 SOURCE FIT descriptors; no head, optimizer or DEV outputs."""
from pathlib import Path
import hashlib
import json
import os
import types
import numpy as np
import torch
from torch.nn import functional as F

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V238_FEATURE_ROUTE_DIAG'


def safe(p):
    p = Path(p)
    assert not any(t in str(p).lower() for t in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'private_evaluation', 'oldval', 'v177', 'v178', 'source_holdout', 'v182'))
    assert not p.is_symlink()
    return p


def bind(p):
    p = safe(p); h = hashlib.sha256(); n = 0
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b); n += len(b)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': n}


def write(p, d):
    with safe(p).open('x') as f:
        json.dump(d, f, indent=2, ensure_ascii=False); f.write('\n')


def main():
    assert not H.exists(); H.mkdir()
    ep = R/'V238_ROOT_CONTEXT_RESIDUAL/ACTUAL_SOURCE_CONTEXT_RESIDUAL_FIT_ENTRY.json'
    assert bind(ep)['sha256'] == '08bd9fd1dd6afabf51d3b6ce2a443e2dce7d88d1485172865fc94903bf192e8b'
    e = json.loads(safe(ep).read_text())
    rows = [json.loads(x) for x in safe(e['sample_map']['path']).read_text().splitlines()]
    chosen = sorted([r for r in rows if r['sample_index'] is not None and r['fit'] and r['crop_available'] and r['target'] >= 0], key=lambda r: r['sample_index'])[:32]
    indices = np.asarray([r['sample_index'] for r in chosen], np.int64)
    gi = np.asarray([r['group_index'] for r in chosen], np.int64); si = np.asarray([r['active_slot'] for r in chosen], np.int64)
    failed = Path(e['output_directory'])
    scope = {'approved_SOURCE_FIT_numerical_diagnosis': True, 'entry': bind(ep), 'code': bind(__file__),
             'sample_indices': indices.tolist(), 'group_indices': gi.tolist(), 'node_indices': si.tolist(),
             'allowed': 'Same official frozen DINO/code as actual entry; original prefix full/centre views for these32 FIT only; NODE_RGB/VALID for these32 active nodes, stored NODE_CLS for these32 only. Whole listed-file hashes allowed opaque. No original image, GT, private/current/held or DEV numeric body.',
             'new_head_or_optimizer_or_threshold_change': False, 'automatic_training_restart': False,
             'frozen_tolerance_not_relaxed': 1e-6}
    write(H/'SCOPE_BEFORE_NUMERICAL_READ.json', scope)
    module = types.ModuleType('v238_diagnosis_trainer'); module.__file__ = e['code']['path']
    assert bind(e['code']['path'])['sha256'] == e['code']['sha256']
    exec(compile(safe(e['code']['path']).read_bytes(), e['code']['path'], 'exec'), module.__dict__)
    utility = module.load_code(e['utility'], e['code_whitelist'], 'v238_diag_utility', module.PINS['utility'])
    source, crop, runtime = utility.load_sources(e); source.configure_runtime()
    preprocess = utility.load_preprocess(e)
    model = source.load_backbone(e['model_loader_config'], 'cuda', treatment=False)
    with np.load(safe(e['samples_npz']['path']), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    prefix = np.load(safe(e['prefix_npy']['path']), mmap_mode='r', allow_pickle=False)
    store = source.PrefixViews(prefix, data)
    rgb = np.load(safe(e['node_rgb']['path']), mmap_mode='r', allow_pickle=False)
    valid = np.load(safe(e['node_valid']['path']), mmap_mode='r', allow_pickle=False)
    stored_nodes = np.load(safe(failed/'NODE_CLS.npy'), mmap_mode='r', allow_pickle=False)
    pixels = torch.from_numpy(np.array(rgb[gi, si], copy=True))
    support = torch.from_numpy(np.array(valid[gi, si, None], copy=True))
    views, fractions = preprocess(pixels, support); x = views[:32].to('cuda')
    desc = {}
    with torch.no_grad():
        old = source.features_for(indices, store, model, runtime, None, 'cuda', False)
        desc['old_prefix_fullcentre64'] = old[:, :384].cpu().numpy()
        desc['stored_new_node'] = np.array(stored_nodes[gi, si], copy=True)
        desc['fresh_fullforward32'] = F.normalize(model.forward_features(x)['x_norm_clstoken'], p=2, dim=1, eps=1e-12).cpu().numpy()
        fresh_prefix = runtime.extract_prefix(model, x, {'view_ids': ['diag'+str(int(i)) for i in indices]})['prefix_tokens']
        old_prefix = np.array(prefix[data['full_view'][indices]], copy=True)
        desc['fresh_prefix_finish32'] = F.normalize(runtime.last_block_features(model, fresh_prefix.to('cuda'), adapter_grad=False)['x_norm_clstoken'], p=2, dim=1, eps=1e-12).cpu().numpy()
        desc['cached_prefix_finish32'] = F.normalize(runtime.last_block_features(model, torch.from_numpy(old_prefix).to('cuda'), adapter_grad=False)['x_norm_clstoken'], p=2, dim=1, eps=1e-12).cpu().numpy()
        desc['fresh_fullforward1'] = F.normalize(model.forward_features(x[:1])['x_norm_clstoken'], p=2, dim=1, eps=1e-12).cpu().numpy()
        desc['cached_prefix_finish1'] = F.normalize(runtime.last_block_features(model, torch.from_numpy(old_prefix[:1]).to('cuda'), adapter_grad=False)['x_norm_clstoken'], p=2, dim=1, eps=1e-12).cpu().numpy()
    metrics = {}
    for name, values in desc.items():
        base = desc['old_prefix_fullcentre64'][:len(values)]
        delta = np.abs(values-base)
        metrics[name] = {'max_abs_vs_old': float(delta.max()), 'mean_abs_vs_old': float(delta.mean()),
                         'cosine_min_vs_old': float(np.sum(base*values, 1).min()),
                         'rows_exceeding_1e_6': int((delta.max(1) > 1e-6).sum()), 'rows': len(values)}
        np.save(safe(H/(name+'.npy')), values, allow_pickle=False)
    pd = np.abs(fresh_prefix.numpy()-old_prefix)
    result = {'completed': True, 'PASS': True, 'scope': bind(H/'SCOPE_BEFORE_NUMERICAL_READ.json'),
              'diagnostic_only_not_model_equivalence_PASS': True, 'samples': indices.tolist(), 'metrics': metrics,
              'prefix_token_max_abs': float(pd.max()), 'prefix_token_mean_abs': float(pd.mean()),
              'new_fullforward_vs_new_prefix_finish_max_abs': float(np.max(np.abs(desc['fresh_fullforward32']-desc['fresh_prefix_finish32']))),
              'stored_nodes_vs_fresh_full32_max_abs': float(np.max(np.abs(desc['stored_new_node']-desc['fresh_fullforward32']))),
              'head_or_optimizer_updates': 0, 'new_DEV_or_current_or_held_output': 0, 'threshold_or_tolerance_change': False}
    write(H/'DIAGNOSIS.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
