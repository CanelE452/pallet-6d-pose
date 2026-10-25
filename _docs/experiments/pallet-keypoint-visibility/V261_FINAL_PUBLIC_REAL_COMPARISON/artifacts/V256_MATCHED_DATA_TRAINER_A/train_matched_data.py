"""V256 fixed CropFamilyHead data-only comparison; no encoder or warm checkpoint."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import traceback
import types
from collections import Counter
import numpy as np
import torch

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
ROOT_POLICY = {'path': str(R/'V256_ROOT_NATIVE_DATA_FIT/PROSPECTIVE_DATA_ONLY_PAIR_POLICY.json'),
               'sha256': 'b9ce7e6ecbd6662aa386f7189eba3b25a4aa1d183b44c07ceb30ad84d7e4408a', 'size_bytes': 4405}
CROP_CODE = {'path': str(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py'),
             'sha256': '44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88', 'size_bytes': 4262}
CONTROL = 'CONTROL_ORIGINAL_FIT'
TREAT = 'TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN'
STEPS = 1000
COLS = ['sample_index', 'row_id', 'family', 'role', 'target', 'available',
        'original_native_address', 'logit0', 'logit1', 'selected_family_E_logit', 'prediction']


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    s = str(path)
    need(not any(t in s.casefold() for t in ('v112', 'cache', '.pyc', 'current223', 'publicheld',
         'oldval', 'holdout', 'private_evaluation')), 'Forbidden path before FS')
    p = Path(s)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute path')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact artifact missing')
            continue
        need(not stat.S_ISLNK(mode), 'No symlink artifact')
    return p


def read_bound(b):
    raw = safe(b['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == b['sha256'] and len(raw) == b['size_bytes'], 'Bound SHA/size')
    return raw


def binding(path):
    raw = safe(path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def write_json(path, value):
    p = safe(path, False)
    need(not p.exists(), 'Fresh output')
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return binding(p)


def load_crop():
    raw = read_bound(CROP_CODE)
    m = types.ModuleType('v256_frozen_crop_44f54')
    m.__file__ = CROP_CODE['path']
    m.__package__ = None
    exec(compile(raw, m.__file__, 'exec'), m.__dict__)
    return m


def configure_runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(a.dtype).encode())
        digest.update(json.dumps(list(a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def validate_samples(s):
    required = ('target', 'family', 'fit', 'available', 'row_id')
    need(set(required) <= set(s), 'Five original semantic metadata fields')
    n = len(s['target'])
    need(all(s[k].shape == (n,) for k in required), 'Dynamic sample vectors')
    need(s['target'].dtype == np.int8 and np.isin(s['target'], [-1, 0, 1]).all(), 'D0 E1 Uminus1')
    need(s['family'].dtype == np.int8 and np.isin(s['family'], [0, 1]).all(), 'Separate SYN/YCB namespaces')
    need(s['fit'].dtype == bool and s['available'].dtype == bool, 'Original bool roles/support')
    need(s['row_id'].dtype.kind == 'U' and len(set(s['row_id'].tolist())) == n, 'Unique Unicode row IDs')
    return n


def matched_draws(samples, eligible):
    """Four independent streams; same YCB pool/order implies exact same YCB draws."""
    n = validate_samples(samples)
    need(eligible.dtype == bool and eligible.shape == (n,), 'Explicit eligible mask')
    need((~eligible | (samples['fit'] & samples['available'] & (samples['target'] >= 0))).all(),
         'Only available known SOURCE FIT enters an optimizer')
    chunks, pools = [], []
    for f in (0, 1):
        for y in (0, 1):
            pool = np.flatnonzero(eligible & (samples['family'] == f) & (samples['target'] == y))
            need(len(pool) > 0, 'Required FIT class pool nonempty; no fallback')
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([17, f, y])))
            chunks.append(rng.choice(pool, size=(STEPS, 8), replace=True))
            pools.append(pool)
    draws = np.concatenate(chunks, axis=1).astype(np.int64)
    need(draws.shape == (STEPS, 32) and eligible[draws].all(), 'Fixed1000x32 FIT draws')
    return draws, pools


def draw_sha(draws, row_ids):
    h = hashlib.sha256()
    h.update(np.asarray(draws, dtype='<i8').tobytes())
    for row in row_ids[draws].reshape(-1):
        b = str(row).encode('utf-8')
        h.update(len(b).to_bytes(8, 'little'))
        h.update(b)
    return h.hexdigest()


def write_draw_ledger(path, arm, draws, samples, original_n):
    exposures = Counter()
    with safe(path, False).open('x', encoding='utf-8') as f:
        for step, ids in enumerate(draws, 1):
            row_ids = samples['row_id'][ids].tolist()
            exposures.update(row_ids)
            f.write(json.dumps({'arm': arm, 'update': step, 'sample_indices': ids.tolist(),
                                'row_ids': row_ids, 'source': ['ORIGINAL' if i < original_n else 'NATIVE_CHILD' for i in ids],
                                'family': samples['family'][ids].tolist(), 'target': samples['target'][ids].tolist(),
                                'all_fit_available_known': True}, sort_keys=True, allow_nan=False)+'\n')
    return binding(path), dict(sorted(exposures.items()))


def fit_arm(arm, head, crop_module, samples, fit_features, fit_map, eligible, draws,
            device, output, initial_state, provenance):
    need(state_sha(head.state_dict()) == state_sha(initial_state), 'Same exact fresh initial head')
    need(sum(p.numel() for p in head.parameters()) == 76098 and len(list(head.parameters())) == 4,
         'Original CropFamilyHead76098 only')
    need(all(p.dtype == torch.float32 for p in head.parameters()), 'FP32 head only')
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    log_path = output/(arm+'_TRAIN1000.jsonl')
    head.train()
    with safe(log_path, False).open('x', encoding='utf-8') as log:
        for step, ids in enumerate(draws, 1):
            need(eligible[ids].all() and (fit_map[ids] >= 0).all(), 'FIT-only numeric feature fetch')
            x = fit_features[torch.as_tensor(fit_map[ids], device=device)]
            y = torch.as_tensor(samples['target'][ids], device=device)
            f = torch.as_tensor(samples['family'][ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = crop_module.family_loss(head(x), y, f)
            need(torch.isfinite(loss), 'Finite fixed family loss')
            loss.backward()
            optimizer.step()
            log.write(json.dumps({'update': step, 'sample_indices': ids.tolist(),
                                  'family0_BCE': float(parts[0].detach()), 'family1_BCE': float(parts[1].detach()),
                                  'total_loss': float(loss.detach()), 'family_counts': [16, 16],
                                  'all_fit_available_known': True}, allow_nan=False)+'\n')
            if step % 100 == 0:
                print(json.dumps({'stage': arm, 'updates': step, 'fixed_final': STEPS}), flush=True)
    need(len(optimizer.state) == 4 and all(float(v['step']) == STEPS for v in optimizer.state.values()),
         'All four AdamW states1000')
    need(all(torch.isfinite(p).all() for p in head.parameters()), 'Finite final head')
    need(all(torch.isfinite(x).all() for v in optimizer.state.values() for x in v.values() if isinstance(x, torch.Tensor)),
         'Finite final optimizer')
    cp_path = output/(arm+'_FINAL1000.pt')
    with safe(cp_path, False).open('xb') as f:
        torch.save({'arm': arm, 'updates': STEPS, 'head_parameters': 76098, 'use_lora': False,
                    'head_state_dict': cpu_copy(head.state_dict()), 'initial_head_state_dict': cpu_copy(initial_state),
                    'initial_head_sha256': state_sha(initial_state), 'final_head_sha256': state_sha(head.state_dict()),
                    'optimizer_state_dict': cpu_copy(optimizer.state_dict()), 'provenance': provenance}, f)
    head.requires_grad_(False)
    head.eval()
    return {'checkpoint': binding(cp_path), 'log': binding(log_path),
            'final_state_sha256': state_sha(head.state_dict()), 'updates': STEPS, 'optimizer_states': 4}


def scalar_metrics(rows):
    """Fixed threshold0; absent classes remain null, never synthetic success."""
    out = {}
    for family in (0, 1):
        for role in ('FIT', 'DEV'):
            rr = [r for r in rows if r['family'] == family and r['role'] == role and r['target'] in (0, 1)]
            cm = [[sum(r['target'] == y and r['prediction'] == p for r in rr) for p in ('D', 'E', 'U')] for y in (0, 1)]
            item = {'rows': len(rr), 'CM_D_E_by_D_E_U': cm, 'original_input_U': sum(not r['available'] for r in rr)}
            for name, k in (('D', 0), ('E', 1)):
                truth, pred, tp = sum(cm[k]), cm[0][k]+cm[1][k], cm[k][k]
                item[name] = {'truth': truth, 'predicted': pred, 'precision': tp/pred if pred else None,
                              'recall': tp/truth if truth else None}
            out[f'family{family}_{role}'] = item
    return out


def prediction_rows(head, features, samples, device):
    """Caller must first persist BOTH_FINAL1000 freeze. No train/normalizer here."""
    n = validate_samples(samples)
    need(features.shape == (n, 1186) and features.dtype == np.float32, 'Fixed feature1186')
    rows = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, n, 64):
            end = min(start+64, n)
            x = np.array(features[start:end], copy=True)
            need(np.isfinite(x).all(), 'Finite saved source descriptors')
            logits = head(torch.as_tensor(x, device=device)).cpu().numpy()
            need(np.isfinite(logits).all(), 'Finite fixed raw logits')
            for z, i in enumerate(range(start, end)):
                family = int(samples['family'][i])
                cond = x[z, -32:]
                if family == 0:
                    need(np.isin(cond[:8], [0, 1]).all() and cond[:8].sum() == 1, 'Original native address onehot')
                    native = int(np.argmax(cond[:8]))
                else:
                    need(np.array_equal(cond, np.zeros(32, np.float32)), 'YCB nativeUNKNOWN zero32')
                    native = None
                val = float(logits[z, family])
                avail = bool(samples['available'][i])
                rows.append({'sample_index': i, 'row_id': str(samples['row_id'][i]), 'family': family,
                             'role': 'FIT' if samples['fit'][i] else 'DEV', 'target': int(samples['target'][i]),
                             'available': avail, 'original_native_address': native,
                             'logit0': float(logits[z, 0]), 'logit1': float(logits[z, 1]),
                             'selected_family_E_logit': val, 'prediction': ('E' if val >= 0 else 'D') if avail else 'U'})
    return rows


def write_rows(path, rows):
    with safe(path, False).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return binding(path)


def open_features(b, n):
    p = safe(b['path'])
    h, size = hashlib.sha256(), 0
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
            size += len(chunk)
    need(h.hexdigest() == b['sha256'] and size == b['size_bytes'], 'Opaque whole feature SHA/size')
    a = np.load(p, mmap_mode='r', allow_pickle=False)
    need(a.dtype == np.float32 and a.shape == (n, 1186), 'Dynamic saved FP32 feature1186')
    return a


def samples_from(b):
    with np.load(io.BytesIO(read_bound(b)), allow_pickle=False) as z:
        s = {k: z[k].copy() for k in ('target', 'family', 'fit', 'available', 'row_id')}
    validate_samples(s)
    return s


def parse_new_map(binding_, samples):
    rows = [json.loads(line) for line in read_bound(binding_).splitlines() if line]
    known, all_keys = {}, set()
    for row in rows:
        key = (row['scene_id'], int(row['active_slot']))
        need(key not in all_keys and row['source_role'] == 'FIT' and row['fit'] is True and
             row['family_code'] == 0 and row['variant'] in ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM'),
             'All new SourceFIT rows and unchanged variant namespace')
        all_keys.add(key)
        i = row['sample_index']
        if i is None:
            need(not row['loss_mask'], 'Masked full map retained outside known pack')
            continue
        need(isinstance(i, int) and 0 <= i < len(samples['target']) and i not in known, 'Unique known map index')
        need(row['row_id'] == str(samples['row_id'][i]) and row['target'] == int(samples['target'][i]) and
             row['loss_mask'] is True and row['crop_available'] is True and
             row['original_query_available'] is True, 'Exact known target/row/availability join')
        known[i] = row
    need(set(known) == set(range(len(samples['target']))), 'Every new known sample mapped once')
    need((samples['family'] == 0).all() and samples['fit'].all() and samples['available'].all() and
         np.isin(samples['target'], [0, 1]).all(), 'New known pack is SYN FIT only')
    return [known[i] for i in range(len(known))], {
        'all_new_scene_node_rows': len(rows), 'known_rows': len(known),
        'masked_rows_retained_in_full_map': len(rows)-len(known),
        'variant_all_rows': dict(Counter(r['variant'] for r in rows)),
        'variant_known_rows': dict(Counter(r['variant'] for r in known.values()))}


def paired_summary(a, b):
    need(len(a) == len(b) and all(x['row_id'] == y['row_id'] for x, y in zip(a, b)), 'Same original evaluation row join')
    out = {}
    for family in (0, 1):
        for role in ('FIT', 'DEV'):
            pairs = [(x, y) for x, y in zip(a, b) if x['family'] == family and x['role'] == role]
            recovered = damaged = 0
            for x, y in pairs:
                truth = 'D' if x['target'] == 0 else 'E'
                recovered += x['prediction'] != truth and y['prediction'] == truth
                damaged += x['prediction'] == truth and y['prediction'] != truth
            out[f'family{family}_{role}'] = {'rows': len(pairs), 'recovered': recovered,
                                           'damaged': damaged, 'net': recovered-damaged}
    return out


def native_metrics(rows):
    return {str(k): scalar_metrics([r for r in rows if r['original_native_address'] == k]) for k in range(8)}


def child_diagnostic(rows, mapped):
    # Descriptive fitting diagnostics only; no selection/admission based on this output.
    before, triplets = {}, {}
    for row, m in zip(rows, mapped):
        if m['variant'] == 'BEFORE':
            before[(m['base_index'], m['active_slot'])] = row
        elif m['active_slot'] == m['intervention_target_native_slot']:
            key = (m['base_index'], m['intervention_target_native_slot'])
            triplets.setdefault(key, {})[m['variant']] = row
    complete, correct = 0, 0
    for key, roles in triplets.items():
        roles = dict(roles)
        if key in before:
            roles['BEFORE'] = before[key]
        if set(roles) == {'BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM'}:
            complete += 1
            correct += all(r['prediction'] == ('D' if r['target'] == 0 else 'E') for r in roles.values())
    return {'fitting_only_not_heldout': True,
            'variant_metrics': {v: scalar_metrics([r for r, m in zip(rows, mapped) if m['variant'] == v])
                                for v in ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')},
            'selected_intervention_k_four_scene_known_groups': complete,
            'four_scene_all_target_predictions_correct': correct,
            'incomplete_known_groups': len(triplets)-complete}


def run(entry_path):
    entry_raw = safe(entry_path).read_bytes()
    entry = json.loads(entry_raw)
    need(entry.get('approved_actual_matched_data_fit') is True, 'Root actual fit entry required')
    own = binding(Path(__file__).absolute())
    need(entry['code'] == own and entry['root_policy'] == ROOT_POLICY and entry['crop_code'] == CROP_CODE,
         'Fixed producer/policy/head CODE bindings')
    policy = json.loads(read_bound(ROOT_POLICY))
    need(policy['optimizer']['updates'] == STEPS, 'Frozen1000 update policy')
    scope = json.loads(read_bound(entry['root_scope']))
    need(scope.get('approved_actual_matched_data_fit') is True and
         scope.get('entry_contract') == {k: v for k, v in entry.items() if k != 'root_scope'}, 'Root exact entry scope')
    need(entry.get('automatic_retry') is False and entry.get('current_or_held_access') is False and
         entry.get('encoder_or_old_checkpoint_read') is False, 'No retry/protected/encoder/old checkpoint')
    out = safe(entry['output_directory'], False)
    need(not out.exists(), 'Fresh output, no retry/resume')
    out.mkdir()
    completed_arms, files, stage = {}, [], 'metadata'
    try:
        old_complete = json.loads(read_bound(entry['original_feature_complete']))
        new_complete = json.loads(read_bound(entry['new_feature_complete']))
        need(old_complete.get('completed') is True and new_complete.get('completed') is True, 'Two closed feature packets')
        for complete_, fk, sk in [(old_complete, 'original_features', 'original_samples'),
                                  (new_complete, 'new_features', 'new_samples')]:
            need(complete_['features']['ORIGINAL_ALL_FEATURES.npy'] == entry[fk] and
                 complete_['samples_npz'] == entry[sk], 'Feature/samples exact COMPLETE closure')
        need(new_complete['sample_map'] == entry['new_sample_map'], 'New full masked map exact closure')
        qa = json.loads(read_bound(entry['new_feature_QA']))
        need(qa.get('completed') is True and qa.get('PASS') is True and
             qa.get('feature_complete') == entry['new_feature_complete'], 'Independent feature/target QA exact packet')
        configure_runtime()
        device = torch.device(entry['device'])
        old = samples_from(entry['original_samples'])
        new = samples_from(entry['new_samples'])
        old_n, new_n = len(old['target']), len(new['target'])
        new_map, census = parse_new_map(entry['new_sample_map'], new)
        child_ids = np.array([i for i, r in enumerate(new_map) if r['variant'] != 'BEFORE'], np.int64)
        need(len(child_ids) > 0, 'All nonBEFORE admitted children, no empty fallback')
        combined = {k: np.concatenate((old[k], new[k][child_ids])) for k in old}
        validate_samples(combined)
        old_fit = old['fit'] & old['available'] & (old['target'] >= 0)
        eligible_c = np.r_[old_fit, np.zeros(len(child_ids), bool)]
        eligible_t = np.r_[old_fit, np.ones(len(child_ids), bool)]
        draws_c, pools_c = matched_draws(combined, eligible_c)
        draws_t, pools_t = matched_draws(combined, eligible_t)
        need(np.array_equal(draws_c[:, 16:], draws_t[:, 16:]) and
             all(np.array_equal(a, b) for a, b in zip(pools_c[2:], pools_t[2:])), 'YCB pool and all draws byteexact')
        old_x = open_features(entry['original_features'], old_n)
        new_x = open_features(entry['new_features'], new_n)
        # Only old FIT and new FIT children are numerically fetched before final freeze.
        fit_ids = np.flatnonzero(eligible_t)
        fit_map = np.full(len(combined['target']), -1, np.int64)
        fit_map[fit_ids] = np.arange(len(fit_ids))
        fit_np = np.concatenate((np.array(old_x[np.flatnonzero(old_fit)], copy=True),
                                 np.array(new_x[child_ids], copy=True)))
        need(fit_np.dtype == np.float32 and fit_np.shape == (len(fit_ids), 1186) and np.isfinite(fit_np).all(),
             'Only finite FIT numeric feature matrix')
        fit_tensor = torch.as_tensor(fit_np, device=device)
        crop = load_crop()
        torch.manual_seed(17)
        initial_head = crop.CropFamilyHead()
        initial_state = cpu_copy(initial_head.state_dict())
        initial_sha = state_sha(initial_state)
        heads = {}
        for arm in (CONTROL, TREAT):
            heads[arm] = crop.CropFamilyHead().to(device)
            heads[arm].load_state_dict(initial_state, strict=True)
            need(state_sha(heads[arm].state_dict()) == initial_sha, 'Same fresh seed17 constructor state')
        ledger_bindings, exposures = {}, {}
        for arm, draws in [(CONTROL, draws_c), (TREAT, draws_t)]:
            lb, exposure = write_draw_ledger(out/(arm+'_DRAW1000.jsonl'), arm, draws, combined, old_n)
            ledger_bindings[arm], exposures[arm] = lb, exposure
            files.append(lb)
        exposure_binding = write_json(out/'ACTUAL_ROW_EXPOSURES.json', exposures)
        files.append(exposure_binding)
        draw_hashes = {CONTROL: draw_sha(draws_c, combined['row_id']), TREAT: draw_sha(draws_t, combined['row_id'])}
        admission = {'completed': True, 'policy': ROOT_POLICY, 'original_count': old_n,
                     'original_FIT_available_known': int(old_fit.sum()), 'new_known_packet': new_n,
                     'new_BEFORE_excluded_from_augmentation': int(new_n-len(child_ids)),
                     'new_nonBEFORE_all_admitted_children': len(child_ids), 'new_map_census': census,
                     'original_masked_node_census': old_complete.get('counts'),
                     'YCB_pools_and_all_draws_byteexact': True, 'initial_state_sha256': initial_sha,
                     'draw_hashes': draw_hashes, 'no_historical_draw_SHA_claim': True,
                     'numeric_features_before_freeze_only_FIT': True, 'DEV_prediction_calls_before_freeze': 0}
        ab = write_json(out/'FIT_ADMISSION_AND_DRAW_QA.json', admission)
        files.append(ab)
        provenance = {'entry': {'path': str(entry_path), 'sha256': hashlib.sha256(entry_raw).hexdigest(), 'size_bytes': len(entry_raw)},
                      'policy': ROOT_POLICY, 'code': own, 'crop_code': CROP_CODE,
                      'original_feature_complete': entry['original_feature_complete'],
                      'new_feature_complete': entry['new_feature_complete'], 'new_feature_QA': entry['new_feature_QA']}
        for arm, draws, eligible in [(CONTROL, draws_c, eligible_c), (TREAT, draws_t, eligible_t)]:
            stage = arm
            completed_arms[arm] = fit_arm(arm, heads[arm], crop, combined, fit_tensor, fit_map, eligible,
                                           draws, device, out, initial_state,
                                           dict(provenance, draws_sha256=draw_hashes[arm], draw_ledger=ledger_bindings[arm]))
            files.extend([completed_arms[arm]['checkpoint'], completed_arms[arm]['log']])
        need(set(completed_arms) == {CONTROL, TREAT}, 'Both final1000/logs before any DEV prediction')
        freeze = write_json(out/'BOTH_FINAL1000_FROZEN_BEFORE_ANY_SOURCE_DEV_PREDICTION.json', {
            'completed': True, 'PASS': True, 'finals': completed_arms, 'draw_ledgers': ledger_bindings,
            'draw_hashes': draw_hashes, 'same_initial_state_sha256': initial_sha,
            'new_optimizer_updates': {CONTROL: STEPS, TREAT: STEPS}, 'head_parameters_both': 76098,
            'DEV_prediction_calls_before_both_finals': 0, 'old_checkpoint_warmstart': False,
            'numeric_feature_rows_before_freeze_only_available_known_FIT': True})
        files.append(freeze)
        stage = 'postfreeze_saved_SOURCE_predictions'
        old_rows, outputs, child_outputs, child_stats = {}, {}, {}, {}
        for arm in (CONTROL, TREAT):
            old_rows[arm] = prediction_rows(heads[arm], old_x, old, device)
            outputs[arm] = write_rows(out/(arm+'_ORIGINAL_ALL_SOURCE_ROWS.csv'), old_rows[arm])
            files.append(outputs[arm])
            rr = prediction_rows(heads[arm], new_x, new, device)
            child_outputs[arm] = write_rows(out/(arm+'_NEW_KNOWN_SOURCE_FIT_ROWS.csv'), rr)
            files.append(child_outputs[arm])
            child_stats[arm] = child_diagnostic(rr, new_map)
            need(state_sha(heads[arm].state_dict()) == completed_arms[arm]['final_state_sha256'],
                 'Final saved head unchanged after all predictions')
        metrics = {arm: scalar_metrics(rows) for arm, rows in old_rows.items()}
        recalls = {arm: [metrics[arm][f'family{f}_DEV'][c]['recall'] for f in (0, 1) for c in ('D', 'E')]
                   for arm in (CONTROL, TREAT)}
        drops = None if any(v is None for a in recalls.values() for v in a) else [a-b for a, b in zip(recalls[CONTROL], recalls[TREAT])]
        summary = {'completed': True, 'arms': [CONTROL, TREAT], 'head_parameters_both': 76098,
                   'new_updates_each': STEPS, 'same_initial_state_sha256': initial_sha,
                   'family_metrics_original': metrics, 'DEV_recalls_D_E_by_SYN_YCB': recalls,
                   'DEV_mean_balanced_recall': {a: None if any(v is None for v in x) else sum(x)/4 for a, x in recalls.items()},
                   'worst_DEV_recall_drop': None if drops is None else max(drops),
                   'paired_original_changes': paired_summary(old_rows[CONTROL], old_rows[TREAT]),
                   'original_native_address_metrics': {a: native_metrics(rr) for a, rr in old_rows.items()},
                   'new_child_fit_diagnostic': child_stats, 'fit_admission': ab,
                   'cluster_uncertainty': {'computed': False, 'reason': 'Original group metadata not admitted in minimal directhead input closure; no independent-row interval claim.'},
                   'no_empty_class_success_fill': True, 'threshold': 0,
                   'automatic_current_held_admission': False, 'model_adopted': False,
                   'source_DEV_reused_exposed_not_fresh_blind': True}
        sb = write_json(out/'SUMMARY.json', summary)
        files.append(sb)
        manifest = write_json(out/'MANIFEST.json', {'files': files, 'treewalk': False})
        cb = write_json(out/'COMPLETE.json', {'completed': True, 'PASS': True, **provenance,
            'inputs': {k: entry[k] for k in ('original_features', 'original_samples', 'new_features', 'new_samples', 'new_sample_map')},
            'checkpoints': {a: x['checkpoint'] for a, x in completed_arms.items()},
            'logs': {a: x['log'] for a, x in completed_arms.items()}, 'draw_ledgers': ledger_bindings,
            'freeze': freeze, 'fit_admission': ab, 'outputs': outputs, 'child_fit_outputs': child_outputs,
            'summary': sb, 'manifest': manifest, 'head_parameters': {CONTROL: 76098, TREAT: 76098},
            'initial_state_sha256': initial_sha, 'current_or_held_admission': False, 'selected_arm': None})
        print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': cb}), flush=True)
        return cb
    except BaseException as exc:
        write_json(out/'FAILURE.json', {'completed': False, 'PASS': False, 'stage': stage,
                                      'error': str(exc), 'traceback': traceback.format_exc(),
                                      'completed_arms': completed_arms, 'written_files': files,
                                      'automatic_retry': False, 'DEV_prediction_calls_before_final_barrier': 0})
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', required=True)
    run(ap.parse_args().entry)
