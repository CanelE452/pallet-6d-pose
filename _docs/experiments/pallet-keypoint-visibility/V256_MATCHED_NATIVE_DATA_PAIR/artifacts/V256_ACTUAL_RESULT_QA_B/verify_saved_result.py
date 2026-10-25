"""Root-authorized V256 saved-result audit; no model/encoder/feature forward."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import traceback

import numpy as np
import torch

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
TRAINER_SHA = '3ec2d316b6dfde1c6ab22f8fb8287fed8a873e29708fe8b44b8e32e8bd20c494'
POLICY_SHA = 'b9ce7e6ecbd6662aa386f7189eba3b25a4aa1d183b44c07ceb30ad84d7e4408a'
INITIAL_SHA = 'd1b8da9721a5492dcfb166cf78943eb5db18c5b14b8e8fed328ccab31c12648c'
CODE_REVIEW = {'path': str(R/'V256_TRAINER_CODE_REVIEW_B/REVIEW.json'),
 'sha256': '91784a2c183a5a34fb091e2a7981b0f63cc73b01446b80eb497faaa0a59ac618', 'size_bytes': 4558}
ARMS = ('CONTROL_ORIGINAL_FIT', 'TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN')
SAMPLE_KEYS = ('target', 'family', 'fit', 'available', 'row_id')
SHAPES = {'head.0.weight': (64, 1186), 'head.0.bias': (64,),
          'head.2.weight': (2, 64), 'head.2.bias': (2,)}
COLS = ('sample_index', 'row_id', 'family', 'role', 'target', 'available',
 'original_native_address', 'logit0', 'logit1', 'selected_family_E_logit', 'prediction')


def need(value, reason):
    if not bool(value): raise ValueError(reason)


def safe(path, fresh=False):
    s = str(path)
    need(not any(t in s.casefold() for t in ('v112', 'cache', '.pyc', 'current223',
         'publicheld', 'oldval', 'holdout', 'private_evaluation')), 'Lexical path rejection before FS')
    p = Path(s)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute exact artifact path')
    for q in [*reversed(p.parents), p]:
        try: mode = os.lstat(q).st_mode
        except FileNotFoundError: continue
        need(not stat.S_ISLNK(mode), 'Symlink artifact/ancestor')
    if fresh: need(not p.exists(), 'Fresh output; no retry')
    return p


def ident(b): return b['path'], b['sha256'], b['size_bytes']


def read_bound(b, white):
    need(ident(b) in white, 'Root exact saved-result read whitelist before file access')
    raw = safe(b['path']).read_bytes()
    need(len(raw) == b['size_bytes'] and hashlib.sha256(raw).hexdigest() == b['sha256'], 'Bound SHA/size')
    return raw


def bind(path):
    p = safe(path); raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def write_json(path, data):
    p = safe(path, True)
    with p.open('x') as f: f.write(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return bind(p)


def jread(b, white): return json.loads(read_bound(b, white))


def state_sha(state):
    h = hashlib.sha256()
    for name, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def check_state(state):
    need(set(state) == set(SHAPES), 'Only four existing head tensors; no LoRA/core')
    for k, shape in SHAPES.items():
        v = state[k]
        need(isinstance(v, torch.Tensor) and v.device.type == 'cpu' and v.dtype == torch.float32
             and tuple(v.shape) == shape and torch.isfinite(v).all(), 'Finite FP32 existing-head state')
    need(sum(v.numel() for v in state.values()) == 76098, 'Existing 76098 parameters')


def samples(b, white):
    with np.load(io.BytesIO(read_bound(b, white)), allow_pickle=False) as z:
        s = {k: z[k].copy() for k in SAMPLE_KEYS}
    n = len(s['target'])
    need(all(s[k].shape == (n,) for k in SAMPLE_KEYS), 'Five vector metadata members')
    need(s['target'].dtype == s['family'].dtype == np.int8 and np.isin(s['target'], [0, 1]).all(), 'Known D/E sample pack')
    need(np.isin(s['family'], [0, 1]).all() and s['fit'].dtype == s['available'].dtype == bool, 'Roles/families/support')
    need(s['row_id'].dtype.kind == 'U' and len(set(s['row_id'].tolist())) == n, 'Unique original row IDs')
    return s


def compare(actual, expected, where='root'):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), 'Exact reported keys '+where)
        for k in expected: compare(actual[k], expected[k], where+'.'+str(k))
    elif isinstance(expected, (list, tuple)):
        need(isinstance(actual, (list, tuple)) and len(actual) == len(expected), 'Reported shape '+where)
        for i, (a, b) in enumerate(zip(actual, expected)): compare(a, b, where+'.'+str(i))
    elif isinstance(expected, float):
        need(isinstance(actual, (int, float)) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), 'Reported ratio '+where)
    else: need(actual == expected, 'Reported value '+where)


def check_map(rows, new):
    known, keys, slots = {}, set(), {}
    for r in rows:
        key = (r['scene_id'], r['active_slot'])
        need(key not in keys and type(r['active_slot']) is int and 0 <= r['active_slot'] < 8, 'Unique all-node map')
        keys.add(key)
        slots.setdefault(r['scene_id'], set()).add(r['active_slot'])
        need(r['source_role'] == 'FIT' and r['fit'] is True and r['family_code'] == 0
             and r['variant'] in ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM'), 'New SOURCE FIT namespace')
        i = r['sample_index']
        if i is None:
            need(r['loss_mask'] is False, 'S/U/locality masked full row preserved')
            continue
        need(type(i) is int and i not in known and 0 <= i < len(new['target']), 'Known sample index join')
        need(r['row_id'] == str(new['row_id'][i]) and r['target'] == int(new['target'][i])
             and r['loss_mask'] is True and r['crop_available'] is True
             and r['original_query_available'] is True, 'Known label/support join')
        known[i] = r
    need(set(known) == set(range(len(new['target']))), 'Every new known row mapped')
    need((new['family'] == 0).all() and new['fit'].all() and new['available'].all(), 'Only known SYN FIT pack')
    need(all(js == set(range(8)) for js in slots.values()), 'Every scene retains all8 nodes')
    mapped = [known[i] for i in range(len(known))]
    census = {'all_new_scene_node_rows': len(rows), 'known_rows': len(known),
      'masked_rows_retained_in_full_map': len(rows)-len(known),
      'variant_all_rows': dict(Counter(r['variant'] for r in rows)),
      'variant_known_rows': dict(Counter(r['variant'] for r in known.values()))}
    return mapped, census


def expected_draws(s, eligible):
    parts, pools = [], []
    for f in (0, 1):
        for y in (0, 1):
            pool = np.flatnonzero(eligible & (s['family'] == f) & (s['target'] == y))
            need(len(pool) > 0, 'No empty class fallback')
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([17, f, y])))
            parts.append(generator.choice(pool, size=(1000, 8), replace=True)); pools.append(pool)
    return np.concatenate(parts, axis=1).astype(np.int64), pools


def draws_sha(draws, ids):
    h = hashlib.sha256(); h.update(draws.astype('<i8').tobytes())
    for row in ids[draws].flat:
        raw = str(row).encode(); h.update(len(raw).to_bytes(8, 'little')); h.update(raw)
    return h.hexdigest()


def check_ledger(rows, arm, draws, s, old_n):
    need(len(rows) == 1000, '1000 ledger rows')
    exposures = Counter()
    for step, (r, ids) in enumerate(zip(rows, draws), 1):
        expected = {'arm': arm, 'update': step, 'sample_indices': ids.tolist(),
          'row_ids': s['row_id'][ids].tolist(), 'source': ['ORIGINAL' if i < old_n else 'NATIVE_CHILD' for i in ids],
          'family': s['family'][ids].tolist(), 'target': s['target'][ids].tolist(), 'all_fit_available_known': True}
        compare(r, expected, 'ledger')
        need(s['fit'][ids].all() and s['available'][ids].all(), 'All actual exposures available FIT')
        for f in (0, 1):
            for y in (0, 1): need(int(((s['family'][ids] == f) & (s['target'][ids] == y)).sum()) == 8, '8D8E each family')
        exposures.update(expected['row_ids'])
    return dict(sorted(exposures.items()))


def prediction_rows(raw, s):
    reader = csv.DictReader(io.StringIO(raw.decode()))
    need(tuple(reader.fieldnames) == COLS, 'Exact saved prediction schema')
    rows = list(reader); need(len(rows) == len(s['target']), 'Fixed known-row denominator')
    for i, r in enumerate(rows):
        r['sample_index'] = int(r['sample_index']); r['family'] = int(r['family']); r['target'] = int(r['target'])
        need(r['available'] in ('True', 'False'), 'Boolean availability CSV')
        r['available'] = r['available'] == 'True'
        need(r['sample_index'] == i and r['row_id'] == str(s['row_id'][i]) and r['family'] == int(s['family'][i])
             and r['target'] == int(s['target'][i]) and r['available'] == bool(s['available'][i])
             and r['role'] == ('FIT' if s['fit'][i] else 'DEV'), 'Original row/role/target unchanged')
        for k in ('logit0', 'logit1', 'selected_family_E_logit'):
            r[k] = float(r[k]); need(math.isfinite(r[k]), 'Finite logits')
        need(r['selected_family_E_logit'] == r['logit'+str(r['family'])], 'Selected semantic family only')
        need(r['prediction'] == (('E' if r['selected_family_E_logit'] >= 0 else 'D') if r['available'] else 'U'), 'Raw threshold0 with input U')
        native = r['original_native_address']
        r['original_native_address'] = int(native) if native else None
        need((r['family'] == 0 and r['original_native_address'] in range(8)) or
             (r['family'] == 1 and r['original_native_address'] is None), 'Separate native address/YCB UNKNOWN')
    return rows


def metrics(rows):
    result = {}
    for f in (0, 1):
        for role in ('FIT', 'DEV'):
            subset = [r for r in rows if r['family'] == f and r['role'] == role and r['target'] in (0, 1)]
            cm = [[sum(r['target'] == y and r['prediction'] == p for r in subset) for p in ('D', 'E', 'U')] for y in (0, 1)]
            item = {'rows': len(subset), 'CM_D_E_by_D_E_U': cm, 'original_input_U': sum(not r['available'] for r in subset)}
            for name, k in (('D', 0), ('E', 1)):
                truth, predicted, tp = sum(cm[k]), cm[0][k]+cm[1][k], cm[k][k]
                item[name] = {'truth': truth, 'predicted': predicted, 'precision': tp/predicted if predicted else None,
                              'recall': tp/truth if truth else None}
            result[f'family{f}_{role}'] = item
    return result


def paired(control, treatment):
    result = {}
    for f in (0, 1):
        for role in ('FIT', 'DEV'):
            pairs = [(a, b) for a, b in zip(control, treatment) if a['family'] == f and a['role'] == role]
            recovered = damaged = 0
            for a, b in pairs:
                need(all(a[k] == b[k] for k in ('row_id', 'target', 'family', 'role', 'available', 'original_native_address')), 'Same evaluation rows/native fields')
                truth = ('D', 'E')[a['target']]
                recovered += a['prediction'] != truth and b['prediction'] == truth
                damaged += a['prediction'] == truth and b['prediction'] != truth
            result[f'family{f}_{role}'] = {'rows': len(pairs), 'recovered': recovered, 'damaged': damaged, 'net': recovered-damaged}
    return result


def child_stats(rows, mapped):
    before, groups = {}, {}
    for r, m in zip(rows, mapped):
        key = (m['base_index'], m['active_slot'])
        if m['variant'] == 'BEFORE': before[key] = r
        elif m['active_slot'] == m['intervention_target_native_slot']:
            groups.setdefault(key, {})[m['variant']] = r
    complete = correct = 0
    for key, roles in groups.items():
        four = dict(roles)
        if key in before: four['BEFORE'] = before[key]
        if set(four) == {'BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM'}:
            complete += 1
            correct += all(r['prediction'] == ('D', 'E')[r['target']] for r in four.values())
    return {'fitting_only_not_heldout': True,
      'variant_metrics': {v: metrics([r for r, m in zip(rows, mapped) if m['variant'] == v]) for v in ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')},
      'selected_intervention_k_four_scene_known_groups': complete,
      'four_scene_all_target_predictions_correct': correct, 'incomplete_known_groups': len(groups)-complete}


def run(entry_path):
    e = json.loads(safe(entry_path).read_bytes())
    need(e.get('approved_actual_saved_result_QA') is True, 'Root late actual saved-result QA entry required before payload access')
    white = {ident(b) for b in e['read_whitelist']}
    need(bind(__file__) == e['code'], 'Pinned QA CODE')
    # The exact Root scope is a direct metadata authority. Do not place its own
    # binding in entry.read_whitelist: that would create a circular SHA contract.
    scope = jread(e['root_scope'], white | {ident(e['root_scope'])})
    need(scope.get('approved_actual_saved_result_QA') is True and scope['entry_contract'] == {k: v for k, v in e.items() if k != 'root_scope'}, 'Exact Root QA scope')
    complete = jread(e['training_complete'], white)
    need(complete.get('completed') is True and complete.get('PASS') is True and complete['code']['sha256'] == TRAINER_SHA, 'Closed fixed trainer COMPLETE')
    review = jread(CODE_REVIEW, white); need(review['completed'] and review['PASS'], 'Prior sealed independent CODE QA')
    fit_entry = jread(complete['entry'], white)
    need(fit_entry['approved_actual_matched_data_fit'] is True and fit_entry['code'] == complete['code'], 'Original exact actual fit authorization')
    policy = jread(complete['policy'], white); need(complete['policy']['sha256'] == POLICY_SHA and policy['optimizer']['updates'] == 1000, 'Frozen scientific policy')
    out = safe(e['output_directory'], True); out.mkdir()
    try:
        old = samples(complete['inputs']['original_samples'], white)
        new = samples(complete['inputs']['new_samples'], white)
        old_complete = jread(complete['original_feature_complete'], white)
        new_complete = jread(complete['new_feature_complete'], white)
        packet_qa = jread(complete['new_feature_QA'], white)
        need(old_complete['completed'] is True and old_complete['PASS'] is True and
             new_complete['completed'] is True and new_complete['PASS'] is True and
             packet_qa['completed'] is True and packet_qa['PASS'] is True and
             packet_qa['feature_complete'] == complete['new_feature_complete'], 'Actual upstream packet/independent QA closure')
        for c, prefix in ((old_complete, 'original'), (new_complete, 'new')):
            need(c['features']['ORIGINAL_ALL_FEATURES.npy'] == complete['inputs'][prefix+'_features']
                 and c['samples_npz'] == complete['inputs'][prefix+'_samples'], 'Feature/sample metadata pointers unchanged; no feature-body read')
        need(new_complete['sample_map'] == complete['inputs']['new_sample_map'], 'Exact upstream full row map')
        fullmap = [json.loads(line) for line in read_bound(complete['inputs']['new_sample_map'], white).splitlines() if line]
        mapped, census = check_map(fullmap, new)
        need(new_complete['counts']['all_new_native_rows'] == len(fullmap)
             and new_complete['counts']['known_rows'] == len(new['target'])
             and new_complete['counts']['masked_rows'] == len(fullmap)-len(new['target']), 'Full map/sample/census retention')
        child = np.asarray([i for i, r in enumerate(mapped) if r['variant'] != 'BEFORE'], np.int64)
        need(len(child) > 0, 'Known nonBEFORE pool nonempty')
        combined = {k: np.concatenate((old[k], new[k][child])) for k in SAMPLE_KEYS}
        old_n = len(old['target']); old_fit = old['fit'] & old['available']
        masks = {ARMS[0]: np.r_[old_fit, np.zeros(len(child), bool)], ARMS[1]: np.r_[old_fit, np.ones(len(child), bool)]}
        draws, pools = {}, {}
        for arm in ARMS: draws[arm], pools[arm] = expected_draws(combined, masks[arm])
        need(np.array_equal(draws[ARMS[0]][:, 16:], draws[ARMS[1]][:, 16:]) and all(np.array_equal(a,b) for a,b in zip(pools[ARMS[0]][2:],pools[ARMS[1]][2:])), 'All YCB pools/draws byteexact')
        hashes = {a: draws_sha(draws[a], combined['row_id']) for a in ARMS}
        exposure = {}; checkpoints = {}; checkpoint_report = {}
        for arm in ARMS:
            ledger = [json.loads(x) for x in read_bound(complete['draw_ledgers'][arm], white).splitlines() if x]
            exposure[arm] = check_ledger(ledger, arm, draws[arm], combined, old_n)
            logs = [json.loads(x) for x in read_bound(complete['logs'][arm], white).splitlines() if x]
            need(len(logs) == 1000, '1000 optimizer log rows')
            for step, (log, ids) in enumerate(zip(logs, draws[arm]), 1):
                need(log['update'] == step and log['sample_indices'] == ids.tolist() and log['family_counts'] == [16,16] and log['all_fit_available_known'] is True, 'Actual log agrees with prospective draws')
                a,b,c = [float(log[k]) for k in ('family0_BCE','family1_BCE','total_loss')]
                need(all(math.isfinite(v) and v >= 0 for v in (a,b,c)) and math.isclose(c,.5*(a+b),rel_tol=2e-7,abs_tol=2e-7), 'Finite family balanced loss arithmetic')
            cp = torch.load(io.BytesIO(read_bound(complete['checkpoints'][arm], white)), map_location='cpu', weights_only=True)
            need(cp['arm'] == arm and cp['updates'] == 1000 and cp['head_parameters'] == 76098 and cp['use_lora'] is False, 'Actual fixed existing-head checkpoint')
            initial, final = cp['initial_head_state_dict'], cp['head_state_dict']
            check_state(initial); check_state(final)
            need(state_sha(initial) == cp['initial_head_sha256'] == INITIAL_SHA and state_sha(final) == cp['final_head_sha256'] != INITIAL_SHA, 'Identical fresh init and genuinely changed finite final head')
            optim = cp['optimizer_state_dict']; need(len(optim['param_groups']) == 1 and len(optim['state']) == 4, 'Independent four-state AdamW only')
            group = optim['param_groups'][0]
            need(len(group['params']) == 4 and set(group['params']) == set(optim['state']) and group['lr'] == .001 and group['weight_decay'] == .0001, 'Fixed optimizer HPS')
            for key, pid in zip(SHAPES,group['params']):
                state = optim['state'][pid]
                need(float(state['step']) == 1000, 'Each optimizer state updated1000')
                for moment in ('exp_avg','exp_avg_sq'):
                    need(tuple(state[moment].shape) == SHAPES[key] and torch.isfinite(state[moment]).all(), 'Finite optimizer moments exact tensor shape')
                need((state['exp_avg_sq'] >= 0).all(), 'Nonnegative second moments')
            expected_provenance = {k: complete[k] for k in ('entry','policy','code','crop_code',
               'original_feature_complete','new_feature_complete','new_feature_QA')}
            expected_provenance.update(draw_ledger=complete['draw_ledgers'][arm], draws_sha256=hashes[arm])
            compare(cp['provenance'], expected_provenance, 'checkpointprovenance')
            checkpoints[arm] = cp
            checkpoint_report[arm] = {'initial_sha256': INITIAL_SHA, 'final_sha256':state_sha(final),'optimizer_states':4,'steps_each':1000,
              'changed_scalar_parameters': sum(int((final[k] != initial[k]).sum()) for k in SHAPES)}
        root = safe(e['training_complete']['path']).parent
        def sibling(name):
            candidates = [b for b in e['read_whitelist'] if b['path'] == str(root/name)]
            need(len(candidates) == 1, 'Explicit sibling binding, no filesystem discovery'); return candidates[0]
        compare(jread(sibling('ACTUAL_ROW_EXPOSURES.json'),white), exposure, 'actualexposure')
        admission = jread(complete['fit_admission'],white)
        compare(admission['new_map_census'],census,'newmapcensus'); compare(admission['draw_hashes'],hashes,'drawhashes')
        need(admission['original_count'] == old_n and admission['original_FIT_available_known'] == int(old_fit.sum())
             and admission['new_known_packet'] == len(new['target']) and admission['new_nonBEFORE_all_admitted_children'] == len(child)
             and admission['new_BEFORE_excluded_from_augmentation'] == len(new['target'])-len(child)
             and admission['YCB_pools_and_all_draws_byteexact'] is True and admission['initial_state_sha256'] == INITIAL_SHA
             and admission['numeric_features_before_freeze_only_FIT'] is True and admission['DEV_prediction_calls_before_freeze'] == 0, 'Admission counts/claims')
        freeze = jread(complete['freeze'],white)
        need(freeze['completed'] is True and freeze['PASS'] is True and freeze['same_initial_state_sha256'] == INITIAL_SHA
             and freeze['DEV_prediction_calls_before_both_finals'] == 0 and freeze['old_checkpoint_warmstart'] is False
             and freeze['head_parameters_both'] == 76098 and freeze['numeric_feature_rows_before_freeze_only_available_known_FIT'] is True,'Persisted frozen barrier claims')
        compare(freeze['draw_ledgers'],complete['draw_ledgers'],'freezedraws'); compare(freeze['draw_hashes'],hashes,'freezedrawhashes')
        for arm in ARMS:
            f = freeze['finals'][arm]
            need(f['checkpoint'] == complete['checkpoints'][arm] and f['log'] == complete['logs'][arm]
                 and f['final_state_sha256'] == checkpoint_report[arm]['final_sha256'] and f['updates'] == 1000
                 and f['optimizer_states'] == 4 and freeze['new_optimizer_updates'][arm] == 1000,'Both final CP/log/state bindings')
        original = {a: prediction_rows(read_bound(complete['outputs'][a],white),old) for a in ARMS}
        children = {a: prediction_rows(read_bound(complete['child_fit_outputs'][a],white),new) for a in ARMS}
        summary = jread(complete['summary'],white)
        expected_metrics = {a:metrics(original[a]) for a in ARMS}
        compare(summary['family_metrics_original'],expected_metrics,'familyCMPRU')
        recalls = {a:[expected_metrics[a][f'family{f}_DEV'][c]['recall'] for f in (0,1) for c in ('D','E')] for a in ARMS}
        compare(summary['DEV_recalls_D_E_by_SYN_YCB'],recalls,'DEVrecalls')
        compare(summary['DEV_mean_balanced_recall'],{a:None if any(v is None for v in rr) else sum(rr)/4 for a,rr in recalls.items()},'meanBR')
        worst = None if any(v is None for rr in recalls.values() for v in rr) else max(a-b for a,b in zip(recalls[ARMS[0]],recalls[ARMS[1]]))
        compare(summary['worst_DEV_recall_drop'],worst,'worstdrop')
        compare(summary['paired_original_changes'],paired(original[ARMS[0]],original[ARMS[1]]),'pairedchanges')
        compare(summary['original_native_address_metrics'],{a:{str(k):metrics([r for r in original[a] if r['original_native_address']==k]) for k in range(8)} for a in ARMS},'nativeaddress')
        compare(summary['new_child_fit_diagnostic'],{a:child_stats(children[a],mapped) for a in ARMS},'childfit')
        need(summary['threshold'] == 0 and summary['head_parameters_both'] == 76098 and summary['new_updates_each'] == 1000
             and summary['same_initial_state_sha256'] == INITIAL_SHA and summary['model_adopted'] is False
             and summary['automatic_current_held_admission'] is False and summary['source_DEV_reused_exposed_not_fresh_blind'] is True
             and summary['no_empty_class_success_fill'] is True and summary['arms'] == list(ARMS)
             and summary['cluster_uncertainty']['computed'] is False,'Report scope/fixed recipe')
        need(complete['selected_arm'] is None and complete['current_or_held_admission'] is False
             and complete['initial_state_sha256'] == INITIAL_SHA
             and complete['head_parameters'] == {a:76098 for a in ARMS},'No model/current admission; fixed parameters')
        manifest = jread(complete['manifest'],white)
        for b in manifest['files']: read_bound(b,white)
        report = {'completed':True,'PASS':True,'training_complete':e['training_complete'],'QA_code':e['code'],'root_scope':e['root_scope'],
          'fixed_CODE_review':CODE_REVIEW,'new_map_census':census,'original_known_rows':old_n,
          'original_DEV_known_rows':int((~old['fit']).sum()),'original_U_known_rows':int((~old['available']).sum()),
          'new_nonBEFORE_known_FIT_pool':len(child),'YCB1000_draws_byteexact':True,'checkpoints':checkpoint_report,
          'family_metrics_original':expected_metrics,'DEV_recalls_D_E_by_SYN_YCB':recalls,
          'paired_original_changes':summary['paired_original_changes'],'all_reported_numeric_reductions_match':True,
          'barrier_order_authority':{'sealed_CODE_has_bothfinal_persist_before_first_DEV':True,'exact_CP_log_draw_barrier_artifacts_verified':True,
           'independent_live_execution_trace_observed':False,'artifact_timestamps_used_as_order_proof':False},
          'data_access':{'own_finalCP_CPU_weights_only_decode':2,'five_original_sample_metadata_members_including_DEV_loaded':True,
           'new_full_teacher_derived_map_read':True,'feature_numeric_or_pixel_or_encoder_or_rawGT_current_held_reads':0,'new_model_forward_or_optimizer_updates':0},
          'limits':['CP/log/ledger/CSV consistency plus sealed-code ordering is verified; saved retrospective claims alone do not independently observe execution chronology.',
            'CSV logits are not neural-forward replayed by this audit; no source feature vectors are read. Native address tables use the saved CSV namespace, not physical-native gold.',
            'New child results are SOURCE FIT diagnostics; reused original DEV is not fresh blind evidence. No recipe, threshold, model or deployment choice is made.']}
        receipt = write_json(out/'REVIEW.json',report)
        print(json.dumps({'completed':True,'PASS':True,'REVIEW':receipt}),flush=True)
        return report
    except BaseException:
        write_json(out/'FAILED_REVIEW.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--entry',required=True)
    run(parser.parse_args().entry)
