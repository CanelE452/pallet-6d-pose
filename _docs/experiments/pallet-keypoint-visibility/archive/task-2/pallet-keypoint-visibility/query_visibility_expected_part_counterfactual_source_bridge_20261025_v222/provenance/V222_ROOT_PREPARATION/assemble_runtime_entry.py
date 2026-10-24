"""Bind a SOURCE-only V222 probe/fixed-fit entry from existing metadata, never payloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
S = P / 'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222'
H = P / 'publication_root_preparation/V222_PAIRED_BRIDGE_TRAINING_B_V2'
R = P / 'publication_root_preparation/V222_ROOT_PREPARATION'


def safe(f):
    f = Path(f)
    assert f.is_absolute() and '..' not in f.parts and 'v112' not in str(f).lower()
    for item in (*reversed(f.parents), f):
        assert not item.is_symlink()
    return f


def bind(f):
    f = safe(f)
    assert f.suffix in {'.json', '.jsonl', '.py', '.md'}, 'Metadata/code only in entry assembly'
    b = f.read_bytes()
    return dict(path=str(f), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


def read_binding(b):
    a = bind(b['path'])
    assert a['sha256'] == b['sha256']
    assert 'size_bytes' not in b or a['size_bytes'] == b['size_bytes']
    raw = safe(b['path']).read_text()
    return [json.loads(line) for line in raw.splitlines() if line.strip()] if Path(b['path']).suffix == '.jsonl' else json.loads(raw)


def write(f, value):
    with safe(f).open('x') as out:
        out.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    return bind(f)


def unique(bindings):
    out = {}
    for b in bindings:
        assert b is not None and 'v112' not in b['path'].lower()
        if b['path'] in out:
            assert out[b['path']]['sha256'] == b['sha256']
        out[b['path']] = b
    return list(out.values())


def main(fit=False):
    ready = read_binding(bind(H / 'FINAL_READY.json'))
    assert ready['code']['sha256'] == '1e88783241417b97554bdc5b33225e0d6b8c2ae937d0f0288575acd1c26b3957'
    assert ready['contract']['sha256'] == '40fb7d9e1994d7cd1aa30d063c19d785789062b7e6176af9cdb3ed5bde973c94'
    for key in ['code', 'contract', 'protocol']:
        assert bind(ready[key]['path']) == ready[key]
    qa_bind = bind(P / 'publication_root_preparation/V222_TRAINER_CODE_QA_A/REVIEW.json')
    assert qa_bind['sha256'] == '7961719bb76d2ec519bdd9556a27b7892fec42fc5127529f7cfd67ece7ebdd85'
    entry = read_binding(ready['false_entry'])
    reference = read_binding(entry['reference_C65_fit_complete'])
    old = read_binding(reference['entry'])
    old_recipe = read_binding(reference['root_recipe'])
    for key in ['strong_target_admission', 'weak_target_admission', 'lineage_assembly_QA',
                'old_caches', 'point_metadata7040', 'weak_rows360', 'new_DEV_component_ids']:
        entry[key] = old[key]
    entry['reference_C65_training_entry'] = reference['entry']
    entry['reference_C65_all2_freeze'] = reference['all2_freeze']
    entry['reference_C65_recipe'] = reference['root_recipe']
    entry['warmstart_checkpoint'] = reference['arms']['STRONG_PLUS_EXPECTED_PART_WEAK']['checkpoint']
    assert entry['warmstart_checkpoint']['sha256'] == '67a947a6ddc9fb1b59f88c6bc88ac13e4b18646ad5abb070989e2175078ad997'
    rows = read_binding(entry['weak_rows360'])
    natural_fit = [r['generated_input']['RGB96'] for r in rows if r['weak_loss_fit_candidate'] is True]
    natural_dev = [r['generated_input']['RGB96'] for r in rows if r['weak_loss_candidate'] is True and r['source_role'] == 'SOURCE_DEV']
    assert len(natural_fit) == 45 and len(natural_dev) == 12
    fit_manifest = read_binding(entry['CF_FIT_manifest'])
    dev_manifest = read_binding(entry['CF_DEV_manifest'])
    assert len(fit_manifest['exact_RGB96_whitelist']) == 117
    assert len(dev_manifest['exact_RGB96_whitelist']) == 27
    binary = [entry['warmstart_checkpoint'], *natural_fit, *fit_manifest['exact_RGB96_whitelist']]
    for domain, record in entry['old_caches'].items():
        binary += [record['inputs'], record['targets']]
        entry['metadata_whitelist'].append(record['metadata'])
    entry['fit_binary_whitelist'] = unique(binary)
    entry['postfreeze_binary_whitelist'] = unique(natural_dev + dev_manifest['exact_RGB96_whitelist'])
    assert not {b['path'] for b in entry['fit_binary_whitelist']} & {b['path'] for b in entry['postfreeze_binary_whitelist']}
    mode = 'FIXED_PAIRED_FIT' if fit else 'THROUGHPUT_PROBE'
    steps = 32
    probe_binding = None
    if fit:
        probe_binding = bind(S / 'throughput_probe_v1/PROBE_COMPLETE.json')
        probe = read_binding(probe_binding)
        assert probe['completed'] and probe['PASS'] and probe['checkpoint_saved'] is False
        assert probe['steps_each'] == 32 and probe['updates'] == 64
        import math
        steps = max(200, min(2000, 100 * math.floor((60 / max(probe['step_seconds_by_arm'].values())) / 100)))
        assert steps == probe['budget_by_prospective_rule']
    scope = write(S / ('ROOT_SCOPE_FIXED_FIT.json' if fit else 'ROOT_SCOPE_THROUGHPUT_PROBE.json'), {
        'completed': True, 'approved_source_pair_fit_or_probe': True, 'mode': mode,
        'same_C65_recipe_continuation_not_protected_result_selection': True,
        'new_source_CF_weak_admission': entry['CF_target_admission'], 'independent_code_QA': qa_bind,
        'protected_evaluation_payloads_or_outputs': False, 'new_PnP': False,
        'no_new_GT_pose_reference_ROI_mask_model_input': True,
        'both_final_models_retained_before_DEV_outputs': True,
        'disposable_probe_states_used_for_final_fit': False,
    })
    recipe = write(S / ('ROOT_FIXED_SOURCE_PAIR_RECIPE.json' if fit else 'ROOT_PROSPECTIVE_PROBE_RECIPE.json'), {
        'completed': True, 'approved_frozen_recipe': True,
        'budget_selection_basis': 'THROUGHPUT_ONLY_NO_SCORE_OR_ARM_SELECTION', 'mode': mode,
        'steps': steps, 'learning_rate': old_recipe['learning_rate'], 'weight_decay': old_recipe['weight_decay'],
        'natural_weak_weight': .25, 'CF_weak_weight': .25, 'CF_parents_per_step': 4,
        'CF_sampling': 'UNIFORM_WITH_REPLACEMENT', 'CF_parent_BCE_weights_BEFORE_COVER_SHAM': [.25, .5, .25],
        'old_batch': 192, 'strong_batch': 32, 'natural_batch': 16, 'CF_treatment_batch': 12,
        'threshold_logit': 0., 'new_hinge': False, 'new_backbone': False,
        'common_draw_seeds': [1810, 1811, 1812], 'independent_CF_draw_seed': 1813,
        'actual_probe_complete': probe_binding, 'early_stop_or_checkpoint_selection': False,
        'bounded_budget_rule': 'max(200,min(2000,100*floor((60/max(step_seconds_by_arm))/100)))',
    })
    entry.update(approved_source_pair_fit_or_probe=True, approved_load_exact_C65_into_both_arms=True,
                 mode=mode, root_scope=scope, root_recipe=recipe, throughput_probe_complete=probe_binding)
    protocol = read_binding(entry['protocol'])
    entry['output_directory'] = protocol['planned_outputs'][mode]
    metadata_keys = ['root_scope', 'root_recipe', 'strong_target_admission', 'weak_target_admission',
                     'lineage_assembly_QA', 'point_metadata7040', 'reference_C65_fit_complete',
                     'reference_C65_training_entry', 'reference_C65_all2_freeze', 'reference_C65_recipe',
                     'CF_attempts48']
    entry['metadata_whitelist'] += [entry[k] for k in metadata_keys] + [qa_bind]
    if probe_binding:
        entry['metadata_whitelist'].append(probe_binding)
    entry['metadata_whitelist'] = unique(entry['metadata_whitelist'])
    name = 'ROOT_ACTUAL_FIXED_SOURCE_PAIR_FIT_ENTRY.json' if fit else 'ROOT_ACTUAL_THROUGHPUT_PROBE_ENTRY.json'
    bound = write(S / name, entry)
    print(json.dumps({'completed': True, 'mode': mode, 'steps_each': steps, 'entry': bound,
                      'fit_binary_bindings': len(entry['fit_binary_whitelist']),
                      'postfreeze_binary_bindings': len(entry['postfreeze_binary_whitelist']),
                      'payloads_decoded_by_assembly': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', action='store_true')
    main(parser.parse_args().fit)
