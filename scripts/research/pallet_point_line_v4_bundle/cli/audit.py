"""Final audit: declared completion vs actual artifacts on disk."""
from __future__ import annotations
import json, sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from cli.common import ROOT, RESULTS, read, write, sha256, git  # noqa: E402

ARMS = ('P', 'S', 'H', 'HA')
SEEDS = (1, 2, 3)
REQUIRED = ['PURPOSE.md', 'SOURCE_REGISTRY.json', 'ADAPTER_MAPPING.md', 'DATA_CONTRACT.json',
            'PREFLIGHT.json', 'PROTOCOL_LOCK.json', 'NATIVE_ORACLE_AUDIT.json', 'EXPORT_COMPLETION.json',
            'BINDING_CHECK.json', 'COMPARISON_synth_val.json', 'VERDICT.json', 'AFFINE_PROBE.json']

protocol = read(RESULTS / 'PROTOCOL_LOCK.json')
steps = protocol['training']['steps']
cells = []
for arm in ARMS:
    for seed in SEEDS:
        d = RESULTS / 'heads' / f'{arm}_seed{seed}'
        done = read(d / 'COMPLETION.json')
        trace = sum(1 for _ in (d / 'TRACE.jsonl').open())
        cells.append(dict(cell=f'{arm}_seed{seed}', optimizer_steps=done['optimizer_steps'],
                          trace_lines=trace, budget_met=done['optimizer_steps'] == steps == trace,
                          checkpoint_sha_matches=done['checkpoint_sha256'] == sha256(d / 'checkpoint_final.pt'),
                          initial_state_sha256=read(d / 'START.json')['initial_state_sha256'],
                          final_state_sha256=done['final_state_sha256'], stage=done['stage']))
missing = [f for f in REQUIRED if not (RESULTS / f).exists()]
scores = [p.name for p in sorted((RESULTS / 'scores').glob('*.json'))]
evaluations = [p.name for p in sorted((RESULTS / 'evaluations').glob('*.json'))]
policies = [p.name for p in sorted((RESULTS / 'policies').glob('*.json'))]
core_now = {p.name: sha256(p) for p in sorted((KIT / 'pointline_v4').glob('*.py'))}
status = git('status', '--short')
touched = [line for line in status.splitlines()
           if not line.split(maxsplit=1)[-1].startswith(('data/pallet/results/pallet_point_line_v4',
                                                         'scripts/research/pallet_point_line_v4_bundle',
                                                         'data/pallet/results/pallet_dht_deepfield_v4',
                                                         '_docs/', 'challenge/'))]
audit = dict(schema='pointline_v4_audit_1',
             cells=cells, all_cells_at_budget=all(c['budget_met'] and c['checkpoint_sha_matches'] and c['stage'] == 'main' for c in cells),
             distinct_final_states=len({c['final_state_sha256'] for c in cells}),
             missing_required_artifacts=missing,
             score_files=len(scores), policy_files=len(policies), evaluation_files=len(evaluations),
             bundle_core_unchanged_since_lock=(core_now == protocol['core_source_sha256']),
             export_manifests_unchanged={role: sha256(RESULTS / 'export' / f'{role}.json') == protocol[f'{role}_manifest_sha256' if role != 'train' else 'train_manifest_sha256']
                                         for role in ('train', 'calibration', 'synth_val')},
             strict_p4_parity_PASS=read(RESULTS / 'EXPORT_COMPLETION.json')['strict_p4_parity_PASS'],
             preflight_all_PASS=read(RESULTS / 'PREFLIGHT.json')['all_gates_PASS'],
             binding_PASS=read(RESULTS / 'BINDING_CHECK.json')['PASS'],
             unrelated_working_tree_changes=touched,
             real_dev_export_created=(RESULTS / 'export' / 'real_dev.json').exists(),
             full_network_run_created=(RESULTS / 'full_network').exists(),
             repository_commit=git('rev-parse', 'HEAD'))
audit['PASS'] = bool(audit['all_cells_at_budget'] and not missing and audit['bundle_core_unchanged_since_lock']
                     and all(audit['export_manifests_unchanged'].values()) and audit['strict_p4_parity_PASS']
                     and audit['preflight_all_PASS'] and audit['binding_PASS']
                     and audit['distinct_final_states'] == 12 and len(scores) == 24 and len(evaluations) == 12
                     and len(policies) == 12 and not touched)
write(RESULTS / 'FINAL_AUDIT.json', audit)
print(json.dumps({k: v for k, v in audit.items() if k != 'cells'}, ensure_ascii=False, indent=1))
