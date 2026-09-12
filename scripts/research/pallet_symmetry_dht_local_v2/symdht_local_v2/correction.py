"""One authorized correction reproduction; original v2 artifacts remain immutable.

Run from repository root: python -m
scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.correction prepare|train|finish
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset, collate, observation_batch
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256, canonical_sha
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from . import assessment, diagnostics, geometry, parity, runner

REPO = Path(__file__).resolve().parents[4]
PACKAGE = REPO / 'scripts/research/pallet_symmetry_dht_local_v2'
OLD = REPO / 'data/pallet/results/pallet_symmetry_dht_local_v2'
RAW = REPO / 'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction'
DOC = REPO / '_docs/experiments/pallet_symmetry_dht_local_v2_wls_correction'
EXPORT = REPO / 'data/pallet/results/pallet_symmetry_dht_local_v1/export'
NAMES = ['point'] + [f'{a}_seed{s}' for a in ('direct', 'hough') for s in (1, 2, 3)]


def preserved_hashes():
    paths = [REPO/'_docs/experiments/pallet_symmetry_dht_local_v1',
             REPO/'_docs/experiments/pallet_symmetry_dht_local_v2', OLD,
             REPO/'data/pallet/results/pallet_symmetry_dht_local_v1']
    return {str(p.relative_to(REPO)): sha256(p) for folder in paths for p in sorted(folder.rglob('*')) if p.is_file()}


def controlled_oracle(output):
    """Same eligible edge intersection for all three branches; same utility rule.

    The rule is fixed gain >0.25px for every branch; selected edges can differ
    with line quality. These GT-assisted comparisons are not causal fractions.
    """
    dataset = ObservationDataset(EXPORT/'synth_val.json', targets=True)
    lattice = Lattice()
    pred_path = REPO/'data/pallet/results/pallet_symmetry_dht_local_v1/predictions/hough_seed1_synth_val.json'
    predicted = {r['frame_id']: r for r in read_json(pred_path)['records']}
    rows = []; all_values = {k: [] for k in ('point', 'exact_GT', 'soft_GT', 'v1_hough_seed1')}
    for item in dataset:
        obs = observation_batch(collate([item]), torch.device('cpu'))
        diagonal = float(item['image_hw'].norm())
        _, gt, valid, choice = diagnostics._choose_gt(item['base_points'], item['point_valid'], item['target_points'], item['target_valid'], item['symmetry_permutations'], diagonal)
        exact = diagnostics._raw_gt_lines(gt)
        soft, _, support, _ = geometry.continuous_line_targets(gt[None], valid[None], obs['box'], lattice)
        decoded = geometry.decode_single_mode(soft.clamp_min(1e-30).log(), torch.ones_like(soft, dtype=torch.bool), lattice, obs['box'])
        old = predicted[item['frame_id']]
        old_lines = torch.tensor(old['raw_lines'], dtype=torch.float32)
        old_weights = torch.tensor(old['absolute_mode_weight'], dtype=torch.float32)
        selected = old_lines[torch.arange(12), old_weights.argmax(-1)][None]
        common = support & decoded['available'] & old_weights.max(-1).values.gt(0)[None]
        branches = {'exact_GT': exact, 'soft_GT': decoded['raw_line'], 'v1_hough_seed1': selected}
        for lines in branches.values():
            common &= torch.isfinite(lines).all(-1) & (lines[..., :2].norm(dim=-1) > 1e-8)
        base_error = diagnostics._point_errors(item['base_points'], item['point_valid'], gt, valid, diagonal)
        row = {'frame_id': item['frame_id'], 'baseline_symmetry_choice': choice, 'common_edge_mask': common[0].tolist(), 'point': float(base_error.mean())}
        for name, lines in branches.items():
            error, _, utility, _ = diagnostics._oracle_correct(obs, gt[None], valid[None], lines, common)
            row[name] = error; row[name+'_oracle_use_fraction'] = utility
        rows.append(row)
        for name in all_values: all_values[name].append(row[name]/diagonal)
    value = {'scope': 'GT-assisted explanatory oracle; no neural forward or optimizer update',
             'common_eligibility_mask': 'intersection of structural support, valid soft decode and existing v1 availability, finite nonzero normals',
             'selection_policy': 'same detached unit-edge gain >0.25px rule for all branches; realized selected subsets can differ',
             'not_causal_contribution_fractions': True, 'predicted_source': str(pred_path.relative_to(REPO)),
             'predicted_source_sha256': sha256(pred_path), 'manifest_sha256': sha256(EXPORT/'synth_val.json'),
             'summary': {k: float(np.mean(v)) for k, v in all_values.items()}, 'records': rows}
    immutable_json(output, value)
    return value


def prepare():
    DOC.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'main'
    immutable_json(DOC/'PRESERVED_ARTIFACT_SHA.json', preserved_hashes())
    protocol = read_json(PACKAGE/'PROTOCOL_LOCK.json')
    immutable_json(DOC/'PROTOCOL_LOCK.json', protocol)
    immutable_json(DOC/'CORRECTION_STATUS.json', {
        'original_commit': '187589af31ece5618cb5f9ee82e9a1bb0ab13f53',
        'original_method_conclusion': 'WITHHELD_DUE_TO_TRAINING_AND_INFERENCE_WLS_BUG',
        'original_oracle_status': 'INVALID_FOR_CAUSAL_DECOMPOSITION_DUE_TO_WLS_BUG',
        'original_result_and_verdict_preserved': True,
        'scope': 'endpoint-specific residual only; no loss/lattice/utility threshold/seed/budget change',
        'authorized_reproduction_updates': 12000})
    par = parity.run(EXPORT/'train.json', DOC/'INIT_PARITY.json')
    train_manifest = read_json(EXPORT/'train.json'); train_sha = sha256(EXPORT/'train.json')
    val_ids = {r['frame_id'] for r in read_json(EXPORT/'synth_val.json')['records']}
    train_ids = {r['frame_id'] for r in train_manifest['records']}
    assert len(train_ids) == 1792 and not train_ids & val_ids and par['PASS']
    starts = []
    for arm in ('direct', 'hough'):
        for seed in (1, 2, 3):
            path = OLD/f'heads/{arm}_seed{seed}/START.json'; start = read_json(path)
            assert Path(start['manifest']).resolve() == (EXPORT/'train.json').resolve()
            assert start['manifest_sha256'] == train_sha
            assert start['minibatch_order_sha256'] == canonical_sha(runner.sequence(1792, 16000, seed))
            expected = par['rows'][seed-1]
            assert start['common_parameter_sha256'] == expected['direct_common_parameter_sha256']
            assert start['initial_utility_head_sha256'] == expected['direct_initial_utility_head_sha256']
            starts.append({'path': str(path.relative_to(REPO)), 'sha256': sha256(path), 'START': start})
    immutable_json(DOC/'TRAIN_SPLIT_CORRECTION.json', {'train_count': 1792, 'train_manifest_sha256': train_sha,
        'train_manifest': str((EXPORT/'train.json').relative_to(REPO)), 'validation_disjoint': True,
        'prior_INIT_PARITY_called_with_validation': True, 'interpretation': 'preflight record/call error; all six saved STARTs bind approved train manifest and 1792 sequence', 'original_starts': starts})
    post = RAW/'posthoc'; evaluations = []
    dataset = ObservationDataset(EXPORT/'synth_val.json', targets=False)
    for name in NAMES:
        source = OLD/f'predictions/{name}_synth_val.json'
        prediction = read_json(source); value = copy.deepcopy(prediction)
        saved = {r['frame_id']: r for r in value['records']}
        if name != 'point':
            for item in dataset:
                row = saved[item['frame_id']]
                corrected, delta = geometry.single_mode_wls(item['base_points'][None], item['point_valid'][None], item['point_sigma'][None],
                    torch.tensor(row['raw_line'], dtype=torch.float32)[None], torch.tensor(row['utility'], dtype=torch.float32)[None],
                    .005*item['image_hw'].norm()[None], item['image_hw'][None])
                row['points'] = corrected[0].tolist(); row['correction'] = delta[0].tolist()
        value.update(scope='fixed WLS applied to original saved raw_line and utility; zero neural forwards, zero optimizer updates', source_prediction_sha256=sha256(source), GT_opened=False)
        pred_path = post/f'predictions/{name}_synth_val.json'; immutable_json(pred_path, value)
        ev = assessment.evaluate(EXPORT/'synth_val.json', pred_path, post/f'evaluations/{name}_synth_val.json')
        old_ev = read_json(OLD/f'evaluations/{name}_synth_val.json')
        keys = ('primary_frame_mean_over_raw_diagonal','pooled_symmetric_median_px','pooled_symmetric_p90_px','good_point_damage_rate','coverage_count','new_catastrophic_frame_count')
        evaluations.append({'name': name, 'buggy': {k: old_ev[k] for k in keys}, 'fixed_posthoc': {k: ev[k] for k in keys}, 'records': ev['records']})
    immutable_json(DOC/'OFFLINE_WLS_REPLAY.json', {'scope':'same saved line/utility, no GT selection; not corrected training', 'records': evaluations})
    oracle = controlled_oracle(DOC/'ORACLE_CONTROLLED.json')
    immutable_json(DOC/'PRETRAINING_COMPLETE.json', {'PASS': True, 'oracle_summary': oracle['summary'],
        'training_manifest_sha256': train_sha, 'core_source_sha256': {f:sha256(PACKAGE/'symdht_local_v2'/f) for f in ('geometry.py','model.py','objective.py','runner.py')},
        'protocol_sha256': sha256(PACKAGE/'PROTOCOL_LOCK.json')})
    print(json.dumps({'oracle': oracle['summary'], 'posthoc': [{k:v for k,v in r.items() if k!='records'} for r in evaluations]}, indent=2), flush=True)


def train_all():
    gate = read_json(DOC/'PRETRAINING_COMPLETE.json'); assert gate['PASS']
    tests = subprocess.run([sys.executable, '-m', 'pytest', str(PACKAGE/'tests'), '-q'], cwd=REPO, capture_output=True, text=True)
    immutable_json(DOC/'REGRESSION_TESTS.json', {'exit_code': tests.returncode, 'stdout':tests.stdout, 'stderr':tests.stderr,
        'test_source_sha256': {str(p.relative_to(REPO)):sha256(p) for p in (PACKAGE/'tests').glob('test_*.py')}})
    assert tests.returncode == 0, tests.stdout + tests.stderr
    for filename, expected in gate['core_source_sha256'].items(): assert sha256(PACKAGE/'symdht_local_v2'/filename) == expected
    assert sha256(PACKAGE/'PROTOCOL_LOCK.json') == gate['protocol_sha256']
    for arm in ('direct','hough'):
        for seed in (1,2,3):
            print(f'START corrected {arm} seed {seed}', flush=True)
            runner.train(argparse.Namespace(output=str(RAW/f'heads/{arm}_seed{seed}'), device='cuda:0', seed=seed,
                config=str(PACKAGE/f'configs/{arm}.json'), manifest=str(EXPORT/'train.json'), steps=2000,batch=8,lr=.001,weight_decay=.0001))
            print(f'DONE corrected {arm} seed {seed}', flush=True)


def finish():
    gate = read_json(DOC/'PRETRAINING_COMPLETE.json')
    for filename, expected in gate['core_source_sha256'].items(): assert sha256(PACKAGE/'symdht_local_v2'/filename) == expected
    checks = []
    par = read_json(DOC/'INIT_PARITY.json')
    for arm in ('direct','hough'):
        for seed in (1,2,3):
            start = read_json(RAW/f'heads/{arm}_seed{seed}/START.json'); end = read_json(RAW/f'heads/{arm}_seed{seed}/COMPLETION.json')
            assert start['manifest_sha256'] == gate['training_manifest_sha256'] and start['steps'] == end['steps'] == 2000
            assert start['minibatch_order_sha256'] == par['rows'][seed-1]['minibatch_order_sha256']
            assert start['common_parameter_sha256'] == par['rows'][seed-1]['direct_common_parameter_sha256']
            assert sha256(end['checkpoint']) == end['checkpoint_sha256']
            checks.append({'START': start, 'COMPLETION': end})
    immutable_json(DOC/'TRAINING_AUDIT.json', {'PASS':True, 'runs':checks})
    runner.score_point(argparse.Namespace(manifest=str(EXPORT/'synth_val.json'),output=str(RAW/'predictions/point_synth_val.json')))
    for name in NAMES:
        pred = RAW/f'predictions/{name}_synth_val.json'
        if name!='point': runner.score(argparse.Namespace(device='cuda:0',checkpoint=str(RAW/f'heads/{name}/checkpoint_final.pt'),manifest=str(EXPORT/'synth_val.json'),output=str(pred),batch=32))
        assessment.evaluate(EXPORT/'synth_val.json',pred,RAW/f'evaluations/{name}_synth_val.json')
    result = assessment.publish(RAW/'evaluations/point_synth_val.json', [RAW/f'evaluations/direct_seed{s}_synth_val.json' for s in (1,2,3)], [RAW/f'evaluations/hough_seed{s}_synth_val.json' for s in (1,2,3)],DOC)
    assert preserved_hashes() == read_json(DOC/'PRESERVED_ARTIFACT_SHA.json')
    immutable_json(DOC/'PRESERVATION_VERIFIED.json', {'PASS':True, 'original_artifacts_count':len(read_json(DOC/'PRESERVED_ARTIFACT_SHA.json'))})
    print(json.dumps(result, indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase',choices=['prepare','train','finish']); args=parser.parse_args()
    {'prepare':prepare,'train':train_all,'finish':finish}[args.phase]()
