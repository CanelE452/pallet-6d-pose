"""Metadata-only final join of the fixed 49 nonseed public source pairs.

Preserve all primary observations and the already sealed secondary5 proposals.
This does not turn AI observations into human/native8 ground truth or admit fit.
"""
import copy
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('immutable_annotation_validator', ROOT / 'validate_and_join_primary_annotations.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
read, bind = validator.read, validator.binding


def write(path, obj):
    with path.open('x') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, rows):
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def main():
    source = read(ROOT / 'ALL50_PAIR_COHORT_LOCK_BEFORE_NEW_RGB.json')
    all_sources = {r['pair_id']: r for r in source['rows']}
    expected = {r['pair_id'] for r in source['rows'] if not r['seed_V84_reuse']}
    assert len(expected) == 49
    primary = []; bindings = [bind(ROOT / 'ALL50_PAIR_COHORT_LOCK_BEFORE_NEW_RGB.json'), bind(ROOT / 'validate_and_join_primary_annotations.py')]
    units = []
    for worker in ('a', 'b'):
        ordered = sorted([r for r in source['rows'] if r['primary_annotator'] == worker and not r['seed_V84_reuse']], key=lambda r: (r['rank_within_split_SHA256'], r['split']))
        actual = []
        for number in range(1, 6):
            suffix = '' if number == 1 else f'/batch{number:02d}'
            manifest = ROOT / f'annotator_{worker}{suffix}/BATCH_MANIFEST.json'
            batch = read(manifest); assignment = ROOT / f'ANNOTATOR_{worker.upper()}_BATCH{number:02d}_BEFORE_NEW_RGB.json'
            plan = read(assignment)
            assert batch['completed'] and batch['annotator'] == worker
            assert [r['pair_id'] for r in batch['pairs']] == [r['pair_id'] for r in plan['rows']]
            bindings += [bind(manifest), bind(assignment)]
            for unit in batch['pairs']:
                pair = unit['pair_id']; actual.append(pair)
                before = validator.bound_read(unit['before_annotation']); after = validator.bound_read(unit['after_annotation'])
                validator.bound_read(unit['before_seal']); validator.bound_read(unit['after_seal'])
                assert after['before_annotation']['sha256'] == unit['before_annotation']['sha256']
                assert after['before_seal']['sha256'] == unit['before_seal']['sha256']
                primary += validator.validate_pair(before, after, all_sources[pair])
                bindings += [unit[k] for k in ('before_annotation', 'after_annotation', 'before_seal', 'after_seal')]
                units.append({'pair_id': pair, 'source_split': all_sources[pair]['split'], 'primary_annotator': worker, 'batch': number, 'before_annotation': unit['before_annotation'], 'after_annotation': unit['after_annotation']})
        assert actual == [r['pair_id'] for r in ordered]
    assert len(units) == 49 and {u['pair_id'] for u in units} == expected and len(primary) == 294
    assert len({(r['pair_id'], r['part_slot']) for r in primary}) == 294
    out = ROOT / 'all49_completed_corpus'
    out.mkdir(exist_ok=False)
    write(out / 'ALL_PRIMARY_COMPLETE_BEFORE_SECONDARY_RECONCILIATION.json', {'completed': True, 'all49_primary_annotations_sealed': True, 'pairs': 49, 'slots': 294, 'primary_bindings': bindings, 'release': 'Primary-after quarantine lifted only now for metadata reconciliation with already sealed secondary5. No primary annotation changed.'})
    # Retain the exact previously sealed two U downgrades; no new consensus or
    # favorable selection is created from the remaining unreviewed primary rows.
    prior_file = ROOT / 'cumulative_20pair_review/RECONCILED_120_CUE_PROPOSALS_GT_ONLY.jsonl'
    prior = {(r['pair_id'], r['part_slot']): r for r in map(json.loads, prior_file.read_text().splitlines())}
    proposed = []; changed = []
    for r in primary:
        key = (r['pair_id'], r['part_slot'])
        if key in prior:
            assert validator.canonical(prior[key]['primary_record_unchanged']) == validator.canonical(r)
            proposal = copy.deepcopy(prior[key])
        else:
            proposal = {'pair_id': r['pair_id'], 'part_slot': r['part_slot'], 'source_split': r['source_split'], 'source_family': r['source_family'], 'reference_xy': r['reference_xy'], 'uncertainty_bbox_xyxy_half_open': r['uncertainty_bbox_xyxy_half_open'], 'proposed_aux_after_state': r['after_state'], 'proposal_rule_reason': 'PRIMARY_ONLY_AI_NOT_IN_SECONDARY5', 'secondary_audited': False, 'primary_record_unchanged': r, 'secondary_record_unchanged': None, 'native8_semantic_id': None, 'human_gold': False, 'training_or_forward_admission': False, 'source_tier': 'PRIMARY_AI_WEAK_CUE_OBSERVATION_ONLY'}
        if proposal['proposed_aux_after_state'] != r['after_state']:
            changed.append({'pair_id': r['pair_id'], 'part_slot': r['part_slot'], 'primary': r['after_state'], 'proposed_aux': proposal['proposed_aux_after_state']})
        proposed.append(proposal)
    assert len(changed) == 2 and all(c['pair_id'] == '1.23m_hall:0024_0025' for c in changed)
    jsonl(out / 'PRIMARY_294_CUE_OBSERVATIONS_GT_ONLY.jsonl', primary)
    jsonl(out / 'PROPOSED_294_CUE_AUXILIARY_GT_ONLY.jsonl', proposed)
    jsonl(out / 'PAIR_SOURCE_ANNOTATION_LINEAGE_GT_ONLY.jsonl', units)
    counts = defaultdict(lambda: {'primary': Counter(), 'proposed_aux': Counter(), 'before': Counter()})
    for r, proposal in zip(primary, proposed):
        group = counts[r['source_split']]
        group['primary'][r['after_state']] += 1
        group['proposed_aux'][proposal['proposed_aux_after_state']] += 1
        group['before'][r['before_state']] += 1
    write(out / 'CORPUS_SUMMARY_KO.json', {'PASS_metadata_identity_and_completeness': True, 'nonseed_pairs': 49, 'part_slots': 294, 'source_split_counts': {s: dict(d) for s, d in counts.items()}, 'primary_after_counts': dict(Counter(r['after_state'] for r in primary)), 'proposed_aux_after_counts': dict(Counter(r['proposed_aux_after_state'] for r in proposed)), 'secondary_reviewed_pairs': 5, 'secondary_slots': 30, 'unchanged_two_previous_U_downgrades': changed, 'seed_V84': 'Separate one pair with8 original proposals/6 linked distinct regions; not folded into canonical6 retrospective rows.', 'full_cohort_including_separate_seed_pairs': 50, 'physical_bin_instances': 1, 'environment_holdout_only': True, 'native8_human_gold_added': 0, 'training_admission': False, 'model_fit_current_PnP': 0, 'definition_KO': 'view-relative physical wall/runner cues; native virtual8과는 별개. before 관찰로 reference/unc를 고정한 뒤 after의 동일 영역을 관찰. 파일 짝홀이나 같은 pose만으로 visible/E로 정하지 않음.', 'bindings': bindings + [bind(prior_file), bind(ROOT / 'secondary_audit/SECONDARY5_FINAL_REVIEW_SEAL.json')], 'join_code': bind(__file__)})
    files = sorted(p for p in out.iterdir() if p.is_file())
    write(out / 'SEAL.json', {'completed': True, 'files': [bind(f) for f in files], 'no_primary_annotation_changed': True, 'current_data_reads': 0, 'model_updates': 0, 'PnP': 0})
    print(json.dumps({'completed': True, 'summary': bind(out / 'CORPUS_SUMMARY_KO.json'), 'seal': bind(out / 'SEAL.json')}))


if __name__ == '__main__':
    main()
