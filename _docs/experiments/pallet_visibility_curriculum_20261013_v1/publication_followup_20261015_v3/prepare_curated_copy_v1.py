"""Copy closed evidence only into the isolated publication worktree; no git writes."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ORIGINAL = Path('/home/minjae/Documents/github/pallet-pose')
TARGET = Path('/tmp/pallet-visibility-main-publication-20261015-v1')
D = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B = D / 'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
CAMPAIGN = D / 'nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1'
S = Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
PUB = D / 'publication_followup_20261015_v3'
PROTECTED = ['.gitignore', 'readme.md',
 '_docs/experiments/pallet_resnet18_refiner_20261001_v1/README_KO.md',
 '_docs/experiments/pallet_three_estimator_20261001_v1/README_KO.md',
 '_docs/experiments/pallet_three_estimator_20261001_v1/STATUS_KO.md',
 'data/evaluation/pallet_eval_v1/reports/ANNOTATION_PROGRESS.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_state(root):
    return {p: {'exists': (root / p).is_file(),
                'sha256': sha(root / p) if (root / p).is_file() else None}
            for p in PROTECTED}


def main():
    out = TARGET / PUB
    if (out / 'COPIED_MANIFEST.json').exists():
        raise FileExistsError('Preserve each publication cutoff')
    before = {str(root): protected_state(root) for root in [ORIGINAL, TARGET]}
    candidates = set()

    def direct(folder, names=None):
        path = ORIGINAL / folder
        for file in sorted(path.iterdir()):
            if file.is_file() and (names is None or file.name in names): candidates.add(file.relative_to(ORIGINAL))

    direct(B / 'geometry_guided_edge_eval_v4_29', {
        'REPORT_KO.md', 'CAL_SCORE.json', 'TEST_SCORE.json', 'REMAINDER_SCORE.json',
        'INPUT_SEAL.json', 'PREDICTION_SEAL.json', 'SCORE_SEAL.json', 'PROTOCOL.json',
        'OPERATIONAL_SOURCE_CLOSURE_RECEIPT_V1.json', 'OWNER_PREDICTION_COMPLETION_RECEIPT_V1.json'})
    direct(B / 'geometry_guided_edge_eval_v4_29/independent_root_audit_v1')
    for folder in ['geometry_guided_process_v4_29_v1', 'geometry_guided_process_plan_v4_29',
        'independent_geometry_guided_process_review_v4_29', 'independent_campaign_accounting_review_v4_29',
        'independent_campaign_accounting_review_v4_29/v2', 'geometry_guided_edge_root_checks_v4_29',
        'CAL_geometry_guided_edge_noise_v4_29_checks', 'CAL_geometry_guided_edge_noise_v4_30_checks',
        'CAL_geometry_guided_edge_noise_plan_v4_30', 'geometry_guided_edge_noise_v4_30_checks',
        'geometry_guided_edge_noise_peer_review_v4_30', 'geometry_guided_edges_peer_review_v4_30',
        'geometry_guided_actual_runtime_smoke_v4_30', 'geometry_guided_fitted_native_independent_smoke_v4_30',
        'observed_bi_associated_edges_plan_v4_30']:
        direct(B / folder)
    for generation in ['v4_29', 'v4_30']:
        folder = B / ('CAL_geometry_guided_edge_noise_' + generation)
        direct(folder, {'MODEL_SEAL.json', 'CAL_LOGO_REPORT.json', 'FITTED_RUNTIME_EXPORTS_V1.json',
            'OWNER_CAL_FIT_COMPLETION_V1.json', 'PREPARATION_SEAL.json', 'PROTOCOL.json'})
        for policy in ['ALL_BI_REPLACE', 'RGB_NONFALLBACK_PRESERVE']:
            direct(folder / policy, {'MODEL.json', 'TANGENT_MODEL.json', 'CAL_ERRORS.jsonl.gz'})
    direct(CAMPAIGN, {'PROGRESS_V10_V29.json', 'RESULTS_V10_V29.html', 'PROGRESS_V10_V29_SEAL.json'})
    # Copy latest direct sources/tests only when absent or different. Root explicitly
    # approved updating the two already-published V30 controller source files.
    source_updates = {}
    for source in sorted((ORIGINAL / S).glob('*.py')):
        rel = source.relative_to(ORIGINAL); target = TARGET / rel
        if not target.exists() or sha(source) != sha(target):
            candidates.add(rel)
            source_updates[str(rel)] = {'previous_f403a66_sha256': sha(target) if target.exists() else None,
                'current_workspace_sha256': sha(source), 'previously_published': target.exists()}
    entries = {}
    for rel in sorted(candidates):
        source = ORIGINAL / rel; target = TARGET / rel
        if 'code_snapshot' in rel.parts or 'code_snapshots' in rel.parts or '__pycache__' in rel.parts:
            raise ValueError('Code snapshot excluded')
        if source.stat().st_size > 5 * 1024 * 1024:
            raise ValueError('Unexpected large curated artifact: ' + str(rel))
        if source.name.endswith('.gz') and source.name not in ['CAL_ERRORS.jsonl.gz', 'SELECTED_POSE_COMPONENTS.jsonl.gz']:
            raise ValueError('Raw journals excluded')
        if target.exists() and sha(target) != sha(source) and str(rel) not in source_updates:
            raise ValueError('Preserve already-published evidence: ' + str(rel))
        if rel.suffix == '.py': ast.parse(source.read_text(), filename=str(rel))
        target.parent.mkdir(parents=True, exist_ok=True)
        source_digest = sha(source); shutil.copyfile(source, target)
        if sha(source) != source_digest or sha(target) != source_digest:
            raise ValueError('Copy was not byte exact')
        entries[str(rel)] = {'sha256': source_digest, 'bytes': source.stat().st_size,
            'kind': 'latest_direct_python' if rel.parts[:len(S.parts)] == S.parts else 'closed_evidence'}
    after = {str(root): protected_state(root) for root in [ORIGINAL, TARGET]}
    if before != after: raise ValueError('Protected file changed')
    cutoff = dict(schema='CURATED_PUBLICATION_CUTOFF_V3', generated_UTC=datetime.now(timezone.utc).isoformat(),
        base_publication_commit='f403a66d6283cd6bbca015a48e7a96b6f2bbfe07',
        source_workspace=str(ORIGINAL), isolated_worktree=str(TARGET),
        V29=dict(predictions=238, arms=5, independent_audit_checks=77571, independent_audit_failures=0,
            fixed_scores_complete=True, default='SAME_FROZEN_E', adopted=False),
        V30=dict(CAL_inventory=46, compatible_CAL=44, policy_specific_line_and_tangent_models=20,
            saved_optimizer_starts=60, all_saved_starts_success=True, saved_error_replay_checks=9342,
            runtime_CAL_smoke_checks=53, independent_native_CAL_smoke_checks=293,
            full_primary_evaluation_complete_at_this_cutoff=False, primary_accuracy_claim=False),
        scope='Closed V29 evidence and V30 CAL/runtime validation only; state describes this timestamp, not future completion.',
        excluded=['Large INPUTS/PREDICTIONS/OBSERVATIONS gzip journals', 'per-frame raw graph/proposal banks',
            'code_snapshot/code_snapshots', 'live V30 OBS/REST folders, spools, LIVE files and incomplete outputs',
            'CAL authenticated raw-input journals', 'source RGB/mask/annotation datasets'],
        included_small_gzip_scope='Saved CAL_ERRORS for independent likelihood replay and selected-pose audit scalar witnesses only.',
        initial_release_unchanged=True, initial_release_cutoff='CAL32/46; later raw238 evidence is not claimed to be in that release.',
        commit_or_push_performed=False, protected_before_after=before)
    (out / 'CUTOFF.json').write_text(json.dumps(cutoff, ensure_ascii=False, indent=2) + '\n')
    for rel in [PUB / 'REPORT_KO.md', PUB / 'CUTOFF.json', PUB / 'prepare_curated_copy_v1.py']:
        file = TARGET / rel
        entries[str(rel)] = {'sha256': sha(file), 'bytes': file.stat().st_size, 'kind': 'publication_documentation'}
    manifest = dict(schema='CURATED_PUBLICATION_COPY_MANIFEST_V3', base_commit=cutoff['base_publication_commit'],
        cutoff_UTC=cutoff['generated_UTC'], file_count=len(entries), total_bytes=sum(e['bytes'] for e in entries.values()),
        entries=entries, direct_source_updates=source_updates, original_workspace_files_modified=0,
        existing_published_evidence_overwritten=0, commit_or_push_performed=False)
    (out / 'COPIED_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(files=len(entries), bytes=manifest['total_bytes'], direct_source_updates=source_updates,
        manifest_sha256=sha(out / 'COPIED_MANIFEST.json')), ensure_ascii=False))


if __name__ == '__main__': main()
