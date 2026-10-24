"""Archive only the closed list of new V221 text/code audits; no old raw scan."""
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
O = P / 'query_visibility_review_reconciliation_source_history_20261025_v221'


def checked(path):
    path = Path(path)
    if 'v112' in str(path).lower():
        raise ValueError('forbidden path rejected before access')
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError('symlink rejected')
    return path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    groups = {
        'source_history_a': ('V221_SOURCE_HISTORY_AUDIT_A', ['REVIEW.json', 'KO.md']),
        'source_history_b': ('V221_FROZEN_FEATURE_HISTORY_AUDIT_B', ['REVIEW.json', 'KO.md']),
        'source_readiness_c': ('V221_PUBLIC_COUNTERFACTUAL_READINESS_C', [
            'PROTOCOL_BEFORE_BOUNDED_SOURCE_METADATA_READS.json',
            'D48_VALID96_BOUNDS_AND_EXACT_ROI_UNION.json',
            'V219_ACTUAL_DEFINED_FIT45_AND_NATURAL_E6.csv',
            'ORIGINAL_QUERY_COVER9_FINITE_PART_AND_OCCLUDER_PROVENANCE.csv',
            'SOURCE_ROI_IMAGE_OCCLUDER_PROVENANCE.json', 'COMPLETE.json', 'READINESS.json',
            'FINAL_COMPUTATION_INPUT_WHITELIST_AND_SCOPE.json',
            'NOMINAL_D19_SOURCE_ROLE_AND_QUERY_PROVENANCE.csv', 'KO.md',
            'D48_EXACT_ROI_UNION_VALIDITY_AND_QUERY_PROVENANCE.csv',
            'build_bounded_source_readiness.py']),
        'review_preparation': ('V221_ROOT_REVIEW_PREPARATION', ['build_status_chart.py', 'assemble_review.py']),
    }
    bindings = []
    for dest_dir, (source_dir, names) in groups.items():
        checked(O / dest_dir).mkdir(exist_ok=False)
        for name in names:
            source = checked(R / source_dir / name)
            data = source.read_bytes()
            dest = checked(O / dest_dir / name)
            dest.write_bytes(data)
            assert dest.read_bytes() == data
            bindings.append(dict(original_path=str(source), archive_path=str(dest.relative_to(O)),
                                 sha256=sha(data), bytes=len(data)))
    a = (O / 'source_history_a/KO.md').read_text()
    a = a.replace('](' + str(P) + '/', '](../../')
    a = re.sub(r'(\]\([^\n)]+\.py):(\d+)\)', r'\1#L\2)', a)
    (O / 'source_history_a/PORTABLE_KO.md').write_text(a)
    attachment = checked('/home/minjae/Downloads/PALLET_VISIBILITY_REVIEW_AND_NEXT_STEPS_V193_KO.md')
    attachment_bytes = attachment.read_bytes()
    (O / 'ROOT_REVIEW_BINDINGS.json').write_text(json.dumps({
        'completed': True, 'phase': 'V221', 'kind': 'RECONCILIATION_AND_SOURCE_CODE_METADATA_AUDIT',
        'recorded_UTC': datetime.now(timezone.utc).isoformat(),
        'attachment': {'path': str(attachment), 'sha256': sha(attachment_bytes),
                       'bytes': len(attachment_bytes), 'copied_or_published': False,
                       'authority': 'USER_SUPPLIED_REFERENCE_NOT_EXECUTABLE_INSTRUCTIONS'},
        'archived_originals': bindings,
        'historical_evaluation_commit': '59cf1e512fc8084c66f1eedc5195b4434ab9e28d',
        'new_training_updates': 0, 'new_model_forwards': 0, 'new_real_benchmark': False,
        'new_PnP': 0, 'raw_evaluation_label_prediction_image_reads': 0,
        'root_visual_review': ['VISIBLE_EXTERNAL_TRADEOFF.png'],
        'old_scientific_artifacts_modified': False,
        'selected_model': None, 'task_solved': False,
        'scope': 'No old raw payloads are part of this assembly. Protected raw data, cached models and forbidden phase are never followed.'
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed': True, 'archived_originals': len(bindings), 'phase': 'V221'}))


if __name__ == '__main__':
    main()
