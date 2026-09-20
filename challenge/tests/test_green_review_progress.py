import json
from pathlib import Path

from scripts.evaluation import eval_workspace as W
from scripts.evaluation.green_review_progress import BEGIN, collect, update


def fixture(tmp_path):
    review = tmp_path / 'review'
    labels = review / 'full_session_annotations/s1_manual_gt'
    labels.mkdir(parents=True)
    rgb = tmp_path / 'raw/s1/rgb'
    rgb.mkdir(parents=True)
    for name in ('old', 'new', 'unlabelled'):
        (rgb / (name + '.png')).write_bytes(b'fake image for path check')
    (review / 'full_sessions_manifest.json').write_text(json.dumps(dict(records=[dict(
        session='s1', source='raw/s1', frame_count=3,
        source_annotations=[dict(name='old.json', sha256='unused')])])) )
    write_label(labels / 'old.json', 'train')
    workspace = tmp_path / 'workspace'
    (workspace / 'reports').mkdir(parents=True)
    (workspace / 'reports/ANNOTATION_PROGRESS.md').write_text('# Original\nPositive total 319\n')
    return review, labels, workspace


def write_label(path: Path, split: str):
    path.write_text(json.dumps(dict(objects=[dict(
        object_type='plastic_standard_110x110x15', split=split,
        keypoint_annotations=[{} for _ in range(9)])])))


def test_copies_not_counted_as_reviewed_eval(tmp_path):
    review, _, _ = fixture(tmp_path)
    counts = collect(review, tmp_path)['totals']
    assert counts['original'] == counts['annotated'] == counts['train'] == 1
    assert counts['eval'] == counts['new'] == 0


def test_save_toggle_delete_counts(tmp_path):
    review, labels, _ = fixture(tmp_path)
    write_label(labels / 'new.json', 'eval')
    assert collect(review, tmp_path)['totals']['eval'] == 1
    write_label(labels / 'old.json', 'eval')
    assert collect(review, tmp_path)['totals']['eval'] == 2
    write_label(labels / 'new.json', 'train')
    assert collect(review, tmp_path)['totals']['eval'] == 1
    (labels / 'old.json').unlink()
    counts = collect(review, tmp_path)['totals']
    assert counts['eval'] == 0 and counts['new'] == 1


def test_invalid_or_missing_source_excluded(tmp_path):
    review, labels, _ = fixture(tmp_path)
    (labels / 'new.json').write_text('{')
    write_label(labels / 'missing.json', 'eval')
    data = collect(review, tmp_path)
    assert len(data['errors']) == 2 and data['totals']['eval'] == 0


def test_update_idempotent_preserves_main_count(tmp_path):
    review, labels, workspace = fixture(tmp_path)
    write_label(labels / 'new.json', 'eval')
    update(review, workspace, tmp_path)
    main = workspace / 'reports/ANNOTATION_PROGRESS.md'
    first = main.read_text()
    stamp = main.stat().st_mtime_ns
    update(review, workspace, tmp_path)
    assert main.read_text() == first and main.stat().st_mtime_ns == stamp
    assert first.count(BEGIN) == 1 and 'Positive total 319' in first


def test_normal_report_refresh_preserves_review_section(tmp_path, monkeypatch):
    review, _, workspace = fixture(tmp_path)
    update(review, workspace, tmp_path)
    monkeypatch.setattr(W, 'load_targets', lambda root: {})
    for name in ('render_progress_report', 'render_composition_report', 'render_priority_report',
                 'render_overlay_audit', 'render_domain_coverage', 'render_paper_domain_coverage'):
        monkeypatch.setattr(W, name, lambda *args: '# refreshed main\n')
    monkeypatch.setattr(W, 'write_duplicate_audit', lambda *args: None)
    W.write_reports(workspace, [])
    text = (workspace / 'reports/ANNOTATION_PROGRESS.md').read_text()
    assert text.startswith('# refreshed main') and text.count(BEGIN) == 1
