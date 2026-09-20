import json
from pathlib import Path
import pytest
from scripts.evaluation import final_dimension_release as R


def test_freeze_requires_explicit_completion():
    with pytest.raises(ValueError, match='confirm-annotation-complete'):
        R.freeze_eval(False)


def test_artifact_immutable(tmp_path):
    path=tmp_path/'lock.json'
    R.freeze(path, {'a':1})
    R.freeze(path, {'a':1})
    with pytest.raises(ValueError, match='Immutable'):
        R.freeze(path, {'a':2})
    assert json.loads(path.read_text()) == {'a':1}


def test_binding_detects_change(tmp_path, monkeypatch):
    monkeypatch.setattr(R,'ROOT',tmp_path)
    path=tmp_path/'model.bin'; path.write_bytes(b'first')
    b=R.binding(path); R.verify(b)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen input changed'):
        R.verify(b)


def test_review_status_is_readonly_and_filters_train(tmp_path, monkeypatch):
    monkeypatch.setattr(R,'ROOT',tmp_path)
    monkeypatch.setattr(R,'REVIEW',tmp_path/'review')
    labels=R.REVIEW/'full_session_annotations/s1_manual_gt'; labels.mkdir(parents=True)
    rgb=tmp_path/'raw/s1/rgb'; rgb.mkdir(parents=True)
    (rgb.parent/'cam_K.txt').write_text('600 0 320\n0 600 240\n0 0 1\n')
    (R.REVIEW/'full_sessions_manifest.json').write_text(json.dumps(dict(records=[dict(
        session='s1',source='raw/s1',annotation_count=1)])))
    for split in ('train','eval'):
        (rgb/(split+'.png')).write_bytes(split.encode())
        ann=dict(object_type='plastic_standard_110x110x15',camera_data=dict(intrinsics=dict(fx=600,fy=600,cx=320,cy=240)),
                 objects=[dict(split=split,keypoint_annotations=[dict(xy=[10,20],visibility=2,source='manual_click')]*9)])
        (labels/(split+'.json')).write_text(json.dumps(ann))
    original={p:R.sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    result=R.review_rows()
    assert result['count']==1 and result['sessions']==1 and not result['issues']
    assert result['records'][0]['id']=='s1__eval'
    assert {p:R.sha(p) for p in tmp_path.rglob('*') if p.is_file()} == original


def test_duplicate_eval_images_rejected(tmp_path, monkeypatch):
    test_review_status_is_readonly_and_filters_train(tmp_path,monkeypatch)
    labels=R.REVIEW/'full_session_annotations/s1_manual_gt'
    (labels/'eval2.json').write_bytes((labels/'eval.json').read_bytes())
    rgb=tmp_path/'raw/s1/rgb'; (rgb/'eval2.png').write_bytes((rgb/'eval.png').read_bytes())
    assert any('Duplicate' in r['issue'] for r in R.review_rows()['issues'])
