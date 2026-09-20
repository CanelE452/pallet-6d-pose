import json
from pathlib import Path

from scripts.evaluation.convert_green_review_to_eval import convert


def test_bulk_conversion_changes_only_split_and_preserves_backup(tmp_path):
    review=tmp_path/'review'
    labels=review/'full_session_annotations/session_manual_gt'
    labels.mkdir(parents=True)
    (review/'full_sessions_manifest.json').write_text(json.dumps(dict(records=[dict(session='session')])))
    original={}
    for split in ('train','eval'):
        doc=dict(note='keep',objects=[dict(object_type='plastic_standard_110x110x15',
            split=split,manual_kps=[[1,2]],pose_transform=[[3,4]],custom={'keep':True})])
        path=labels/(split+'.json');path.write_text(json.dumps(doc));original[split]=(doc,path.read_bytes())
    report=convert(review)
    assert report['changed']==1 and report['processed']==2
    for split,(doc,before) in original.items():
        assert (Path(report['backup'])/'session_manual_gt'/(split+'.json')).read_bytes()==before
        after=json.loads((labels/(split+'.json')).read_text())
        doc['objects'][0]['split']='eval'
        assert after==doc
    # A repeat is safe and does not rewrite already-EVAL labels.
    assert convert(review)['changed']==0
