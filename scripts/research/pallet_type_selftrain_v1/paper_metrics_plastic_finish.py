"""Bridge canonical ':' IDs to archived '__' pose IDs by exact image/label paths.

The negative-inference protocol and its source remain immutable. Use this entry
point for scoring: python -m scripts.research.pallet_type_selftrain_v1.paper_metrics_plastic_finish
"""
from unittest.mock import patch
from . import paper_metrics_plastic as M


def main():
    C=M.C
    legacy=C.read(M.OLD/'AXIS_REVIEW_MANIFEST.json')['frames_list']
    by_image={r['image']:r for r in legacy}
    mapping={}
    for r in C.read(M.POS)['items']:
        old=by_image[r['image_path']]
        assert old['annotation']==r['gt_v2_path'] and old['object_type']==C.TYPES['PLASTIC']
        mapping[r['frame_id']]=old['frame_id']
    assert len(mapping)==len(set(mapping.values()))==194
    C.freeze(M.DOC/'POSE_ID_BRIDGE.json',dict(
        reason="Original paper manifest uses ':'; archived closure manifest uses '__'. Exact image and annotation identity required.",
        source=C.bound(__file__),parent_protocol=C.bound(M.DOC/'PROTOCOL.json'),mapping=mapping))
    original_positive=M.positive
    original_pose=M.pose
    def pose_positive(arm):
        payload,predictions=original_positive(arm)
        assert set(predictions)==set(mapping)
        return payload,{mapping[k]:v for k,v in predictions.items()}
    def mapped_pose(arm):
        with patch.object(M,'positive',pose_positive):
            return original_pose(arm)
    with patch.object(M,'pose',mapped_pose):
        M.score()


if __name__=='__main__':main()
