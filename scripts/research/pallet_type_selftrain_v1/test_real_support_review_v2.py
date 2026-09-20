import copy
from .real_support_review_v2 import eligible


def test_eligibility_not_correctness():
    p={'selected_index':0,'candidates':[{'score':.7,'box_xyxy':[0,0,120,10],'keypoints_xy':[[30,30]]*9}]}
    assert eligible(p)  # Shape plausibility is intentionally NOT a filter.
    for patch in [{'score':.49},{'box_xyxy':[0,0,40,40]},{'keypoints_xy':[[-1,-1]]*9}]:
        q=copy.deepcopy(p);q['candidates'][0].update(patch);assert not eligible(q)
    assert not eligible({'selected_index':None,'candidates':[]})
