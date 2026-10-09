"""Synthetic evidence and preservation tests; no real labels or pose calls."""
import numpy as np
import cv2

from . import methods as M


def fixture():
    image=np.full((280,340),210,np.uint8)
    truth=np.array([[100.,90.],[245.,115.],[225.,210.],[90.,180.]],np.float64)
    cv2.fillConvexPoly(image,truth.astype(np.int32),40,lineType=cv2.LINE_AA)
    points=np.full((9,2),-1.,np.float64);points[:4]=truth
    points[[0,3],0]-=20.;points[8]=[173.,147.]
    support=np.zeros(9,bool);support[:4]=True;support[8]=True
    return image,truth,points,support


def test_shared_boundary_recovers_large_offset_without_right_angles():
    image,truth,points,support=fixture()
    output,diag=M.correct(image,points,support,'BOUNDARY')
    assert np.linalg.norm(points[0]-truth[0])==20
    assert np.max(np.linalg.norm(output[:4]-truth,axis=1))<1.6
    assert 3 in diag['corner_records'][0]['selected_edges']
    assert 3 in diag['corner_records'][3]['selected_edges']
    shared=diag['edge_records'][3]
    residual=output[[0,3]]@shared['normal']+shared['offset']
    np.testing.assert_allclose(residual,0,atol=1e-9,rtol=0)
    # These projected polygon edges have no imposed image-space 90 degree angle.
    directions=np.diff(output[[3,0,1]],axis=0)
    cosine=abs(float(np.dot(*directions)/np.prod(np.linalg.norm(directions,axis=1))))
    assert cosine>.04


def test_no_two_line_evidence_retains_seed_on_flat_or_single_edge():
    _,_,points,support=fixture()
    flat=np.full((280,340),120,np.uint8)
    result,diag=M.correct(flat,points,support,'BOUNDARY')
    np.testing.assert_array_equal(result,points)
    assert diag['shared_edge_fits']==0
    edge=flat.copy();edge[:,100:]=30
    result,diag=M.correct(edge,points,support,'BOUNDARY')
    np.testing.assert_array_equal(result,points)
    assert all(r['status']!='refined' for r in diag['corner_records'])


def test_local_strong_distractors_do_not_make_full_supported_boundary():
    _,_,points,support=fixture();gray=np.full((280,340),128,np.uint8)
    # A strong black/white texture patch occurs at one endpoint but does not
    # supply a coherent edge across 60% of either incident projected edge.
    for y in range(75,106,4):
        gray[y:y+2,70:112]=255;gray[y+2:y+4,70:112]=0
    result,diag=M.correct(gray,points,support,'BOUNDARY')
    np.testing.assert_array_equal(result,points)
    assert all(r['status']!='refined' for r in diag['corner_records'])


def test_coherent_distractor_documents_semantic_failure():
    _,truth,points,support=fixture()
    # A separate coherent quadrilateral has exactly the same RGB geometry as
    # a plausible pallet boundary. With no semantics, the method follows it.
    # This test deliberately records that limitation rather than claiming an
    # image-only consistency score certifies the target object's identity.
    distracting=truth+np.array([-12.,0.])
    gray=np.full((280,340),210,np.uint8)
    cv2.fillConvexPoly(gray,distracting.astype(np.int32),40,lineType=cv2.LINE_AA)
    result,diag=M.correct(gray,points,support,'BOUNDARY')
    assert np.max(np.linalg.norm(result[:4]-distracting,axis=1))<1.6
    assert np.min(np.linalg.norm(result[:4]-truth,axis=1))>10
    assert diag['shared_edge_fits']>=4


def test_missing_center_support_and_outside_contract_both_methods():
    image,_,points,support=fixture()
    points[2]=[np.nan,np.nan];points[4]=[-1,-1];points[5]=[-12,90]
    points[6]=[180,120];support[2]=True;support[4]=True;support[5]=True;support[6]=False
    original=points.copy();mask=support.copy()
    for method in M.METHODS:
        result,diag=M.correct(image,points,support,method)
        np.testing.assert_array_equal(result[[2,4,5,6,8]],points[[2,4,5,6,8]])
        np.testing.assert_array_equal(points,original);np.testing.assert_array_equal(support,mask)
        assert result.dtype==np.float64 and result.shape==(9,2)
        assert diag['center_preserved'] and not diag['GT_inputs'] and not diag['external_cap_applied']


def test_paired_endpoint_unavailable_retains_both_seeds():
    image,_,points,support=fixture()
    for unavailable in ('unsupported','outside','missing'):
        initial=points.copy();mask=support.copy()
        if unavailable=='unsupported':mask[3]=False
        elif unavailable=='outside':initial[3]=[-5.,180.]
        else:initial[3]=[-1.,-1.]
        result,diag=M.correct(image,initial,mask,'BOUNDARY')
        np.testing.assert_array_equal(result[[0,3]],initial[[0,3]])
        assert not diag['paired_records'][0]['accepted']
        assert diag['paired_records'][0]['status']=='paired_endpoint_unavailable'


def test_one_endpoint_intersection_missing_rejects_whole_pair():
    _,_,points,support=fixture();gray=np.full((280,340),210,np.uint8)
    # The common 0/3 height edge and the top edge support corner0. No bottom
    # boundary is present anywhere in the corner3 normal-search corridor.
    gray[90:,100:]=40
    result,diag=M.correct(gray,points,support,'BOUNDARY')
    assert diag['edge_records'][3]['supported']
    assert '0' in diag['paired_records'][0]['endpoint_proposals']
    assert '3' not in diag['paired_records'][0]['endpoint_proposals']
    assert not diag['paired_records'][0]['accepted']
    np.testing.assert_array_equal(result[[0,3]],points[[0,3]])


def test_all_accepted_pairs_lie_on_the_same_required_height_line():
    image,_,points,support=fixture();result,diag=M.correct(image,points,support,'BOUNDARY')
    assert diag['accepted_pairs']==2 and diag['atomic_paired_endpoints']
    for pair in diag['paired_records']:
        if not pair['accepted']:continue
        first,second=pair['corners'];shared=pair['shared_edge'];line=diag['edge_records'][shared]
        np.testing.assert_allclose(result[[first,second]]@line['normal']+line['offset'],0,atol=1e-9,rtol=0)
        assert shared in diag['corner_records'][first]['selected_edges']
        assert shared in diag['corner_records'][second]['selected_edges']


def test_wide_subpix_exception_retains_input(monkeypatch):
    image,_,points,support=fixture()
    def failure(*args,**kwargs):raise cv2.error('synthetic OpenCV failure')
    monkeypatch.setattr(cv2,'cornerSubPix',failure)
    result,diag=M.correct(image,points,support,'WIDE_SUBPIX')
    np.testing.assert_array_equal(result,points)
    assert diag['algorithm_corner_calls']==4
    assert diag['status_counts']['function_error']==4


def run():
    # Root invokes this lightweight fixture gate before any real evaluation.
    test_shared_boundary_recovers_large_offset_without_right_angles()
    test_no_two_line_evidence_retains_seed_on_flat_or_single_edge()
    test_local_strong_distractors_do_not_make_full_supported_boundary()
    test_coherent_distractor_documents_semantic_failure()
    test_missing_center_support_and_outside_contract_both_methods()
    test_paired_endpoint_unavailable_retains_both_seeds()
    test_one_endpoint_intersection_missing_rejects_whole_pair()
    test_all_accepted_pairs_lie_on_the_same_required_height_line()
    return dict(status='PASS',tests=8,shared_edge_large_offset_recovered=True,
        two_line_evidence_required=True,local_texture_false_boundary_rejected=True,
        coherent_semantic_distractor_failure_documented=True,center_missing_support_outside_preserved=True,
        atomic_paired_endpoints=True,all_accepted_pairs_share_required_height_line=True,
        neural_forwards=0,pose_calls=0,real_annotation_access=False)


if __name__=='__main__':
    import json
    print(json.dumps(run()))
