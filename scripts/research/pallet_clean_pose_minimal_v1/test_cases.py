import numpy as np
import pytest

from .cases import approved_bindings,check_public_image,choose,project_saved_pose,safe_tag


def metric(t,r,available=True):
    return dict(translation_cm=t,rotation_deg=r,available=available)


def test_cases_rank_identically_with_restricted_scope_and_stable_id_tie():
    rows=[dict(id=x,severity='SEVERE',recording='rec') for x in ['b','a','c','d','e']]
    before={r['id']:metric(10,8) for r in rows}
    after=dict(a=metric(8,7),b=metric(8,6),c=metric(30,30),d=metric(9,9),e=metric(None,None,False))
    full=choose(before,after,rows)
    public=choose(before,after,rows,{'b','d'})
    assert [r['id'] for r in full['both_improved']['examples']]==['a','b']
    assert [r['id'] for r in public['both_improved']['examples']]==['b']
    assert public['both_worsened']['status']=='NA'
    assert [r['id'] for r in full['both_worsened']['examples']]==['c']
    assert full['largest_final_T']['examples'][0]['id']=='c'
    assert full['failure_frames']==[dict(id='e',before_valid=True,after_valid=False)]
    assert full['both_worsened']['examples'][0]['repeated_across_categories']


def test_public_requires_prior_id_and_exact_historical_image_binding():
    binding=dict(path='old/image.png',sha256='fixed',bytes=10)
    approved=approved_bindings(dict(examples=[dict(frame_id='yes')]),
        [dict(id='yes',image=binding),dict(id='no',image=binding)],
        dict(examples=[dict(id='yes',material='PLASTIC',status='AVAILABLE',image=binding)]))
    check_public_image('yes',binding,approved)
    with pytest.raises(AssertionError):check_public_image('no',binding,approved)
    with pytest.raises(AssertionError):check_public_image('yes',dict(binding,sha256='changed'),approved)
    with pytest.raises(AssertionError):approved_bindings(dict(examples=[dict(frame_id='absent')]),[],dict(examples=[]))


def test_saved_pose_projection_uses_cf_pose_and_no_reference_or_solver():
    pose=dict(available=True,cf_extents=[2.,2.,2.],R_cf=np.eye(3).tolist(),centroid=[0.,0.,5.])
    K=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
    projected=project_saved_pose(pose,K)
    np.testing.assert_allclose(projected[0],[25.,25.])
    np.testing.assert_allclose(projected[6],[50.+100./6,50.+100./6])
    assert project_saved_pose(dict(available=False),K) is None
    behind=dict(pose,centroid=[0.,0.,-5.])
    assert np.isnan(project_saved_pose(behind,K)).all()


def test_empty_approved_subset_na_and_no_fallback_publication():
    row=dict(id='a',severity='MODERATE',recording='r')
    result=choose({'a':metric(10,8)},{'a':metric(9,7)},[row],set())
    assert result['eligible_frames']==0
    assert all(result[k]['status']=='NA' for k in ('both_improved','both_worsened','largest_final_T','largest_final_R'))


def test_tag_cannot_escape_new_namespace():
    assert safe_tag('FINAL_s42')=='FINAL_s42'
    for bad in ('../old','a/b','','a.png','a b'):
        with pytest.raises(AssertionError):safe_tag(bad)
