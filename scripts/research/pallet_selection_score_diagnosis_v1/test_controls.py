"""Small checks of cohort aliases and the separate-minima diagnostic contract."""
import pytest

from .baseline import group, truth
from .no_candidate import pair_summary
from .gallery import projection, select_representatives
import numpy as np


def test_group_aliases_use_original_oracle_fields():
    assert group({'final_joint_success':True,'earliest_observed_failure_stage':None})=='FINAL_SUCCESS_CONTROL'
    for stage,want in (
        ('family_selection','FAMILY_SELECTION_LOSS'),
        ('within_family_final_selection','WITHIN_FAMILY_SELECTION_LOSS'),
        ('confirmed_original_consumer_observation_or_proposal_limit','NO_SUCCESS_CANDIDATE')):
        assert group({'final_joint_success':False,'earliest_observed_failure_stage':stage})==want
    with pytest.raises(KeyError):
        group({'final_joint_success':False,'earliest_observed_failure_stage':'unknown'})


def test_unknown_boolean_is_not_failure_zero():
    assert truth('') is None
    assert truth('True') is True
    with pytest.raises(ValueError):
        truth('NA_WITHOUT_EXPLANATION')


def test_start_endpoint_counts_do_not_form_virtual_success():
    pairs=[{'stage':'G13','before_metric':{'success_le5':False,'translation_cm':2.,'rotation_deg':20.},
            'after_metric':{'success_le5':False,'translation_cm':20.,'rotation_deg':2.},
            'joint_success_to_failure':False}]
    x=pair_summary(pairs,'G13')
    assert x['starts_joint_success']==x['ends_joint_success']==0
    assert x['start_T_le5_count']==x['end_R_le5_count']==1
    assert x['start_R_le5_count']==x['end_T_le5_count']==0


def test_display_projection_keeps_original_native_origin():
    q=projection(np.eye(3),[0,0,4],[[0,0,0],[2,0,0]],[[100,0,50],[0,100,30],[0,0,1]])
    assert np.array_equal(q,[[50,30],[100,30]])


def test_representative_ranking_preserves_precommitted_recording_diversity():
    fs=[dict(id=k,partition='REMAINDER',group='FAMILY_SELECTION_LOSS')for k in('a','b','c')]
    metrics={k:{'E0':{'T_cm':t,'cluster_id':rec}}for k,t,rec in(('a',50,'R1'),('b',40,'R1'),('c',30,'R2'))}
    chosen=select_representatives(fs,metrics,set())
    assert [r['id']for r in chosen]==['a','c']
