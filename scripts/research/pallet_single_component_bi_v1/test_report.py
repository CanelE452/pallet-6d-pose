"""Saved-report policy checks; no image, label or inference IO."""
from copy import deepcopy

from . import report as R


def test_eight_arm_table_has_full_denominator_NA_and_unavailable_counts():
    def summary():
        return dict(denominator=192,successes_le5=100,no_pose_n=2,accepted_n=180,
            fallback_n=10,acceptance_known_n=192,errors={k:dict(mean=8.,sample_std=2.,max=90.,n=190)
                for k in ('translation_cm','rotation_deg')})
    arms={a:summary() for a in R.ARMS}
    for a in ('R0','N3_F','SUBPIX_F','N0'):arms[a]['acceptance_known_n']=0
    text=R.table(dict(arms=arms))
    assert len([line for line in text.splitlines() if '100/192' in line])==8
    assert text.count('NA | NA | 2 |')==4
    assert '(n=190)' in text
    assert '90.000' in text


def test_success_rate_unit_and_damage_are_not_hidden_by_net_gain():
    comparison=dict(success_rate=dict(mean_difference=.125,CI95=[-.1,.2]),
        transition=dict(counts=dict(failure_to_success=30,success_to_failure=6,both_success=80,
            both_failure=76,reference_unresolved=0)))
    population=dict(comparisons={'N3_F-R0':comparison})
    sentence=R.effect_text(population,'N3_F-R0')
    assert '12.500%p' in sentence and '[-10.000, 20.000]%p' in sentence
    assert '회복30·손상6' in sentence
    assert '| 30 | 6 | 80 | 76 | 0 |' in R.paired_table(population)


def test_compliance_update_preserves_original_audit_and_remaining_limits():
    original=dict(requirements=[dict(id='single_components',status='not_done',finding='old'),
        dict(id='new_main_push',status='pending',finding='old'),
        dict(id='resources',status='partial',finding='not exactly timed')])
    before=deepcopy(original)
    result=R.compliance_final(original,dict(observed_main_head='a'*40),rows=1904,checks=2000)
    assert original==before and result['original_v1_audit_preserved']
    assert result['requirements'][0]['status']=='COMPLETE_NEW_INDIVIDUAL_COMPONENT_EXPERIMENT'
    assert result['requirements'][2]['status']=='partial'
    assert result['method_or_default_selection'] is False
    assert 'recorded separately' in result['new_publication_status']
    assert len(result['remaining_limits'])==5


def test_comparison_description_does_not_substitute_combined_for_individual():
    assert R.NAMES['N3_F']=='N3 단독 + F'
    assert R.NAMES['SUBPIX_BI']=='SubPix 단독 + B/I'
    assert 'SubPix 추가' in R.MEANING['N1-N3_BI']
    assert 'N3 추가' in R.MEANING['N1-SUBPIX_BI']
    assert len(R.MEANING)==12
