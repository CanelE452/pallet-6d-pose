"""Check adoption language and complete saved-failure presentation boundaries."""
from copy import deepcopy

from . import report as R


def fixture():
    rows=[]
    for ident,partition,e0,eb in [('old','TEST',True,True),('damage','REMAINDER',True,False),
                                 ('recover','REMAINDER',False,True)]:
        rows.append(dict(id=ident,partition=partition,metrics={
            'E0':dict(success_le5=e0),'EB':dict(success_le5=eb)}))
    results=dict(populations={'PRIMARY':{'arms':{
        'E0':dict(successes_le5=2,denominator=3),
        'EB':dict(successes_le5=2,denominator=3)}}})
    return results,rows


def test_CAL_choice_with_equal_net_score_and_one_damage_is_not_adoptable():
    results,rows=fixture()
    result=R.recommendation(results,{'selected_arm':'EB'},rows)
    assert result['selected_arm']=='EB'
    assert result['label']=='CAL_SELECTED_BUT_PRIMARY_NOT_ADOPTABLE'
    assert result['PRIMARY_versus_E0']['counts']['success_to_failure']==1
    assert not result['default_adoption_recommended']
    assert not result['all192_solved'] and result['primary_failures']==1


def test_unselected_PRIMARY_better_arm_does_not_replace_CAL_choice():
    results,rows=fixture()
    results['populations']['PRIMARY']['arms']['EA']=dict(successes_le5=3,denominator=3)
    out=R.recommendation(results,{'selected_arm':'EB'},rows)
    assert out['selected_arm']=='EB' and '실패·미채택' in out['text']


def test_original_E_selected_does_not_become_new_method_success():
    results,rows=fixture()
    out=R.recommendation(results,{'selected_arm':'E0'},rows)
    assert out['label']=='NO_NEW_METHOD_ADOPTED'
    assert out['primary_failures']==1 and not out['all192_solved']
    assert '새 B/I 보정 후보는 최종 방법으로 채택하지 않는다' in out['text']


def test_no_damage_result_is_only_research_candidate_and_keeps_failures():
    results,rows=fixture()
    rows[1]['metrics']['EB']['success_le5']=True
    results['populations']['PRIMARY']['arms']['EB']['successes_le5']=3
    out=R.recommendation(results,{'selected_arm':'EB'},rows)
    assert out['label']=='CAL_SELECTED_RESEARCH_CANDIDATE_ONLY'
    assert not out['default_adoption_recommended']


def test_markdown_table_rows_are_contiguous():
    text=R.paragraphs(['문단','| A | B |','|---|---|','| 1 | 2 |','다음 문단'])
    assert '| B |\n|---|' in text
    assert '|---|---|\n| 1 |' in text
    assert '문단\n\n|' in text and '| 2 |\n\n다음' in text


def test_none_metric_is_explicit_NA():
    assert R.mean_sd(dict(mean=None,sample_std=None,n=0))=='NA±NA (n=0)'
