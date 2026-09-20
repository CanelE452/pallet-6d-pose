from .real_support_review import closure, select_spread


def test_transitive_alias_and_partial_overlap():
    groups={'groups':[{'is_collection':False,'sessions':[{'session_key':'a'},{'session_key':'alias_a'}]}],
            'partial_overlap_pairs':[{'session_a':'alias_a','session_b':'b'},{'session_a':'b','session_b':'c'}]}
    assert closure({'a'},groups)=={'a','alias_a','b','c'}
    assert closure({'safe'},groups)=={'safe'}


def test_validation_never_depends_on_uncertainty():
    rows=[{'id':str(i),'image':{'path':f'{i:03d}.png'},'review_uncertainty':i} for i in range(100)]
    a=select_spread(rows,10,True)
    b=select_spread([dict(r,review_uncertainty=-r['review_uncertainty']) for r in rows],10,True)
    assert [r['id'] for r in a]==[r['id'] for r in b]
    assert len({r['temporal_bin'] for r in a})==10


def test_support_has_both_priority_and_control_not_labels():
    rows=[{'id':str(i),'image':{'path':f'{i:03d}.png'},'review_uncertainty':i} for i in range(100)]
    selected=select_spread(rows,10)
    assert len({r['id'] for r in selected})==10
    assert sum('control' in r['selection_reason'] for r in selected)==3
    assert all('NOT_verified' in r['selection_reason'] for r in selected)
