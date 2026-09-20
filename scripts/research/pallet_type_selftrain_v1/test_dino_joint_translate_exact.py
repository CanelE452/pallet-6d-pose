from . import dino_joint_translate_exact as X


def row(view,gain,before,after):
    return dict(view=view,proposal=dict(available=True,permutation=0,delta_crop=[4.,0.],gain=gain),errors=[[before]*8,[after]*8])


def test_frontier_does_not_drop_gains_above16():
    rows=[row(0,18.,1.,20.),row(1,19.,100.,1.)]
    best,grid=X.calibrate(rows)
    assert best['threshold']==18. and best['combined']['selected']==1 and best['combined']['recovered']==8
    assert {r['threshold'] for r in grid}=={0.,18.,19.,1e9}


def test_no_beneficial_choice_retains_null():
    best,_=X.calibrate([row(0,10.,1.,20.),row(1,19.,1.,20.)])
    assert best['threshold']==1e9 and best['combined']['selected']==0


def test_scope_restores_parent_paths():
    old=(X.T.PHASE,X.T.DOC,X.T.RAW)
    with X.scope():assert X.T.PHASE=='dino_joint_translate_exact' and X.T.DOC==X.DOC
    assert (X.T.PHASE,X.T.DOC,X.T.RAW)==old
