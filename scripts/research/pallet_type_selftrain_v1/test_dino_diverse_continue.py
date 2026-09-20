from . import dino_diverse_continue as L


def test_extended_shuffle_keeps_partial_epoch_prefix():
    rows=list(range(37));old=L.X.sample_schedule(rows,30);new=L.X.sample_schedule(rows,80)
    assert new[:30]==old
    assert new[30]!=new[0]


def test_scope_restores_parent_paths_count_and_arms():
    old=(L.D.PHASE,L.D.DOC,L.D.RAW,L.D.STEPS,L.D.M,L.D.ARMS)
    with L.scope():assert L.D.PHASE=='dino_diverse_continue' and L.D.STEPS==20000 and L.D.ARMS==['SYN'] and L.D.M is L.V.V
    assert (L.D.PHASE,L.D.DOC,L.D.RAW,L.D.STEPS,L.D.M,L.D.ARMS)==old
