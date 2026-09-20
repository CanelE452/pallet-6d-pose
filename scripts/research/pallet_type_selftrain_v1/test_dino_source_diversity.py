import numpy as np
from . import dino_source_diversity as X


def test_selection_is_existing_train_only_retains_parent():
    rows=X.choose_rows(np.arange(100),[1,8,12],32)
    assert rows==X.choose_rows(np.arange(100),[1,8,12],32)
    assert len(rows)==len(set(rows))==32 and {1,8,12}<=set(rows)<=set(range(100))


def test_complete_epoch_schedule_and_fixed_count():
    rows=list(range(32));schedule=X.sample_schedule(rows,10)
    assert np.shape(schedule)==(10,8) and schedule==X.sample_schedule(rows,10)
    flat=np.asarray(schedule).ravel()
    assert set(flat[:32])==set(flat[32:64])==set(rows)


def test_lazy_feature_stack_and_cache_bounded(tmp_path):
    X.read_feature.cache_clear();features=[]
    for i in range(3):
        path=tmp_path/f'{i}.npz';np.savez(path,feature=np.full((2,3),i,dtype=np.float16));features.append(X.LazyFeature(path))
    np.testing.assert_array_equal(np.stack(features),np.stack([np.full((2,3),i) for i in range(3)]))
    assert X.read_feature.cache_info().maxsize==128
    X.read_feature.cache_clear()


def test_scope_restores_all_parent_settings():
    old=(X.D.PHASE,X.D.DOC,X.D.RAW,X.D.STEPS,X.D.M)
    with X.scope():assert X.D.PHASE=='dino_source_diversity' and X.D.STEPS==5000 and X.D.M is X.V.V
    assert (X.D.PHASE,X.D.DOC,X.D.RAW,X.D.STEPS,X.D.M)==old
