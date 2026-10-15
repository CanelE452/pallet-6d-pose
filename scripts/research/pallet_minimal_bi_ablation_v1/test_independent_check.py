import math
import numpy as np
import pytest
from scripts.research.pallet_minimal_bi_ablation_v1 import independent_check as I


def test_sample_standard_deviation_keeps_missing_denominator():
    d=I.sample_distribution([1.,3.,None])
    assert d==dict(n=2,denominator=3,missing=1,mean=2.,sample_std=math.sqrt(2),max=3.)


def test_recording_bootstrap_is_frame_weighted_and_same_draws():
    rows=[dict(cluster_id='A',session='a')]*3+[dict(cluster_id='B',session='b')]
    weights,meta=I.frame_resample_weights(rows)
    expected=np.random.default_rng(20260917).multinomial(2,[.5,.5],size=10000).astype(np.uint16)
    assert np.array_equal(weights,expected[:,[0,0,0,1]])
    d=I.paired_interval([0.,0.,0.,1.],rows,weights)
    assert d['mean_difference']==.25
    assert d['CI95']==[0.,1.]
    assert meta['units']==2


def test_threshold_transitions_retain_all_failures_and_unresolved():
    rows=[]
    for k,(a,b) in enumerate([(True,False),(False,True),(True,True),(False,False),(None,False)]):
        rows.append(dict(id=str(k),metrics={'R1':{'success_le5':a},'R0':{'success_le5':b}}))
    assert I.transition_table(rows,'R1','R0')['counts']==dict(failure_to_success=1,success_to_failure=1,both_success=1,both_failure=1,reference_unresolved=1)


def test_no_GO_stops_before_any_IO(tmp_path):
    with pytest.raises(PermissionError):
        I.run(score_folder=tmp_path/'absent',input_seal=tmp_path/'none',expected_score_seal_sha256='0'*64,output=tmp_path/'output')
    assert not (tmp_path/'output').exists()
