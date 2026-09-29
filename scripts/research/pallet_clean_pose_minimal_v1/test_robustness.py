from .robustness import cluster_interval,select_cases


def metric(t,r,valid=True):
    return dict(available=valid,translation_cm=t,rotation_deg=r)


def test_cluster_uses_recordings_and_same_pairs():
    rows=[dict(id=str(i),recording='A' if i<3 else 'B') for i in range(4)]
    before={r['id']:metric(10,8) for r in rows}
    after={r['id']:metric(9,6) for r in rows}
    result=cluster_interval(before,after,rows,repeats=32)
    assert result['recording_count']==2
    assert result['recording_sizes']==dict(A=3,B=1)
    assert result['intervals']['translation_cm']['percentile95']==[-1,-1]
    assert result['intervals']['rotation_deg']['percentile95']==[-2,-2]


def test_case_failure_and_tradeoff_not_joint_gain():
    rows=[dict(id=str(i),recording='A',severity='CLEAN') for i in range(4)]
    before={str(i):metric(10,8) for i in range(4)}
    after={'0':metric(8,6),'1':metric(20,16),'2':metric(7,9),'3':metric(None,None,False)}
    cases=select_cases(before,after,rows)
    assert [r['id'] for r in cases['both_improved']]==['0']
    assert [r['id'] for r in cases['both_worsened']]==['1']
    assert cases['failure_frames']==[dict(id='3',before_valid=True,after_valid=False)]
