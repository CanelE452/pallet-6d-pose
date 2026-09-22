"""Small regression tests for the authorized same-session Clean19/300 pilot."""
import numpy as np
from scripts.research import pallet_replay_clean19_v1 as P


def test_no_overlap_and_fixed_counts():
    s=P.read(P.DOC/'SPLIT.json')
    assert len(s['train'])==19 and len(s['evaluation'])==300 and not s['withheld']
    for field in ('id',):
        assert not {r[field] for r in s['train']}&{r[field] for r in s['evaluation']}
    assert not {r['image']['sha256'] for r in s['train']}&{r['image']['sha256'] for r in s['evaluation']}
    assert len({r['image']['sha256'] for r in s['train']+s['evaluation']})==319


def test_clean_only_and_manual_supervision():
    s=P.read(P.DOC/'SPLIT.json');support=P.read(P.DOC/'TRAIN_SUPPORT.json')
    assert all(r['severity']=='CLEAN' and r['manual_count']>=3 for r in s['train'])
    assert support['manual']==support['supported']==87
    for r in support['records']:
        assert not r['manual'][8] and not r['crop_supported'][8]
        assert all(not v or m for v,m in zip(r['crop_supported'],r['manual']))


def test_user_authorized_session_overlap_not_hidden():
    s=P.read(P.DOC/'SPLIT.json')
    assert s['same_session_allowed_by_user'] and not s['recording_disjoint']
    assert {r['session'] for r in s['train']}&{r['session'] for r in s['evaluation']}
    assert s['near_duplicate_audit']['minimum_cross_split_MAD']>2


def test_protocol_fixed_no_eval_selection():
    p=P.read(P.DOC/'PROTOCOL.json')
    assert p['seed']==1 and p['steps']==300 and p['evaluation']==300
    assert p['selected_checkpoint']=='last300 only; no threshold/checkpoint/seed search'
    assert not p['auto_promote'] and not p['independent_unseen_test']
    assert not p['green_used']


def test_real_and_source_order_contract():
    order=dict(np.load(P.RAW/'ORDERS.npz'))
    assert order['real_rows'].shape==(300,8) and order['source_rows'].shape==(300,8)
    assert set(order['real_rows'].ravel())==set(range(19))
    assert not set(order['source_rows'].ravel())&set(order['held_rows'])
    historical=dict(np.load(P.N.RAW/'ORDERS.npz'))
    assert np.array_equal(order['source_rows'],historical['source_rows'])


def test_immutable_inputs():
    P.verify()
