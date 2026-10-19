from copy import deepcopy

from scripts.research.pallet_bi_conditioned_observation_v1.test_solver import fixture,partial_selection
from scripts.research.pallet_bi_full_execution_v1 import qcov as Q,qcov_fixed_gamma as G


def test_covariance_donor_cannot_change_common_gamma_or_typed_archives():
    frame,common,_,_,truth=fixture()
    for row in common['full_points']:row['fit_reliability_weight']=.8
    common=partial_selection(frame,common,truth);bi=deepcopy(common)
    partial_ids={row['native_corner']for row in common['partial_lines']}
    for row in bi['full_points']:
        if row['native_corner']not in partial_ids:
            row['fit_reliability_weight']=1.;row['xy'][0]+=2;row['fit_covariance_px2']=[[9.,0.],[0.,16.]]
    for qb,cb in ((0,0),(1,0),(0,1),(1,1)):
        old,_=Q.fixed_common_packet(frame,common,bi,qb,cb);new,provenance=G.fixed_gamma_packet(frame,common,bi,qb,cb)
        assert new['partial_lines']==common['partial_lines']
        for a,b in zip(old['full_points'],new['full_points']):
            original=next(p for p in common['full_points']if p['native_corner']==b['native_corner'])
            assert b.get('fit_reliability_weight')==original.get('fit_reliability_weight')
            restored=deepcopy(b);restored['fit_reliability_weight']=a['fit_reliability_weight'];assert restored==a
        assert provenance['Gamma_donor']=='COMMON_FIXED_IN_ALL4CELLS'
        Q.parsed_solver_payload(frame,new)
    baseline,_=G.fixed_gamma_packet(frame,common,bi,0,0)
    assert Q.parsed_solver_payload(frame,baseline)==Q.parsed_solver_payload(frame,common)


def test_identical_gamma_is_exact_old_cell_reuse():
    frame,common,*_=fixture();bi=deepcopy(common)
    for row in bi['full_points']:row['xy'][0]+=2;row['fit_covariance_px2']=[[9.,0.],[0.,16.]]
    for qb,cb in ((0,0),(1,0),(0,1),(1,1)):
        old,_=Q.fixed_common_packet(frame,common,bi,qb,cb);new,_=G.fixed_gamma_packet(frame,common,bi,qb,cb)
        assert Q.parsed_solver_payload(frame,old)==Q.parsed_solver_payload(frame,new)
