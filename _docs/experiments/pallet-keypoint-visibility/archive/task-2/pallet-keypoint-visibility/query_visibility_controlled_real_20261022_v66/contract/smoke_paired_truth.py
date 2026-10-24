"""Independent exhaustive completion oracle and symmetric numerical counterexamples."""
from pathlib import Path
from itertools import product
import hashlib,json,time
import numpy as np
import paired_truth as helper


def main():
    started=time.monotonic();patterns=completions=0
    # All expressible, logically closed partial per-pixel facts. Oracle enumerates
    # all physical V/E/O completions independently of helper region/joint code.
    templates=[((1,0,1,0),(0,)),((0,1,1,0),(1,)),((0,1,0,1),(2,)),
               ((0,0,1,0),(0,1)),((0,1,0,0),(1,2)),((0,0,0,0),(0,1,2))]
    pairs=((0,0),(1,0),(1,1),(2,0),(2,1),(2,2))
    for n in (1,3,5):
        for choice in product(range(6),repeat=n):
            facts=np.asarray([templates[i][0]for i in choice],bool)
            cert=helper.construct_certainty(*[facts[:,k][None,:]for k in range(4)])
            center=n//2;got=helper.query_truth(cert,[center,0]);possible=set();exs=set();nrs=set()
            for physical in product(*[templates[i][1]for i in choice]):
                ex=physical[center];near=0 if 0 in physical else 1 if 1 in physical else 2
                possible.add(pairs.index((ex,near)));exs.add(ex);nrs.add(near);completions+=1
            ex=next(iter(exs))if len(exs)==1 else -1;nr=next(iter(nrs))if len(nrs)==1 else -1
            assert got['joint_compatible_codes']==sorted(possible),(choice,got,possible)
            assert got['exact_state']==ex and got['radius2_state']==nr
            assert got['joint_known']==(ex!=-1 and nr!=-1)
            assert got['joint_compatible_mask']==sum(1<<v for v in possible)
            assert not got['radius2_strict_E_core']and not got['radius2_strict_O_core']
            patterns+=1
    shape=(9,9);yes=np.ones(shape,bool);no=np.zeros(shape,bool)
    def pair(a,b,m,ta=yes,tb=no,eligible=True,gap=50):
        return helper.construct_symmetric_pair(ta,tb,np.full(shape,a),np.full(shape,b),np.full(shape,m),pair_eligible=eligible,depth_gap_mm=gap)
    regular=pair(1000,800,1000)
    assert regular['A']['exact_state'][4,4]==0 and regular['B']['exact_state'][4,4]==1
    assert helper.query_truth(regular['B'],[4,4])['radius2_strict_E_core']
    swap=pair(800,1000,1000,ta=no,tb=yes)
    for side,other in [('A','B'),('B','A')]:
        for key in regular[side]:assert np.array_equal(regular[side][key],swap[other][key]),(side,key)
    far_margin_fail=pair(8000,7841,8000)
    far_margin_pass=pair(8000,7840,8000)
    assert far_margin_fail['B']['exact_state'][4,4]==-1
    assert far_margin_pass['B']['exact_state'][4,4]==1
    assert pair(8000,7950,8000)['B']['exact_state'][4,4]==-1
    assert pair(1000,950,1000)['B']['exact_state'][4,4]==1 # inclusive fixed50mm
    assert pair(1000,950.0001,1000)['B']['exact_state'][4,4]==-1
    for specimen in [pair(1000,800,1000,eligible=False),pair(1000,800,1000,ta=no,tb=no),pair(0,800,1000),pair(np.nan,800,1000)]:
        for side in ('A','B'):
            assert (specimen[side]['exact_state']==-1).all()
            assert not specimen[side]['full_negative'].any()
    for specimen in [regular,swap,far_margin_fail,far_margin_pass]:
        assert not specimen['A']['full_negative'].any()and not specimen['B']['full_negative'].any()
    # Unknown visibility is never inferred negative merely from absent teacher.
    va=np.zeros(shape,bool);va[4,4]=True
    cert=helper.construct_certainty(no,no,va,no)
    assert helper.query_truth(cert,[4,4])['exact']['possible_states']==[0,1]
    assert helper.query_truth(cert,[4,4])['radius2_state']==-1
    for q in [[np.nan,0],[-.01,0],[8.9,0],[9,0]]:assert not helper.query_truth(cert,q)['valid']
    assert helper.query_truth(cert,[8.49,0])['valid']
    for facts in [(yes,yes,yes,no),(yes,no,no,yes)]:
        try:helper.construct_certainty(*facts)
        except ValueError:pass
        else:raise AssertionError('Contradictory certainty accepted')
    path=Path(__file__);result=dict(schema='V66_PAIRED_TRUTH_EXHAUSTIVE_MOCK_PROOF',PASS=True,
      helper_sha256=hashlib.sha256((path.parent/'paired_truth.py').read_bytes()).hexdigest(),
      test_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),patterns=patterns,
      physical_completions=completions,all_expressible_pixel_fact_types=6,
      oracle='Enumerate physical V/E/O assignments; native1x1/1x3/1x5 windows; exact/shared-neighbor joint correlation',
      numerical_tests=['A/B swap equivariance','visible-reference2*tau margin counterexample159vs160mm at8m',
      'inclusive50mm boundary','unadmitted pair allUNKNOWN','no teachers allUNKNOWN',
      'missing depth allUNKNOWN','no full-negative/O created','no teacher-negative inference','native rounding/frame validity','contradiction rejection'],
      public_or_private_images_labels_scores_read=False,models_fitting_PnP_GPU_calls=0,seconds=time.monotonic()-started)
    out=path.parent/'PAIRED_TRUTH_MOCK_PROOF.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
