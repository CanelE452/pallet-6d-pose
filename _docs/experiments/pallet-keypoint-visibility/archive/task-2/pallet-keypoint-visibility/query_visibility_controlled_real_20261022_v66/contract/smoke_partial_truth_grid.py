"""Compare independent vectorized counts with scalar shared-pixel completions."""
from itertools import product
from pathlib import Path
import hashlib,json,time
import numpy as np
import cv2
from paired_truth import construct_certainty,query_truth
from partial_truth_grid import partial_truth_grid


def main():
    cv2.setNumThreads(1);started=time.monotonic();patterns=native_cells=0
    templates=np.array([(1,0,1,0),(0,1,1,0),(0,1,0,1),(0,0,1,0),(0,1,0,0),(0,0,0,0)],bool)
    def check(cert):
        nonlocal native_cells
        grid=partial_truth_grid(cert);h,w=cert['exact_state'].shape
        for y in range(h):
            for x in range(w):
                q=query_truth(cert,[x,y])
                for k in ['exact_state','radius2_state','joint_state','joint_known','joint_compatible_mask','radius2_strict_E_core','radius2_strict_O_core','radius2_clipped_by_frame']:
                    assert grid[k][y,x]==q[k],(x,y,k,grid[k][y,x],q[k])
                assert grid['exact_possible_mask'][y,x]==sum(1<<j for j in q['exact']['possible_states'])
                assert grid['radius2_possible_mask'][y,x]==sum(1<<j for j in q['radius2']['possible_states'])
                assert grid['radius2_window_pixels'][y,x]==q['radius2']['pixels']
                native_cells+=1
    for n in (1,3,5):
        for choice in product(range(6),repeat=n):
            a=templates[np.asarray(choice)];check(construct_certainty(*[a[:,k][None,:]for k in range(4)]));patterns+=1
    rng=np.random.default_rng(20261022)
    for i in range(100):
        h,w=[int(v)for v in rng.integers(1,37,size=2)]
        # Half use complete visible negatives (controlled contract); half retain
        # arbitrary visible uncertainty (paired contract including teacher0).
        allowed=np.array([0,1,2,4])if i<50 else np.arange(6)
        a=templates[rng.choice(allowed,size=(h,w))]
        check(construct_certainty(*[a[:,:,k]for k in range(4)]))
    a=templates[rng.integers(0,6,size=(480,640))]
    cert=construct_certainty(*[a[:,:,k]for k in range(4)])
    t=time.monotonic();grid=partial_truth_grid(cert);large_seconds=time.monotonic()-t
    assert grid['joint_compatible_mask'].dtype==np.uint8 and grid['joint_compatible_mask'].shape==(480,640)
    try:partial_truth_grid(dict(visible_positive=np.ones((2,2),bool),visible_negative=np.zeros((2,2),bool),full_positive=np.zeros((2,2),bool),full_negative=np.zeros((2,2),bool)))
    except ValueError:pass
    else:raise AssertionError('Non-closed facts accepted')
    path=Path(__file__);result=dict(schema='V66_NATIVE_PARTIAL_TRUTH_GRID_MOCK_PROOF',PASS=True,
      helper_sha256=hashlib.sha256((path.parent/'partial_truth_grid.py').read_bytes()).hexdigest(),
      scalar_reference_sha256=hashlib.sha256((path.parent/'paired_truth.py').read_bytes()).hexdigest(),
      exhaustive_partial_patterns=patterns,random_native_grids=100,scalar_native_cells_compared=native_cells,
      patterns_include='All6 logically expressible closed pixel facts; complete-negative controlled and partial-visible paired cases',
      comparison='Exact/r2 possibilities, correlated6joint mask, unique states, strict25core, clipped window counts',
      native480x640_decoder_seconds=large_seconds,cv2_version=cv2.__version__,
      source_or_private_images_labels_scores_read=False,new_model_training_GPU_PnP_calls=0,seconds=time.monotonic()-started)
    (path.parent/'PARTIAL_TRUTH_GRID_MOCK_PROOF.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
