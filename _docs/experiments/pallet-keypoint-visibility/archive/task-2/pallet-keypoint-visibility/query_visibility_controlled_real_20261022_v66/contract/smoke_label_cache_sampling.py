"""Native-grid sampling phase/bounds mock; no source/private data read."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import paired_truth
from partial_truth_grid import partial_truth_grid
from build_partial_label_caches import sample

def main():
    root=Path(__file__).resolve().parents[1]
    path=root.parent/'query_visibility_amodal_global_20261022_v65/global_recipe.py'
    spec=importlib.util.spec_from_file_location('frozen_global_recipe_mock',path)
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    rng=np.random.default_rng(20261022);h,w=17,23
    templates=np.array([(1,0,1,0),(0,1,1,0),(0,1,0,1),(0,0,1,0),(0,1,0,0),(0,0,0,0)],bool)
    a=templates[rng.integers(0,6,size=(h,w))];cert=paired_truth.construct_certainty(*[a[:,:,k]for k in range(4)])
    grid=partial_truth_grid(cert)
    queries=np.array([[0.,0.],[22.49,16.49],[22.5,16.5],[3.49,4.51],[10.,8.],[8.5,8.5],[1.1,15.8],[20.8,1.2]])
    side=9.7;axis=(np.arange(9)-4)*side/9;xx,yy=np.meshgrid(axis,axis)
    xy=queries[:,None,None,:]+np.stack([xx,yy],-1)[None]
    assert np.array_equal(xy[:,4,4],queries)
    boxes=[[-1.,-2.,25.,20.],[2.,3.,17.,13.]]
    tested=0
    for coordinates in [xy]+[recipe.grid(b)[0]for b in boxes]:
        got=sample(grid,coordinates,recipe)
        for idx in np.ndindex(coordinates.shape[:-1]):
            q=coordinates[idx];truth=paired_truth.query_truth(cert,q)
            assert bool(got['valid'][idx])==truth['valid']
            if truth['valid']:
                for key in ['exact_state','radius2_state','joint_known','joint_state']:assert got[key][idx]==truth[key]
                assert got['allowed'][idx]==truth['joint_compatible_mask']
            else:
                assert got['allowed'][idx]==63 and not got['joint_known'][idx]
                assert got['exact_state'][idx]==got['radius2_state'][idx]==got['joint_state'][idx]==-1
            tested+=1
    report=dict(schema='V66_PARTIAL_LABEL_CACHE_SAMPLING_MOCK_PROOF',PASS=True,native_samples_compared=tested,
      native9_query_centers_unchanged=True,invalid_allowed63_validfalse=True,
      global64_recipe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      builder_sha256=hashlib.sha256((root/'contract/build_partial_label_caches.py').read_bytes()).hexdigest(),
      source_or_private_data_read=False,features_models_GPU_training_PnP_calls=0)
    (root/'contract/LABEL_CACHE_SAMPLING_MOCK_PROOF.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
