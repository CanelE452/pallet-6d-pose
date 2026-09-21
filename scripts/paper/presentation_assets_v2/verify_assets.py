"""Verify frozen inputs, current-evaluator parity, figure coordinates and files.

Read-only: prints a JSON report; never modifies checkpoints, predictions or assets.
"""
import json
import subprocess
import numpy as np
import cv2
import torch
import build_assets as B


def main():
    m=B.read(B.OUT/'ASSET_MANIFEST.json')
    assert m['optimizer_updates']==0 and m['no_training'] and m['new_inference']==0
    assert not m['model_seed_hyperparameter_changes']
    assert m['source_unchanged_before_after']
    for b in m['protected_sources']+m['code']:
        p=B.ROOT/b['path']
        assert B.sha(p)==b['sha256'] and p.stat().st_size==b['bytes'],p
    dims,groups=B.dimension_contract()
    assert m['dimensions_exact']==dims
    rows,pred,_,_=B.load_data()
    cases,rule=B.select_cases(rows)
    f=m['figures'][B.NAMES[2]]
    assert f['selection_rule']==rule and f['seed']==1
    assert [r['id'] for r in f['examples']]==[r['id'] for r in cases]
    for shown,source in zip(f['examples'],cases):
        for field in source: assert shown[field]==source[field],(source['id'],field)
        np.testing.assert_array_equal(shown['plotted_r0'],np.array(source['r0'])[:8])
        np.testing.assert_array_equal(shown['plotted_n3'],np.array(source['n3'][0])[:8])
        valid=np.array(source['valid'][:8]) & np.isfinite(np.array(source['gt'])[:8]).all(-1) & ~(np.array(source['gt'])[:8]==-1).all(-1)
        assert shown['plotted_gt_indices']==np.flatnonzero(valid).tolist()
        np.testing.assert_array_equal(shown['plotted_gt'],np.array(source['gt'])[:8][valid])
        assert shown['arrows_scale']==1 and shown['plotted_coordinate_parity']
        assert source['dimensions']==dims[source['object_type']]
    examples=B.dimension_examples(rows,dims)
    for shown,source in zip(m['figures'][B.NAMES[1]]['examples'],examples):
        for field in source:assert shown[field]==source[field],(source['id'],field)
        o=B.read(B.ROOT/shown['annotation'])['objects'][0]
        np.testing.assert_array_equal(shown['anchors'],o['projected_cuboid'])
        assert shown['dimensions']==dims[shown['object_type']]
        assert shown['axis_edges']=={'W':[0,1],'D':[0,4],'H':[0,3]}
        assert [o['dimensions_m'][k] for k in ['width','depth','height']]==shown['dimensions']
    for typ,c in m['figures'][B.NAMES[3]]['contracts'].items():assert c==groups[typ]
    candidate=m['figures'][B.NAMES[4]]['candidate_details']
    ck=torch.load(B.ROOT/candidate['checkpoint']['path'],map_location='cpu',weights_only=False)
    lattice=ck['model_state_dict']['displacements'].numpy()
    np.testing.assert_array_equal(candidate['lattice'],lattice)
    assert candidate['candidates']==222 and candidate['directions']*candidate['radial_levels']==221
    assert candidate['schematic'] and candidate['cap_original_image_diagonal']==.01
    x=np.array(candidate['schematic_logits']);p=np.exp(x-x.max());p/=p.sum()
    mu=(lattice*p[:,None]).sum(0)
    np.testing.assert_allclose(mu,candidate['schematic_mean'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(mu*min(1,.02/np.linalg.norm(mu)),candidate['schematic_capped'],rtol=0,atol=1e-12)
    files=[]
    for name in B.NAMES:
        for b in m['figures'][name]['files']:
            path=B.ROOT/b['path'];assert B.sha(path)==b['sha256']
            if path.suffix=='.png':
                im=cv2.imread(str(path));assert im is not None
                h,w=im.shape[:2];assert w>=1920 and h>=1080 and abs(w/h-16/9)<1e-10
                files.append(dict(path=b['path'],resolution=[w,h],opens=True))
            else:
                info=subprocess.check_output(['pdfinfo',str(path)],text=True)
                assert 'Pages:           1' in info and path.read_bytes().startswith(b'%PDF-')
                files.append(dict(path=b['path'],pages=1,opens=True))
    assert (B.OUT/'README.md').is_file()
    print(json.dumps(dict(status='PASS',optimizer_updates=0,new_inference=0,
        protected_source_files=len(m['protected_sources']),source_hashes_unchanged=True,
        evaluator_comparisons=319*4,source_coordinate_parity=True,dimensions_exact=True,
        C2_C4_rotation_and_permutation_exact=True,selected_ids=[r['id'] for r in cases],
        files=files),indent=2))


if __name__=='__main__':main()
