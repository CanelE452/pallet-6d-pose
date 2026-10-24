"""Generated-only discrete coverage tests, no corpus/model/current I/O."""
import hashlib,json
from pathlib import Path
import numpy as np
from opaque_pair_geometry import donor_geometry,rotate,compose
HERE=Path(__file__).resolve().parent
patterns=[]
patterns.append(np.ones((1,1),bool));patterns.append(np.ones((1,19),bool));patterns.append(np.ones((17,1),bool));patterns.append(np.ones((13,25),bool))
x=np.zeros((21,23),bool);x[2:19,2:7]=1;x[14:19,2:21]=1;patterns.append(x)
x=np.ones((19,19),bool);x[6:13,6:13]=0;patterns.append(x)
cases=[]
for mi,m in enumerate(patterns):
    g=donor_geometry(m);image=np.full((*g['mask'].shape,3),[30,90,180],np.uint8)
    for turn in range(4):
        mask,rgb,c=rotate(g['mask'],image,g['center_xy'],turn)
        for r in [[0.,0.],[.231,.817],[31.333,29.789],[63.8,47.7],[-.1,25.4]]:
            for rho in [4.,4.001,10.333,37.8]:
                before=np.full((48,64,3),[200,150,100],np.uint8)
                after,paste,info=compose(before,rgb,mask,c,g['interior_distance'],r,rho)
                assert info['disk_all_covered']
                np.testing.assert_array_equal(after[~paste],before[~paste])
                assert paste.dtype==bool and after.dtype==np.uint8
                cases.append(dict(pattern=mi,turn=turn,r=r,rho=rho,d=g['interior_distance'],scale=info['scale']))
out=HERE/'GENERATED_DISCRETE_COVERAGE_TESTS.json';assert not out.exists()
out.write_text(json.dumps({'PASS':True,'cases':len(cases),'patterns':['singlepixel','horizontal1px','vertical1px','rectangle','Lshape','hole'],
 'all4quarterturns':True,'fractional_reference_and_frameborder_clipping':True,'unpaintedRGB_exact':True,'details':cases,
 'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),HERE/'opaque_pair_geometry.py']},
 'source_current_RGB_masks_models_weights_reads':0,'forward_fit_PnP':0},indent=2)+'\n')
print('GENERATED_COVERAGE_PASS',len(cases),hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
