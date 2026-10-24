from pathlib import Path
import tempfile,json,hashlib
import numpy as np
from PIL import Image
import infer_fixed_four as r
import canonical_decode_four as d
H=Path(__file__).resolve().parent
q=np.array([[100.+k,120.+2*k]for k in range(9)]);c=lambda score,pts:dict(score=score,keypoints_xy=pts)
z=dict(candidates=[c(.8,q),c(.8,q+20)],selected_index=0);a=r.selected_queries(z,[1024,1296]);assert len(a)==8 and a[0]['query_xy']==q[0].tolist()
q2=q.copy();q2[8]=[100000.,100000.];assert r.selected_queries(dict(candidates=[c(.8,q2)],selected_index=0),[1024,1296])==a
no=r.selected_queries(dict(candidates=[],selected_index=None),[1024,1296]);assert len(no)==8 and all(x['query_xy']is None and x['roi_side']is None for x in no)
for xy,cont,near in [([-.25,10.],False,True),([1295.75,10.],True,False),([0.,0.],True,True),([1296.,0.],False,False),([-100.,200.],False,False)]:
 qq=q.copy();qq[0]=xy;x=r.selected_queries(dict(candidates=[c(.9,qq)],selected_index=0),[1024,1296])[0];assert x['query_xy']==xy and x['original_query_continuous_valid']==cont and x['query_nearest_pixel_valid']==near and x['query_rgb96_centre_domain_valid']==near
with tempfile.TemporaryDirectory(prefix='v104_generated_decode_')as tmp:
 p=Path(tmp)/'generated.jpg';Image.fromarray(np.random.default_rng(194).integers(0,256,(17,23,3),np.uint8)).save(p);rgb=np.array(Image.open(p).convert('RGB'));row=dict(image=str(p),image_sha256=d.sha(p),raw_hw=[17,23],decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());assert np.array_equal(d.decode_and_check(row),rgb)
 try:d.decode_and_check(dict(row,decoded_RGB_sha256='0'*64))
 except AssertionError:pass
 else:raise AssertionError('Decoder mismatch accepted')
proof=dict(completed=True,PASS=True,initializer_code=r.bind(r.__file__),decoder_code=d.bind(d.__file__),test=r.bind(__file__),native_grid_code=r.bind(r.ROI),firstscore_tie_first=True,all8_nodetection_null_preserved=True,finiteOOB_unchanged=True,separate_validity_bounds=True,centroid_excluded_from_L=True,generated_JPEG_decode_exact_and_wrong_SHA_rejected=True,source_RGB_checkpoint_model_forward_fit_current_PnP=0)
r.write(H/'GENERATED_QUERY_AND_DECODER_TESTS.json',proof);print(json.dumps(proof))
