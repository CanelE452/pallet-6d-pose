"""Reconstruct source-only dense validity, exact targets, context and M0 states."""
import csv,gzip,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
TASK=BASE.parent.parent
V60=TASK/'external_ray_proxy_head_20261022_v60'
V61=TASK/'target_presence_classifier_20261022_v61'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

inputs=json.load(gzip.open(V60/'PREPARED_INPUTS_SYNTH.json.gz','rt'))
metadata=json.loads((V60/'SYNTH_FEATURE_METADATA.json').read_text())
original=np.load(V60/'SYNTH_ROI_CACHE.npz',allow_pickle=False)['x']
original_labels=np.load(V61/'PRESENCE_LABELS.npz',allow_pickle=False)
maskpaths={r['id']:r for r in csv.DictReader((V61/'MASK_CONTRACT_AUDIT.csv').open())}
deny=set(json.loads((HERE/'EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
by={};bindings=[];source_splits={}
for split in ['train','validation']:
    p=BASE/'source_cache/v2'/('SYNTH_'+split.upper()+'.npz')
    with np.load(p,allow_pickle=False) as archive:
        z={name:archive[name]for name in archive.files}
        assert z['context'].shape==(len(z['y']),64)
        assert np.isfinite(z['context']).all()
        assert set(z['image_sha256']).isdisjoint(deny)
        for j,(ident,k) in enumerate(zip(z['id'].astype(str),z['native_corner'].astype(int))):
            assert (ident,k) not in by
            by[ident,k]={name:z[name][j].copy() for name in ['x','y','dense_y','dense_valid','context','image_sha256']}
            h=str(z['image_sha256'][j]);assert h not in source_splits or source_splits[h]==split;source_splits[h]=split
        bindings.append(dict(path=str(p),sha256=sha(p),queries=len(z['y']),positive=int(z['y'].sum()),frames=len(set(z['id']))))
rows=[];dense_invalid_repaired=Counter();stats=Counter();context_max=0.;context_per_frame={}
for i,u in enumerate(inputs):
    ident=u['id'];split=u['split'];assert metadata[i]['id']==ident
    a=maskpaths[ident]
    full=np.asarray(Image.open(a['amodal_path']))>127
    vis=np.asarray(Image.open(a['visible_path']))>127
    assert full.shape==vis.shape and not (vis&~full).any()
    h,w=vis.shape;L=float(metadata[i]['side'])
    vc=original[i,:,64].astype(np.float32)
    expected_context=(original[i,:,:64].astype(np.float32)*vc[:,None]).sum(axis=(0,2,3))/np.maximum(vc.sum(),1)
    context_per_frame[ident]=expected_context
    offsets=((np.arange(9)+.5)/9-.5)*L;dx,dy=np.meshgrid(offsets,offsets)
    for k,q in enumerate(u['initial_points'][:8]):
        if not original_labels['valid'][i,k]:continue
        cached=by.pop((ident,k));qx,qy=q;xx=qx+dx;yy=qy+dy
        ix=np.floor(xx+.5).astype(int);iy=np.floor(yy+.5).astype(int)
        nearest=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        continuous=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        valid=nearest&continuous
        dense_invalid_repaired[split]+=int((nearest&~continuous).sum())
        dense=vis[np.clip(iy,0,h-1),np.clip(ix,0,w-1)].astype(np.uint8)
        assert np.array_equal(cached['dense_valid'],valid)
        assert np.array_equal(cached['dense_y'],dense)
        assert np.array_equal(cached['x'],original[i,k])
        gap=float(np.max(np.abs(cached['context']-expected_context)));context_max=max(context_max,gap)
        assert np.allclose(cached['context'],expected_context,atol=2e-6,rtol=1e-6),(ident,gap)
        assert str(cached['image_sha256'])==u['image_sha256']
        px,py=np.floor(np.array(q)+.5).astype(int)
        m0,m4=bool(full[py,px]),bool(vis[py,px])
        assert cached['y']==m4==bool(original_labels['y'][i,k])
        state='VISIBLE_TARGET' if m4 else 'EXTERNAL_OCCLUDED_SUPPORT' if m0 else 'OUTSIDE_TARGET_SUPPORT'
        stats[(split,state)]+=1
        local=vis[max(0,py-1):min(h,py+2),max(0,px-1):min(w,px+2)]
        rows.append(dict(id=ident,split=split,native_corner=k,query_x=qx,query_y=qy,pixel_x=int(px),pixel_y=int(py),source_state=state,exact_presence=int(m4),M0_member=int(m0),M4_member=int(m4),visible_boundary_uncertain=bool(local.size<9 or local.sum()not in(0,9)),semantic_self='UNKNOWN',semantic_direct='UNKNOWN',image_sha256=u['image_sha256']))
assert not by and len(rows)==7415
with (HERE/'SYNTH_SOURCE_QUERY_STATES.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(schema='SYNTHETIC_V2_CACHE_INDEPENDENT_RECONSTRUCTION_AUDIT_V1',private_evaluation_input_reads=0,new_model_forward_or_fit_or_PnP=0,
            cache_bindings=bindings,query_count=len(rows),original_frozen_P3_features_exactly_preserved=True,all_scalar_M4_labels_exactly_preserved=True,
            dense_grid_M4_labels_exact_reconstruction=True,dense_valid_continuous_and_nearest_exact_reconstruction=True,
            corrected_dense_cells_by_split=dict(dense_invalid_repaired),context64_input_only_mean_reconstructed=True,context_max_abs_difference=context_max,
            evaluation_image_sha_overlap=0,source_train_val_image_sha_overlap=0,
            source_state_counts=[dict(split=s,state=c,N=n)for(s,c),n in sorted(stats.items())],
            source_state_definition='VISIBLE=M4=1; EXTERNAL_SUPPORT=M0=1,M4=0; OUTSIDE=M0=0,M4=0; semantic self and DIRECT unavailable',
            source_states_csv=dict(path=str(HERE/'SYNTH_SOURCE_QUERY_STATES.csv'),sha256=sha(HERE/'SYNTH_SOURCE_QUERY_STATES.csv')),
            context_caution='Context combines all8initialqueryROIs, including geometrically hidden ones. It is input-only context, not an instance-mask oracle and not a semantic self label.')
(HERE/'SYNTH_V2_CACHE_RECONSTRUCTION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
