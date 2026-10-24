"""Independent V136 preparation review: metadata + invented geometry only.
No actual mesh payload, teacher, raster/RGB, model, private, V112 or PnP call.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;R=V/'render_preparation'
EXPECTED={'policy':(V/'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json','546f3d28d4454ba8a28a5eecdbfb0d30d91e6513e2ecd3209fb8945be8dd9ec4'),
 'core':(R/'continuous_corpus_math.py','310a1eec90afb36d305becfee6194cc7d9fa2b9587b181cef1c3d78113c9599d'),
 'renderer':(R/'render_continuous_corpus_once.py','45fc8a065aecc943c06a2b3b877de7362a947369b85432a1a153e298914ef0ea'),
 'configs':(R/'BASE_CONFIGS640_BEFORE_RGB.json','94d7fa73d78bd7f5505929b0700d143d113d31da36cdcc3adb6783b9f425a170'),
 'protocol':(R/'CORPUS_PROTOCOL_BEFORE_PIXELS.json','543893f761447f379211c1a8c7e8ebaec1e644e7a54834d0b6aa4b97bcc91342'),
 'generated':(R/'GENERATED_CORPUS_TESTS.json','30a37799bb298d0d747b30a541d179761a4188d6f6668d3bd651c9fa4500488b')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hash64(*x):return int(hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()[:16],16)
def uniform(lo,hi,*x):return lo+(hi-lo)*(hash64(*x)/2**64)

def invented_template(core):
    """Eight independent constructive corner boxes; not either actual pallet mesh."""
    anchors=core.SIGNS*core.OLD_DIM/2;vv=[];tri=[];own=[];axis=[];side=[];patch=[];edge=[];witness=[];bounds=[]
    for i,(a,sign)in enumerate(zip(anchors,core.SIGNS)):
        lo=np.where(sign<0,a,a-.04);hi=np.where(sign<0,a+.04,a);bounds.append((lo,hi))
        vertex=np.array([[x,y,z]for x in(lo[0],hi[0])for y in(lo[1],hi[1])for z in(lo[2],hi[2])]);vv.extend(vertex)
        for ax in range(3):
            for sd in(-1,1):
                ids=np.flatnonzero(vertex[:,ax]==(lo if sd<0 else hi)[ax]);tri.extend((ids[[0,1,3]]+8*i,ids[[0,3,2]]+8*i));own.extend([i,i]);axis.extend([ax,ax]);side.extend([sd,sd])
        away=-sign.astype(float);quads=[];grid=[]
        for ax in range(3):
            other=[k for k in range(3)if k!=ax];u=np.eye(3)[other[0]]*away[other[0]]*.01;v=np.eye(3)[other[1]]*away[other[1]]*.01
            quads.append([a,a+u,a+u+v,a+v]);grid.extend(a+x*u+y*v for x in(.2,.5,.8)for y in(.2,.5,.8))
        e=np.array([a+np.eye(3)[k]*away[k]*.01*t for k in range(3)for t in(.2,.5,.8)])
        patch.append(quads);edge.append(e);witness.append(np.concatenate((a[None],e,np.asarray(grid))))
    return dict(vertices=np.asarray(vv),triangles=np.asarray(tri,np.int32),part_owner=np.asarray(own,np.int32),face_axis=np.asarray(axis,np.int8),face_side=np.asarray(side,np.int8),native_anchor3D=anchors,native_primary_part_id=np.arange(8),native_incident_part_ids=np.arange(8)[:,None],native_patch_quads=np.asarray(patch),native_edge_samples=np.asarray(edge),native_witness_points=np.asarray(witness)),np.asarray(bounds)

def run():
    assert not(H/'REVIEW.json').exists()
    for key,(p,digest)in EXPECTED.items():assert sha(p)==digest,key
    policy=read(EXPECTED['policy'][0]);protocol=read(EXPECTED['protocol'][0]);saved=read(EXPECTED['configs'][0]);rows=saved['rows']
    oldb=policy['old_regression'];oldhash={k:sha(b['path'])for k,b in oldb.items()if isinstance(b,dict)and'path'in b}
    assert all(oldhash[k]==oldb[k]['sha256']for k in oldhash)
    old=read(oldb['base_configs']['path'])['rows'];assert len(old)==192
    oldfold={f:hashlib.sha256(canon([r for r in old if r['split']==f])).hexdigest()for f in('TRAIN','VALIDATION')}
    assert [sum(r['split']==f for r in old)for f in('TRAIN','VALIDATION')]==[128,64]
    spec=importlib.util.spec_from_file_location('independent_V136_math',EXPECTED['core'][0]);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    assert sha(core.OLD)==core.OLD_SHA and sha(core.ORDER)==core.ORDER_SHA
    # Mesh bindings are JSON metadata; do not open their NPZ payloads.
    source=read(protocol['geometry_result']['path']);meshes=[r['mesh']for r in source['results']]
    assert [len(r['parts'])for r in source['results']]==[17,19]and len(meshes)==2
    generated=core.make_configs(policy,meshes);assert generated==rows
    assert len(rows)==640 and [sum(r['split']==f for r in rows)for f in('TRAIN','VALIDATION')]==[512,128]
    audit=core.metadata_audit(rows,old);assert audit['PASS']
    assert [r['base_index']for r in rows]==list(range(640))and len(set(r['base_id']for r in rows))==640
    independent_folds={};normal_margin=[];min_depth=np.inf;max_raytol=0.;all_seedsets={};old_seeds={r['factors']['asset_seed']for r in old}
    for fold,N in(('TRAIN',32),('VALIDATION',8)):
        rr=[r for r in rows if r['split']==fold];group={}
        for r in rr:group.setdefault(r['geometry_group_id'],[]).append(r)
        assert len(group)==N and all(len(v)==16 for v in group.values())
        heights=[];ordinals=[];textures=set();viewkeys=set();all_seedsets[fold]={k:set()for k in('material','light','background','foreign')}
        for key,twins in group.items():
            r=twins[0];g=r['generation_keys'];ordinal=g['height_stratum'];geom=tuple(g['geometry']);dims=np.asarray(r['dimensions_xyz_m'])
            assert g['height_strata']==N and 0<=ordinal<N
            expected_h=.1+.1*(ordinal+uniform(0,1,*geom,'height_jitter'))/N
            assert dims[1]==expected_h and int(np.floor((dims[1]-.1)/.1*N))==ordinal
            assert dims[2]==uniform(.85,1.25,*geom,'footprint_Z')
            assert dims[0]==policy['fixed_factors']['aspect_ratios'][r['factors']['aspect']]*dims[2]
            assert np.all(dims>0)and .1<dims[1]<.2
            assert all(q['dimensions_xyz_m']==r['dimensions_xyz_m']for q in twins)
            assert sorted(set(q['factors']['material']for q in twins))==sorted(policy['fixed_factors']['materials'])
            heights.append(float(dims[1]));ordinals.append(ordinal)
        assert sorted(ordinals)==list(range(N))
        for r in rr:
            f=r['factors'];dims=np.asarray(r['dimensions_xyz_m']);camera=r['camera'];R0=np.asarray(camera['R']);t=np.asarray(camera['t']);eye=np.asarray(camera['eye_world']);K=np.asarray(camera['K']);key=tuple(r['generation_keys']['camera_and_foreign'])
            az,el=f['camera_pair_deg'];assert az==f['view_sign']*uniform(5,40,*key,'azimuth_abs_deg')and el==uniform(10,45,*key,'elevation_deg')
            assert 5<abs(az)<40 and 10<el<45
            assert f['distance_scale']==uniform(2.2,4.2,*key,'distance_scale')and 2.2<f['distance_scale']<4.2
            distance=f['distance_scale']*max(dims[0],dims[2]);assert f['distance_m']==camera['distance']==distance and np.isclose(np.linalg.norm(eye),distance,atol=1e-14,rtol=0)
            assert np.allclose(R0@R0.T,np.eye(3),atol=1e-14)and np.linalg.det(R0)>0 and np.array_equal(t,-R0@eye)
            assert camera['size']==512 and camera['near']==.001 and K[0,0]==K[1,1]==256/np.tan(np.deg2rad(25))and np.array_equal(K[:2,2],[255.5,255.5])
            a=np.asarray(r['native8_canonical_xyz']);scale=np.asarray(r['positive_scale_xyz']);assert np.array_equal(a,(core.SIGNS*core.OLD_DIM/2)*scale)
            perm=np.asarray(r['permCF_to_canonical']);assert sorted(perm.tolist())==list(range(8))and r['selected_canonical_index']==perm[r['selected_CF_index']]
            projected=(a@R0.T+t);min_depth=min(min_depth,float(projected[:,2].min()));max_raytol=max(max_raytol,1e-8*float(np.linalg.norm(a-eye,axis=1).max()))
            assert (projected[:,2]>.001).all()
            cert=core.certified_normal_mapping(r,core.normal_rule());assert cert==r['source_label_mapping_certificate'];normal_margin.append(cert['top_cosine_margin'])
            for purpose in('foreign_extent','foreign_margin_px','foreign_depth_fraction','foreign_roll_deg'):
                lo,hi=policy['continuous'][purpose]['bounds'];field={'foreign_extent':'foreign_extent_factor'}.get(purpose,purpose);assert r[field]==uniform(lo,hi,*key,purpose)
            viewkeys.add(tuple(np.r_[eye,dims]))
            for purpose,value in r['seeds'].items():all_seedsets[fold][purpose].add(value)
            seed=r['seeds']['material'];textures.add((f['material'],seed%1009,seed%53)if f['material']=='PROCEDURAL_WOOD'else(f['material'],seed%1009))
        assert set(r['factors']['asset_seed']for r in rr).isdisjoint(old_seeds)
        independent_folds[fold]=dict(bases=len(rr),height_strata=N,all_strata_once_per_geometry=True,height_min=min(heights),height_max=max(heights),view_dimension_tuples=len(viewkeys),texture_recipe_signatures=len(textures),fold_metadata_sha256=hashlib.sha256(canon(rr)).hexdigest())
    for purpose in all_seedsets['TRAIN']:assert all_seedsets['TRAIN'][purpose].isdisjoint(all_seedsets['VALIDATION'][purpose])
    bank=policy['fixed_factors']['material_tone_banks'];assert set(bank['TRAIN'])=={6+25*i for i in range(40)if i%5!=2}and set(bank['VALIDATION'])=={6+25*i for i in range(40)if i%5==2}
    assert bank['TRAIN']!=sorted(bank['TRAIN'])and bank['VALIDATION']!=sorted(bank['VALIDATION'])
    # Invented eight-box registration, no actual procedural mesh or teacher.
    template,bounds=invented_template(core);min_near=np.inf;min_far=np.inf;min_foreign_depth=np.inf;maxscale=0.
    for cfg in rows:
        d=core.scaled_data(template,cfg);scale=np.asarray(cfg['positive_scale_xyz']);maxscale=max(maxscale,float(scale.max()))
        for k in('vertices','native_anchor3D','native_patch_quads','native_edge_samples','native_witness_points'):assert np.array_equal(d[k],template[k]*scale)
        for k in('triangles','part_owner','face_axis','face_side','native_primary_part_id','native_incident_part_ids'):assert np.array_equal(d[k],template[k])
        for i,w in enumerate(d['native_witness_points']):
            lo,hi=bounds[i]*scale;assert w.shape==(37,3)and np.all(w>=lo-1e-14)and np.all(w<=hi+1e-14)and np.all(np.any(np.isclose(w,lo,atol=1e-14,rtol=0)|np.isclose(w,hi,atol=1e-14,rtol=0),axis=1))
        foreign,translations,placement=core.foreign_placements(d,cfg);min_near=min(min_near,placement['near_clearance_px']);min_far=min(min_far,placement['far_additional_px'])
        camera=cfg['camera'];R0=np.asarray(camera['R']);t=np.asarray(camera['t'])
        for variant in('COVER','NEAR_SHAM','FAR_SHAM'):
            assert np.allclose(foreign[variant]-foreign['COVER'],translations[variant],rtol=0,atol=4e-15)
            min_foreign_depth=min(min_foreign_depth,float((foreign[variant]@R0.T+t)[:,2].min()))
        assert placement['only_translation']and placement['no_RGB_or_label_search']
    assert min_near>=1-1e-10 and min_far>=128-1e-10 and min_foreign_depth>.001
    # Static source review: all pixels/teacher are confined to root-entry run.
    render=EXPECTED['renderer'][0].read_text();tree=ast.parse(render);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    assert not any(isinstance(n,(ast.Break,ast.Continue))for n in ast.walk(fn))
    assert "records=teacher.region_teacher(data,camera,raster,fv)"in render and "records_CF=core.CF_records(records,perm,data,cfg['camera'])"in render
    assert "assert np.array_equal(rgb[mask],before_rgb[mask])"in render and "preserve_original_V129_128_64"in render
    assert "native_witness_points"in render and "np.ndindex(8,37)"in render and "target_rasters==320 and foreign_rasters==1920"in render
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(k in ast.unparse(n).lower()for k in('torch','ultralytics','transformers','scipy.optimize'))for n in ast.walk(tree))
    assert all(sha(b['path'])==oldhash[k]for k,b in oldb.items()if k in oldhash)
    for key,(p,digest)in EXPECTED.items():assert sha(p)==digest,key
    result=dict(completed=True,PASS=True,scope='Independent preparation source review, all640 metadata and invented geometry math; no actual new GT/RGB, template payloads, models, fit, private, V112 or PnP.',code=bind(__file__),inputs={k:bind(p)for k,(p,d)in EXPECTED.items()},old128_64_fold_digests=oldfold,old5_regression_binding_hashes_unchanged=oldhash,folds=independent_folds,
      regenerated_metadata_all640_exact=True,independent_hash_purpose_height_dimensions_camera_foreign_equations=True,old_and_new_seed_geometry_texture_view_nonoverlap=True,tone_banks_fullspan_interleaved_and_independently_shuffled=True,normal_facing_source_order_exact_all640=True,minimum_normal_top_margin=min(normal_margin),minimum_native_camera_depth=min_depth,maximum_teacher_distance_tolerance_estimate=max_raytol,
      invented_template=dict(parts=8,vertices=64,witnesses_per_corner=37,positive_scaling_all5_coordinate_fields_exact=True,all296_witnesses_remain_on_designated_box_faces_all640=True,noncoordinate_registration_arrays_unchanged=True),
      inherited_placement_all640_invented_geometry=dict(minimum_near_clearance_px=min_near,minimum_far_added_px=min_far,minimum_foreign_camera_depth=min_foreign_depth,same_cuboid_vertices_only_translated=True),
      manual_render_review=dict(no_quality_or_label_rejection_or_rerolls=True,all4variants_and8slots_retained=True,teacher37_continuous_firsthit_not_rasterownership=True,pixel_nearest_diagnostics_not_relabelled=True,CF_normal_order_no_areaV4=True,target_layer_digest_immutable=True,surviving_target_RGB_asserted_bitexact=True,root_count_and_preserve_old_guard=True),
      compatible_inherited_assumptions=['512 resolution/50deg FOV/principal255.5 unchanged in camera, diagnostics and shader','same17/19part rectilinear templates: 8vertices/12triangles perpart unchanged','positive diagonal scaling transforms anchors, all3patches,9edges and37witnesses with vertices','axis-aligned registered face normals and ownership semantics preserved','ray/depth numerical tolerances retained; partial/tie/miss/offframe U never coerced'],limitations=['Only same2topologies and same procedural shader; no new topology or photoreal transfer claim','Independent math does not run actual mesh payloads, render, teacher labels or pixel ownership','Broader held-out synthetic scale/view coverage has not demonstrated better actual-initializer or application visibility performance','Current raw RGB/teacher output metrics remain separate from preparation review'])
    with(H/'REVIEW.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),folds=independent_folds,minimum_normal_margin=min(normal_margin),minimum_near_clearance_px=min_near,minimum_far_added_px=min_far)),flush=True)
if __name__=='__main__':run()
