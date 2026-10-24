"""One frozen V99 photometric corpus. No learned model, feature or private reader.

RGBA is separate from ownership. Cue-guided foreign placement is input to this
source generator, never an input to a future learned inference model.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
os.environ['__GL_SHADER_DISK_CACHE']='0'
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,shutil,time
import numpy as np
from render_math import STAGES,array_sha,affine_vertices,gl_pose_from_cv,gl_intrinsics_from_integer_camera,check_complete_clip,composite_premultiplied_rgba

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_DATA_DESIGN_BEFORE_CORPUS_RGB.json'
SCOPE_SHA='4fbcff4606e0f2d437b8111d9081f50ef797e1503146928b273192a237579998'
GEOM=P/'query_visibility_synthetic_part_teacher_20261023_v93/geometry_inventory/SCENE3_GEOMETRY_ASSET_LOCAL_ONLY.npz'
INV=GEOM.parent/'GEOMETRY_INVENTORY.json'
OLD=P/'query_visibility_query_anchored_cue_20261023_v96/synthetic_appearance_feasibility/render_one_appearance.py'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
PROTOCOL=H/'RENDER_PROTOCOL_BEFORE_CORPUS_RGB.json';OUT=H/'actual_corpus'
GENERATED=H/'GENERATED_RENDER_TESTS_FINAL.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rr):
    with Path(p).open('x')as f:
        for r in rr:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def crop_operator():
    assert sha(CROP)=='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
    fn=next(n for n in ast.parse(CROP.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop')
    ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(CROP),'exec'),ns)
    return ns['crop'],hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()

def make_protocol(jobs,geometry_seal,query_inputs,gt_targets):
    """Metadata/code/hash only; no source geometry array or source pixel decode."""
    assert sha(SCOPE)==SCOPE_SHA and not OUT.exists()
    cfgseal=V/'BASE96_CONFIG_SEAL.json';cs=read(cfgseal);config=V/'BASE96_SOURCE_CONFIG_INPUT_ONLY.jsonl'
    rr=rows(jobs);cc=rows(config);qq=rows(query_inputs);assert len(rr)==len(cc)==len(qq)==96
    for i,(r,c,q)in enumerate(zip(rr,cc,qq)):
        assert r['base_index']==c['base_index']==q['base_index']==i and r['base_id']==c['base_id']==q['base_id']
        for k in('split','appearance_family','appearance_seed','foreign_material_seed'):
            assert r[k]==c[k],(i,k)
        assert r['variants']==list(STAGES)and np.shape(r['query8_xy'])==(8,2)and np.shape(r['L'])==(8,)
        assert np.isfinite(r['query8_xy']).all()and np.all(np.asarray(r['L'])>0)
        assert r['camera']['size']==640 and r['camera']['near']>0 and r['camera']['far']>r['camera']['near']
        assert r['GT_only_targets']==bind(gt_targets)
    bgseal=V/'background_catalog/CATALOG_SEAL.json';bg=read(bgseal)
    assert bg['completed']and bg['PASS']and read(GENERATED)['PASS']
    assert read(GENERATED)['producer']==bind(__file__)
    for k in('background_math','assignments','selected96_validation'):check(bg[k])
    _,astsha=crop_operator()
    geomseal=read(geometry_seal);assert geomseal['completed']and geomseal.get('PASS',True)
    bgjoinseal=V/'background_catalog/ROOT_CONFIG_JOIN_SEAL.json';assert read(bgjoinseal)['completed']
    bgjoined=V/'background_catalog/BACKGROUND_ASSIGNMENTS96_WITH_ROOT_CONFIG_INPUT_ONLY.jsonl'
    return dict(completed=True,source_only=True,code=bind(__file__),math=bind(H/'render_math.py'),generated=bind(GENERATED),
        scope=bind(SCOPE),base_config_seal=bind(cfgseal),base_config=bind(config),geometry_preparation_seal=bind(geometry_seal),
        jobs=bind(jobs),query_inputs=bind(query_inputs),GT_only_targets=bind(gt_targets),
        background_seal=bind(bgseal),background_root_config_join_seal=bind(bgjoinseal),background_root_config_assignments=bind(bgjoined),background_assignments=bg['assignments'],background_validation=bg['selected96_validation'],background_math=bg['background_math'],
        geometry=bind(GEOM),inventory=bind(INV),wood_UV_helper=bind(OLD),crop_code=bind(CROP),crop_AST_sha256=astsha,
        cpu_raster=bind(P/'query_visibility_synthetic_part_teacher_20261023_v93/raster_preparation/cpu_part_raster.py'),
        owner_math=bind(P/'query_visibility_expected_cue_triplets_20261023_v97/render_preparation/matched_render_math.py'),
        base_count=96,full_RGB_count=288,query_count=2304,stages=list(STAGES),
        appearance=dict(target='Same V96 V3 procedural wood UV law, or fixed muted uniform polymer tint; exact root config seed. Float-white factors, metal0,roughness.88/.94,emission0. No part-ID palette.',
            target_batching='Face-expanded positions/normals/UV from frozen helper, concatenate primitives by original doubleSided. Original world triangles unchanged; new appearance baseline, no bitexact V97 reuse claim.',
            foreign='One procedural matte cardboard texture per root foreign_material_seed; same cuboid and material in COVER/SHAM, translation only.',
            lighting='V97 warm directional1/.95/.86 intensity3.2 at camera and ambient.22/.24/.27. No floor. Shadows included and may differ acrossvariants.',
            RGBA='Transparent black0/0/0/0, original opaque materials, 4xMSAA resolved RGB coverage-premultiplied. Generated EGL probe validates. Composite RGBpremult+(1-alpha)*background, float64, floor(clip+.5) uint8. Display-code-space approximation, not radiometrically linear.',
            background='Pinned Pillow12.2.0 decoded/crop SHA replay, same source frame all3 variants; real objects in background may create ambiguity.'),
        camera_policy='Exact B precomputed camera K/R/t/eye/near/far, CV integer centres0..639/c319.5; GL pose R.T@diag(1,-1,-1), GL physicalpixel principal=Kprincipal+.5 (320), equivalent V97Perspective50degrees. Generated100point projection replay. Complete target+foreign clip bounds asserted before eachbase.',
        query_policy='Exact frozen B q8/L all3 variants, old native RGB96 bilinear+valid crop; no q search, no relocalization, no adaptive reroll.',
        output_policy='288fullRGB PNG; 2304 RGB96uint8+valid arrays in12chunks of8base; image/query input-only JSONL. B sparseGT and ownership only in separateGTprovenance; fixed firstbase denseauxQA. First2base preview contacts saved immediately; all attempts retained.',
        storage_budget_bytes=350000000,output_directory=str(OUT),runtime='pallet-pose Python3.10.20 NumPy1.26.4 Pillow12.2.0 pyrender0.1.45 NVIDIA EGL; GL_DITHER disabled; no shader diskcache',
        native8_mapping='UNKNOWN',human_gold=False,training_admission=False,new_features_model_fit_private_current_PnP=0)

def authorize(entry):
    e=read(entry);assert e['root_authorized_one_corpus']is True and e['source_only']is True
    assert all(e[k]is False for k in('training_allowed','current_allowed','PnP_allowed','query_reselection_allowed'))
    for k in('code','protocol','generated','root_authorization'):check(e[k])
    assert e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and read(e['generated']['path'])['PASS']
    p=read(PROTOCOL);assert p['code']==e['code']and p['generated']==e['generated']and not OUT.exists()
    for k in('math','scope','base_config_seal','base_config','geometry_preparation_seal','jobs','query_inputs','GT_only_targets','background_seal','background_root_config_join_seal','background_root_config_assignments','background_assignments','background_validation','background_math','geometry','inventory','wood_UV_helper','crop_code','cpu_raster','owner_math'):check(p[k])
    assert p['training_admission']is False and p['native8_mapping']=='UNKNOWN'
    assert shutil.disk_usage(V).free>450000000,'Need enough space for strict350MB output budget'
    return e,p

def cardboard_texture(seed):
    rng=np.random.default_rng(int(seed));u,v=np.meshgrid(np.arange(256),np.arange(256))
    noise=rng.normal(0,1.6,(256,256));fiber=1.7*np.sin(v*.47)+.8*np.sin(u*.71)
    rgb=np.array([151.,125.,89.])+noise[...,None]+fiber[...,None]*np.array([1.,.9,.7])
    return np.floor(np.clip(rgb,0,255)+.5).astype(np.uint8)

def build_target_mesh(old,xyz,tri,pid,inventory,pyrender):
    """No ID colors: merge unchanged face-expanded geometry into two materials."""
    groups={False:[],True:[]}
    dummy=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=.88)
    for part in range(308):
        info=inventory['per_mesh'][part]
        m=old.textured_part(xyz,tri[pid==part],part,info['USD_orientation'],dummy)
        for prim in m.primitives:
            assert prim.indices is None and prim.color_0 is None
            groups[bool(info['double_sided'])].append(prim)
    pp=[]
    for ds,prims in groups.items():
        if not prims:continue
        mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=.88,emissiveFactor=[0.,0.,0.],alphaMode='OPAQUE',doubleSided=ds)
        pp.append(pyrender.Primitive(positions=np.concatenate([a.positions for a in prims]),normals=np.concatenate([a.normals for a in prims]),texcoord_0=np.concatenate([a.texcoord_0 for a in prims]),material=mat))
    assert sum(len(a.positions)//3 for a in pp)==len(tri)
    return pyrender.Mesh(pp)

def set_appearance(mesh,old,pyrender,family,seed):
    if family=='PROCEDURAL_WOOD':
        old.SEED=int(seed);tex=old.wood_texture();factor=[1.,1.,1.,1.];rough=.88
        texture=pyrender.Texture(source=tex,source_channels='RGB')
    elif family=='MATTE_POLYMER':
        rng=np.random.default_rng(int(seed));tint=np.array([.43,.48,.42])+rng.uniform(-.10,.10,3)
        factor=[*tint.tolist(),1.];rough=.94;texture=None;tex=None
    else:raise ValueError(family)
    for prim in mesh.primitives:
        ds=prim.material.doubleSided
        prim.material=pyrender.MetallicRoughnessMaterial(baseColorTexture=texture,baseColorFactor=factor,metallicFactor=0.,roughnessFactor=rough,emissiveFactor=[0.,0.,0.],alphaMode='OPAQUE',doubleSided=ds)
    return dict(family=family,seed=int(seed),texture_RGB_sha256=None if tex is None else array_sha(tex),float_baseColorFactor=factor,roughness=rough)

def external_mesh(pyrender,trimesh,g,seed):
    tex=cardboard_texture(seed)
    m=trimesh.Trimesh(vertices=np.array(g['cover_vertices_world']),faces=np.array(g['triangles']),process=False)
    m.visual=trimesh.visual.texture.TextureVisuals(uv=np.stack((np.arange(8)%2,(np.arange(8)//2)%2),axis=1).astype(float))
    mat=pyrender.MetallicRoughnessMaterial(baseColorTexture=pyrender.Texture(source=tex,source_channels='RGB'),baseColorFactor=[1.,1.,1.,1.],metallicFactor=0.,roughnessFactor=.92,emissiveFactor=[0.,0.,0.],alphaMode='OPAQUE',doubleSided=False)
    return pyrender.Mesh.from_trimesh(m,material=mat,smooth=False),array_sha(tex)

def dense_first_base(p,xyz,tri,pid,inventory,camera,g,rgba,RGBdepth):
    """Only first fixedbase; categorical IDs from geometry, never RGB decoding."""
    cpu=module(p['cpu_raster']['path'],'v99_first_dense_cpu');mm=module(p['owner_math']['path'],'v99_first_owner_math')
    camera={k:np.asarray(v,np.float64)if k in('K','R','t','eye_world','target_world','world_up','direction')else v for k,v in camera.items()}
    size=camera['size'];best=np.full((size,size),np.inf);second=best.copy();owner=np.full(best.shape,-1,np.int16)
    vcam=cpu.world_to_camera(xyz,camera)
    for part in range(308):
        take=pid==part;info=inventory['per_mesh'][part];ori=-1 if info['USD_orientation']=='leftHanded'else 1
        dd,ff,_=cpu.rasterize_parts(vcam,tri[take],np.zeros(int(take.sum()),np.int32),np.flatnonzero(take).astype(np.int32),np.full(int(take.sum()),info['double_sided'],bool),np.full(int(take.sum()),ori,np.int8),camera,1)
        mm.update_nearest_two(best,second,owner,dd[0],part)
    digest=array_sha(best)+array_sha(second)+array_sha(owner);ids=[];amb=[];depth=[]
    for stage in STAGES:
        a,b,o=best.copy(),second.copy(),owner.copy()
        if stage!='BEFORE'and g is not None:
            verts=np.asarray(g['cover_vertices_world']if stage=='COVER'else g['sham_vertices_world'])
            tt=np.asarray(g['triangles']);d,_,_=cpu.rasterize_parts(cpu.world_to_camera(verts,camera),tt,np.zeros(12,np.int32),np.arange(12,dtype=np.int32),np.zeros(12,bool),np.ones(12,np.int8),camera,1)
            mm.update_nearest_two(a,b,o,d[0],308)
        ii,aa=mm.final_ownership(a,b,o);ids.append(ii);amb.append(aa);depth.append(a)
    assert digest==array_sha(best)+array_sha(second)+array_sha(owner)
    np.savez_compressed(OUT/'FIRST_BASE_DENSE_AUX_GEOMETRY_ONLY.npz',part_id=np.array(ids,np.int16),ambiguity=np.array(amb,bool),depth=np.array(depth,np.float64),RGB_depth=np.array(RGBdepth,np.float32),photometric_RGBA=np.array(rgba,np.uint8),camera_K=np.array(camera['K']),camera_R=np.array(camera['R']),camera_t=np.array(camera['t']))

def contact(base,images,crops,states):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    out=Image.new('RGB',(1440,780),'#eef2f6');d=ImageDraw.Draw(out)
    d.text((12,8),base['base_id']+' | '+base['split']+' | '+base['appearance_family']+' | fixed q/L; no ID palette',font=font,fill='#14283c')
    k=base['selected_aux_index']
    for j,stage in enumerate(STAGES):
        d.text((j*480+12,37),stage+(' | selected AUX'+str(k)+' '+states[stage][k]['state']if k is not None else ' | no direct target, retained U'),font=font,fill='#14283c')
        out.paste(Image.fromarray(images[j]).resize((456,456),Image.Resampling.LANCZOS),(j*480+12,65))
        if k is not None:out.paste(Image.fromarray(crops[j,k].transpose(1,2,0)).resize((192,192),Image.Resampling.NEAREST),(j*480+12,545))
        d.text((j*480+216,555),'Unmarked RGB96\n2x view only\nSource-emulated AUX\nNative8 UNKNOWN',font=small,fill='#14283c',spacing=8)
    d.text((12,750),'Geometry-guided synthetic cue proxy; background is public RGB. No learned model, features, fit, current or PnP.',font=small,fill='#14283c')
    path=OUT/f'FIRST_FIXED_BASE_{base["base_index"]:03d}_CONTACT.png';out.save(path);return bind(path)

def run(entry):
    e,p=authorize(entry) # All actual source arrays/pixels and graphics below gate.
    import pyrender,trimesh,PIL
    from PIL import Image
    from OpenGL import GL
    assert PIL.__version__=='12.2.0'and pyrender.__version__=='0.1.45'and np.__version__=='1.26.4'
    begin=time.monotonic();OUT.mkdir();(OUT/'full_rgb').mkdir();(OUT/'rgb96_chunks').mkdir()
    write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),source_only=True,expected_passes=288))
    jobs=rows(p['jobs']['path']);bgrows=rows(p['background_assignments']['path']);bgvalid=rows(p['background_validation']['path']);qrows=rows(p['query_inputs']['path']);gt=rows(p['GT_only_targets']['path'])
    old=module(p['wood_UV_helper']['path'],'v99_immutable_wood_uv');bgmath=module(p['background_math']['path'],'v99_fixed_background_math')
    inv=read(p['inventory']['path'])
    with np.load(p['geometry']['path'],allow_pickle=False)as z:original_xyz=z['vertices_world_m'];tri=z['triangles_global'];pid=z['triangle_asset_local_part_id']
    assert len(tri)==110478 and len(original_xyz)==172656
    mesh_by_aspect={};xyz_by_aspect={}
    for aspect in (0,1):
        job=next(j for j in jobs if j['aspect_index']==aspect);xyz=affine_vertices(original_xyz,job['target_affine_world_row'])
        xyz_by_aspect[aspect]=xyz;mesh_by_aspect[aspect]=build_target_mesh(old,xyz,tri,pid,inv,pyrender)
    class ExactRGBRenderer(pyrender.OffscreenRenderer):
        def render(self,*a,**kw):
            self._platform.make_current();GL.glDisable(GL.GL_DITHER);return super().render(*a,**kw)
    renderer=ExactRGBRenderer(640,640);image_rows=[];query_rows=[];facts=[];chunks=[];chunk_c=[];chunk_v=[];preview=[];count_states={s:{}for s in STAGES}
    for i,(job,bg,bgv,qr,gtr)in enumerate(zip(jobs,bgrows,bgvalid,qrows,gt)):
        assert job['base_index']==bg['base_index']==bgv['base_index']==qr['base_index']==gtr['base_index']==i
        assert job['base_id']==qr['base_id']==gtr['base_id']and job['split']==bg['split']==gtr['split']
        assert qr['query8_xy']==job['query8_xy']and qr['L']==job['L']
        cam=job['camera'];g=job['foreign_geometry'];xyz=xyz_by_aspect[job['aspect_index']]
        for x in jobs:
            if x['aspect_index']==job['aspect_index']:assert x['target_affine_world_row']==job['target_affine_world_row']
        clipping=check_complete_clip(cam,xyz,None if g is None else g['cover_vertices_world'],None if g is None else g['sham_vertices_world'])
        imagepath=check(bg['image']);raw=np.array(Image.open(imagepath).convert('RGB'),copy=True)
        background,bgf=bgmath.prepare_background(raw,(640,640));assert bgf['decoded_native_RGB_sha256']==bgv['decoded_native_RGB_sha256']and bgf['canvas_RGB_sha256']==bgv['canvas_RGB_sha256']
        mesh=mesh_by_aspect[job['aspect_index']];matfacts=set_appearance(mesh,old,pyrender,job['appearance_family'],job['appearance_seed'])
        scene=pyrender.Scene(ambient_light=[.22,.24,.27],bg_color=[0.,0.,0.,0.]);scene.add(mesh)
        pose=gl_pose_from_cv(cam);scene.add(pyrender.IntrinsicsCamera(**gl_intrinsics_from_integer_camera(cam)),pose=pose)
        scene.add(pyrender.DirectionalLight(color=[1.,.95,.86],intensity=3.2),pose=pose)
        foreignnode=None;foreigntexsha=None;images=[];cropped=[];valids=[];rawrgba=[];depths=[]
        crop,astsha=crop_operator();assert astsha==p['crop_AST_sha256'];q=np.asarray(job['query8_xy'],np.float64);L=np.asarray(job['L'],np.float64)
        for si,stage in enumerate(STAGES):
            if stage=='COVER'and g is not None:
                fm,foreigntexsha=external_mesh(pyrender,trimesh,g,job['foreign_material_seed']);foreignnode=scene.add(fm)
            if stage=='SHAM'and foreignnode is not None:
                T=np.eye(4);T[:3,3]=g['sham_world_translation'];scene.set_pose(foreignnode,T)
                assert np.allclose(np.asarray(g['cover_vertices_world'])+T[:3,3],g['sham_vertices_world'],atol=2e-12,rtol=0)
            rgba,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            assert rgba.shape==(640,640,4)and rgba.dtype==np.uint8 and np.isfinite(depth).all()
            assert np.all(rgba[rgba[...,3]==0,:3]==0),'Transparent background must stay black premultiplied'
            rgb=composite_premultiplied_rgba(rgba,background);images.append(rgb)
            image_id=f'{job["base_id"]}_{stage}';path=OUT/'full_rgb'/f'{image_id}.png';Image.fromarray(rgb).save(path)
            cr=[];va=[]
            for k in range(8):a,b=crop(rgb,q[k],L[k],96);cr.append(a);va.append(b)
            cropped.append(cr);valids.append(va)
            image_rows.append(dict(image_index=3*i+si,image_id=image_id,base_index=i,base_id=job['base_id'],variant=stage,split=job['split'],image=bind(path),decoded_RGB_sha256=array_sha(rgb),raw_hw=[640,640],background_id=bg['background_id'],query_inputs=p['query_inputs'],query_input_row=i))
            if i==0:rawrgba.append(rgba);depths.append(depth)
            for st in gtr['stages'][stage]:count_states[stage][st['state']]=count_states[stage].get(st['state'],0)+1
        cropped=np.array(cropped,np.uint8);valids=np.array(valids,bool);assert cropped.shape==(3,8,3,96,96)and valids.shape==(3,8,1,96,96)
        chunk_c.append(cropped);chunk_v.append(valids)
        for si,stage in enumerate(STAGES):
            for k in range(8):query_rows.append(dict(occurrence_index=i*24+si*8+k,base_index=i,base_id=job['base_id'],image_index=3*i+si,image_id=image_rows[3*i+si]['image_id'],variant=stage,split=job['split'],aux_corner_index=k,query_xy=job['query8_xy'][k],L=job['L'][k],source_emulated_not_initializer=True,native8_mapping='UNKNOWN',rgb96_sha256=array_sha(cropped[si,k]),crop_valid96_sha256=array_sha(valids[si,k]),chunk_index=i//8,index_in_chunk_base=i%8,stage_index=si))
        facts.append(dict(base_index=i,base_id=job['base_id'],camera=cam,target_affine_world_row=job['target_affine_world_row'],clip=clipping,target_material=matfacts,foreign_material_seed=job['foreign_material_seed'],foreign_texture_RGB_sha256=foreigntexsha,background=bgf,all3_qL_exact=True,rgba_partial_alpha_counts=[int(np.sum((a[...,3]>0)&(a[...,3]<255)))for a in rawrgba]if i==0 else None))
        if i==0:dense_first_base(p,xyz,tri,pid,inv,cam,g,rawrgba,depths)
        if i<2:
            preview.append(contact(job,images,cropped,gtr['stages']));write(OUT/f'FIRST_FIXED_BASE_{i:03d}_PREVIEW_READY.json',dict(completed=True,contact=preview[-1],base_index=i,raw_RGB=[image_rows[3*i+s]['image']for s in range(3)],already_rendered=True,new_model_calls=0))
            print('PREVIEW_READY '+json.dumps(preview[-1]),flush=True)
        if (i+1)%8==0:
            lo=i-7;path=OUT/'rgb96_chunks'/f'BASES_{lo:03d}_{i:03d}.npz'
            np.savez_compressed(path,rgb96=np.array(chunk_c,np.uint8),crop_valid96=np.array(chunk_v,bool),query8_xy=np.array([j['query8_xy']for j in jobs[lo:i+1]],np.float64),L=np.array([j['L']for j in jobs[lo:i+1]],np.float64),base_index=np.arange(lo,i+1,dtype=np.int32),stage=np.array(STAGES))
            chunks.append(dict(chunk_index=len(chunks),start_base=lo,end_base=i+1,base_count=8,query_count=192,file=bind(path)));chunk_c.clear();chunk_v.clear()
        used=sum(f.stat().st_size for f in V.rglob('*')if f.is_file());assert used<350000000,used
        print(f'BASE_COMPLETE {i+1}/96 passes={(i+1)*3} elapsed={time.monotonic()-begin:.3f}s V99bytes={used}',flush=True)
    renderer.delete();assert len(image_rows)==288 and len(query_rows)==2304 and len(chunks)==12
    write_rows(OUT/'FULL_RGB288_INPUT_ONLY.jsonl',image_rows);write_rows(OUT/'QUERY_OCCURRENCES2304_INPUT_ONLY.jsonl',query_rows);write_rows(OUT/'RENDER_FACTS96_PROVENANCE.jsonl',facts)
    write(OUT/'GT_PROVENANCE_ONLY.json',dict(completed=True,GT_only_targets=p['GT_only_targets'],geometry_preparation_seal=p['geometry_preparation_seal'],jobs=p['jobs'],native8_mapping='UNKNOWN',human_gold=False,source_generator_uses_cue_guided_foreign_placement=True,model_input_GT_fields=False))
    write(OUT/'PACK_MANIFEST.json',dict(completed=True,base_count=96,frame_count=288,query_count=2304,query_inputs=p['query_inputs'],full_RGB=bind(OUT/'FULL_RGB288_INPUT_ONLY.jsonl'),input_occurrences=bind(OUT/'QUERY_OCCURRENCES2304_INPUT_ONLY.jsonl'),chunks=chunks,array_schema=dict(rgb96='uint8[8base,3stage,8aux,3,96,96]',crop_valid96='bool[8base,3stage,8aux,1,96,96]',query8_xy='float64[8base,8,2]',L='float64[8base,8]'),native8_mapping='UNKNOWN',human_gold=False,training_admission=False,new_features_model_current_fit_PnP=0))
    result=dict(completed=True,PASS_execution=True,entry=bind(entry),protocol=bind(PROTOCOL),base_count=96,actual_photometric_RGB_passes=288,logical_query_occurrences=2304,all_attempts_preserved=True,split_base_counts={s:sum(j['split']==s for j in jobs)for s in('TRAIN','VALIDATION')},cue_state_counts_GT_only=count_states,GT_unresolved_bases=sum(j['after_force_U']for j in jobs),qL_reselection=0,first2_previews=preview,elapsed_seconds=time.monotonic()-begin,native8_mapping='UNKNOWN',human_gold=False,training_admission=False,limits=['Same source pallet geometry across folds; no unseen instance or sim2real evidence.','Synthetic finite endpoint face/edge witness proxy, not human/native8 semantic gold.','Photometric procedural materials and public clutter backgrounds are approximations; floating foreign cuboid and varying shadows may be shortcuts.','RGB compositor blends display-code values; alpha edge antialiasing is separate from sparse geometric witness truth.','No features, head, optimizer, current223 or PnP executed.'])
    write(OUT/'RESULT.json',result);files=[bind(f)for f in sorted(OUT.rglob('*'))if f.is_file()]
    seal=dict(completed=True,PASS_execution=True,result=bind(OUT/'RESULT.json'),manifest=bind(OUT/'PACK_MANIFEST.json'),entry=bind(entry),protocol=bind(PROTOCOL),files=files,storage_bytes_before_seal=sum(f.stat().st_size for f in V.rglob('*')if f.is_file()),new_model_features_fit_current_PnP=0)
    assert seal['storage_bytes_before_seal']<349000000;write(OUT/'RENDER_SEAL.json',seal)
    print('COMPLETE '+json.dumps(dict(seal=bind(OUT/'RENDER_SEAL.json'),elapsed_seconds=result['elapsed_seconds'],state_counts=count_states)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-jobs',type=Path);ap.add_argument('--geometry-seal',type=Path);ap.add_argument('--query-inputs',type=Path);ap.add_argument('--GT-targets',type=Path);ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.freeze_jobs:write(PROTOCOL,make_protocol(a.freeze_jobs,a.geometry_seal,a.query_inputs,a.GT_targets));print(json.dumps(bind(PROTOCOL)))
    elif a.entry:run(a.entry)
    else:raise SystemExit('Choose metadata-only --freeze-jobs + bindings, or --entry')
