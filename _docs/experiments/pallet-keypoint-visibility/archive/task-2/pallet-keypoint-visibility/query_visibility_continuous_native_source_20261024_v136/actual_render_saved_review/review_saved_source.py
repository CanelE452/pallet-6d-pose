"""Independent saved V136 corpus audit and deterministic scientific previews.

Only completed saved bases may be previewed. Full audit requires completed
RENDER_COMPLETE. No renderer/teacher execution, model, fitting, private data,
V112, PnP, query inference, correction or source-file writes occur here.
"""
from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SOURCE=V/'actual_corpus_v1';COMPLETE=SOURCE/'RENDER_COMPLETE.json'
CONFIGS=V/'render_preparation/BASE_CONFIGS640_BEFORE_RGB.json'
PROTOCOL=V/'render_preparation/CORPUS_PROTOCOL_BEFORE_PIXELS.json'
POLICY=V/'ROOT_CONTINUOUS_CORPUS_POLICY_BEFORE_PIXELS.json'
CORE=V/'render_preparation/continuous_corpus_math.py'
ENTRY_SHA='863945bb79433b4ee092260c45034c5ce8268581cee424a7004a50894b9a3dba'
PREVIEW_BASES=(0,21,42,63)
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
ARRAY_KEYS=('vertices','triangles','part_owner','face_axis','face_side','native_anchor3D','native_primary_part_id','native_incident_part_ids','native_patch_quads','native_edge_samples','native_witness_points')
checked={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']).resolve()
    if str(p)not in checked:checked[str(p)]=sha(p)
    assert checked[str(p)]==b['sha256'],str(p)
    return p
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def mod(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def arrays(path,keys):
    with np.load(path,allow_pickle=False)as f:return{k:f[k].copy()for k in keys}
def mask_box(mask):
    y,x=np.nonzero(mask)
    return None if not len(x)else[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]

def source_metadata():
    p=read(PROTOCOL);assert p['code']['sha256']=='45fc8a065aecc943c06a2b3b877de7362a947369b85432a1a153e298914ef0ea'
    for key in('code','core_code','generated','policy','base_configs','geometry_result','geometry_code','geometry_design','raster_code','reviewed_renderer_helper','immutable_math','source_label_order','metadata_audit'):check(p[key])
    assert read(p['generated']['path'])['PASS']and read(p['metadata_audit']['path'])['PASS']
    for key in('policy','base_configs','render_complete','source_math','renderer'):check(p['old_regression'][key])
    cfg=read(CONFIGS)['rows'];assert len(cfg)==640
    return p,cfg,mod(CORE,'v136_saved_review_pure_math')

def base_packet(b,configs):
    cfg=configs[b];base=SOURCE/'bases'/cfg['base_id'];cp=base/'BASE_COMPLETE.json'
    assert cp.exists(),'Base is incomplete; no partial files may be consumed.'
    r=read(cp);assert r['completed']and r['PASS']and r['base_index']==b and r['base_id']==cfg['base_id']and r['scenes']==4 and r['no_retry']
    assert r['config']['file']==bind(CONFIGS)and r['config']['row_index']==b
    assert r['target_layers_before']==r['target_layers_after']
    ip,gp=check(r['input_scenes']),check(r['GT_scenes']);check(r['geometry']);check(r['foreign_geometry'])
    inputs=read(ip);gts=read(gp);assert inputs['completed']and gts['completed']and len(inputs['rows'])==len(gts['rows'])==4
    return cfg,r,inputs['rows'],gts['rows']

def canonical_rule(record,xyz,witness,camera,core):
    """Recompute rule algebra from saved 37 ray classifications, not new rays."""
    i=record['native_canonical_index'];anchor=xyz[i]
    front=np.diag(core.SIGNS[i])@(np.asarray(camera['eye_world'])-anchor)>1e-12
    active=[0]
    for axis in range(3):
        if front[[a for a in range(3)if a!=axis]].any():active.extend(range(1+3*axis,4+3*axis))
        if front[axis]:active.extend(range(10+9*axis,19+9*axis))
    active=np.asarray(sorted(set(active)),np.int32)
    assert record['front_face_axes']==np.flatnonzero(front).tolist()and record['active_witness_indices']==active.tolist()
    base=np.asarray(record['target_witness_states']);combined=np.asarray(record['combined_witness_states'])
    assert base.shape==combined.shape==(37,)and np.isin(base,list('DESU')).all()and np.isin(combined,list('DESU')).all()
    for arr,key in((base,'target_counts'),(combined,'combined_counts')):assert record[key]=={s:int((arr==s).sum())for s in'DSEU'}
    projected=core.project(witness[i],camera);inframe=bool(((projected>=0)&(projected<=511)).all())
    assert record['all_region_witnesses_in_frame']==inframe
    assert np.array_equal(np.asarray(record['projected_reference_xy']),projected[0])
    geometric_s=bool(np.all(base=='S'))
    before='S'if geometric_s else'D'if front.sum()>=2 and np.all(base[active]=='D')else'U'
    state='U'if not inframe else'S'if before=='S'else'U'if before!='D'else'E'if np.all(combined[active]=='E')else'D'if np.all(combined[active]=='D')else'U'
    assert record['geometric_S']==geometric_s and record['target_alone_region_state']==before and record['state']==state
    reason='offframe'if not inframe else'target_alone_S'if before=='S'else'target_alone_U'if before!='D'else'active_all_E'if state=='E'else'active_all_D'if state=='D'else'partial_external_U'
    return state,reason

def preview_base(b,configs):
    """Raw PNGs unchanged; scientific overlays are separate output files."""
    from PIL import Image,ImageDraw,ImageFont
    cfg,checkpoint,inputs,gts=base_packet(b,configs)
    out=H/'previews';out.mkdir(exist_ok=True);path=out/(cfg['base_id']+'_GT_ANCHOR_DIAGNOSTIC.png')
    if path.exists():return bind(path)
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),17)if fontpath.exists()else ImageFont.load_default()
    small=ImageFont.truetype(str(fontpath),14)if fontpath.exists()else ImageFont.load_default()
    title=ImageFont.truetype(str(fontpath),21)if fontpath.exists()else ImageFont.load_default()
    panel=512;top=102;scene_y=top+32;zoom_y=scene_y+panel+38;canvas=Image.new('RGB',(4*panel,zoom_y+286),(245,246,248));d=ImageDraw.Draw(canvas)
    f=cfg['factors'];selected=cfg['selected_CF_index'];pair=f['camera_pair_deg']
    d.text((12,8),f"{cfg['base_id']} | {f['geometry_family']} | {f['aspect']} | {f['material']}",font=title,fill=(15,20,28))
    d.text((12,35),f"height={cfg['dimensions_xyz_m'][1]:.4f}m | az/el={pair[0]:.2f}/{pair[1]:.2f} deg | quarter={f['signed_quarter']} | foreign={f['foreign_family']}",font=font,fill=(15,20,28))
    d.text((12,60),'CYAN = known teacher GT anchor; ORANGE = selected GT anchor. No inferred query is shown.',font=font,fill=(15,20,28))
    d.text((12,82),'D direct / E external cover / S self hidden / U conservative unknown. Original raw RGB files are preserved.',font=small,fill=(35,40,48))
    records=[]
    targetmask=arrays(check(gts[0]['raw_buffers']),('part_id',))['part_id']>=0
    targetbox=mask_box(targetmask);targetwh=[targetbox[2]-targetbox[0],targetbox[3]-targetbox[1]]if targetbox else[0,0]
    for vi,(ir,gr)in enumerate(zip(inputs,gts)):
        assert ir['variant']==gr['variant']==VARIANTS[vi];png=check(ir['raw_RGB_PNG'])
        im=Image.open(png).convert('RGB');rgb=np.asarray(im);assert rgb.shape==(512,512,3)
        assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==ir['decoded_RGB_sha256']
        x0=vi*panel;canvas.paste(im,(x0,scene_y));d.text((x0+8,top+4),ir['variant']+' | '+' '.join(f'{k}{s}'for k,s in enumerate(gr['state8_CF'])),font=small,fill=(15,20,28))
        refs=np.asarray(gr['native8_CF_xy']);xy=refs[selected];cx,cy=int(np.floor(xy[0]+.5)),int(np.floor(xy[1]+.5));box=(cx-64,cy-64,cx+64,cy+64)
        # This is an explicitly teacher-centered diagnostic window, not a model crop.
        zoom=im.crop(box).resize((256,256),Image.Resampling.NEAREST);canvas.paste(zoom,(x0+128,zoom_y+24))
        d.text((x0+12,zoom_y+2),f'GT {selected},128px at 2x | target {targetwh[0]}x{targetwh[1]}px | foreign {gr["foreign_pixels"]}px',font=small,fill=(15,20,28))
        for k,(x,y)in enumerate(refs):
            color=(255,153,0)if k==selected else(0,235,255);px=x0+x;py=scene_y+y
            d.ellipse((px-5,py-5,px+5,py+5),outline=(0,0,0),width=4);d.ellipse((px-5,py-5,px+5,py+5),outline=color,width=2)
            # Spread close front/rear endpoint labels vertically; anchors stay exact.
            tx=px+6;ty=py+(7 if k>=4 else-21)
            label=f'{k}:{gr["state8_CF"][k]}'
            d.text((tx,ty),label,font=small,fill=color,stroke_width=2,stroke_fill=(0,0,0))
        zx=x0+128+2*(xy[0]-box[0]);zy=zoom_y+24+2*(xy[1]-box[1]);d.ellipse((zx-7,zy-7,zx+7,zy+7),outline=(255,153,0),width=3)
        active=gr['records_CF'][selected]['projected_active_witness_bbox_xyxy']
        records.append(dict(scene_index=ir['scene_index'],variant=ir['variant'],raw_RGB_PNG=ir['raw_RGB_PNG'],teacher_state8=gr['state8_CF'],teacher_anchor8_xy=gr['native8_CF_xy'],selected_teacher_slot=selected,diagnostic_GT_window_xyxy=list(box),inferred_query_shown=False,
            BEFORE_visible_target_bbox_xyxy=targetbox,BEFORE_visible_target_bbox_wh_px=targetwh,BEFORE_visible_target_pixels=int(targetmask.sum()),foreign_visible_pixels=gr['foreign_pixels'],selected_active_witness_bbox_wh_native_px=[active[2]-active[0],active[3]-active[1]],
            diagnostic_scope='Small target/near same-material shams may be visually subtle; no quality rejection or obviousness assumption. GT-centered display only, not an initializer query crop.'))
    canvas.save(path)
    write(out/(cfg['base_id']+'_PREVIEW_SOURCE_BINDINGS.json'),dict(completed=True,code=bind(__file__),source_base_checkpoint=bind(SOURCE/'bases'/cfg['base_id']/'BASE_COMPLETE.json'),scientific_overlay=bind(path),rows=records,raw_RGB_modified=False,teacher_anchor_not_initializer_query=True))
    return bind(path)

def previews():
    p,cfg,core=source_metadata();ready=[];pending=[]
    for b in PREVIEW_BASES:
        if(SOURCE/'bases'/cfg[b]['base_id']/'BASE_COMPLETE.json').exists():ready.append(preview_base(b,cfg))
        else:pending.append(b)
    print(json.dumps(dict(predeclared_preview_bases=list(PREVIEW_BASES),ready=ready,pending=pending,actual_model_fit=0)),flush=True)

def audit():
    started=time.monotonic();assert COMPLETE.exists(),'Full saved audit waits for completed render receipt.'
    receipt=read(COMPLETE);assert receipt['completed']and receipt['PASS']and receipt['bases']==640 and receipt['scenes']==2560
    assert receipt['target_rasters']==320 and receipt['foreign_rasters']==1920 and receipt['all_variants_retained']and receipt['same_target_RGB_bitexact']
    assert receipt['entry']['sha256']==ENTRY_SHA;entry=read(check(receipt['entry']))
    assert entry['approved_source_corpus_render']and entry['counts']==read(POLICY)['counts']and entry['preserve_original_V129_128_64']
    p,configs,core=source_metadata()
    for key in('code','core_code','protocol','policy','base_configs'):assert receipt[key]==bind({'code':Path(p['code']['path']),'core_code':CORE,'protocol':PROTOCOL,'policy':POLICY,'base_configs':CONFIGS}[key])
    assert len(receipt['base_checkpoints'])==640
    # Hash every declared saved output; cached bindings are re-used below.
    for i,binding in enumerate(receipt['outputs']):
        check(binding)
        if(i+1)%1024==0:print(json.dumps(dict(verified_declared_outputs=i+1,total=len(receipt['outputs']))),flush=True)
    ip=read(check(receipt['input_scenes']));gp=read(check(receipt['GT_scenes']));assert ip['completed']and gp['completed']and len(ip['rows'])==len(gp['rows'])==2560
    source=read(p['geometry_result']['path']);templates=[]
    for mesh in p['meshes']:templates.append(arrays(check(mesh),ARRAY_KEYS))
    geomcache={};counts=collections.Counter();reasons=collections.Counter();factorcounts=collections.defaultdict(collections.Counter)
    target_preserved=0;minimum_clear=np.inf;minimum_far=np.inf;front_margins=[];slots=0;teacher_copy_exact=0;resolution=[]
    keysbuf=('RGB_uint8','camera_depth','part_id','global_triangle_id','ambiguity','native8_CF_xy','native8_canonical_xy','permCF_to_canonical','state8_CF','camera_K','camera_R','camera_t')
    for b in range(640):
        cfg,checkpoint,inputs,gts=base_packet(b,configs)
        assert receipt['base_checkpoints'][b]==bind(SOURCE/'bases'/cfg['base_id']/'BASE_COMPLETE.json')
        assert inputs==ip['rows'][4*b:4*b+4]and gts==gp['rows'][4*b:4*b+4]
        group=cfg['geometry_group_id']
        if group not in geomcache:
            data=arrays(check(checkpoint['geometry']),ARRAY_KEYS);expected=core.scaled_data(templates[cfg['template_index']],cfg)
            for key in ARRAY_KEYS:assert np.array_equal(data[key],expected[key]),(b,key)
            for k,part in enumerate(data['native_primary_part_id']):assert np.any(np.all(data['vertices'][8*part:8*(part+1)]==data['native_anchor3D'][k],axis=1))
            geomcache[group]=data
        data=geomcache[group];foreign,translations,placement=core.foreign_placements(data,cfg)
        savedf=arrays(check(checkpoint['foreign_geometry']),('cover_vertices','near_sham_vertices','far_sham_vertices'))
        for variant,key in(('COVER','cover_vertices'),('NEAR_SHAM','near_sham_vertices'),('FAR_SHAM','far_sham_vertices')):assert np.array_equal(savedf[key],foreign[variant])
        minimum_clear=min(minimum_clear,placement['near_clearance_px']);minimum_far=min(minimum_far,placement['far_additional_px'])
        cert=core.certified_normal_mapping(cfg,core.normal_rule());front_margins.append(cert['top_cosine_margin'])
        assert cert==cfg['source_label_mapping_certificate'];perm=cfg['permCF_to_canonical'];xyz=data['native_anchor3D'];references=core.project(xyz,cfg['camera'])
        beforebuf=None;before_records=None
        for vi,(ir,gr)in enumerate(zip(inputs,gts)):
            si=4*b+vi;identity=dict(scene_index=si,base_index=b,id='V136_SCENE_%04d'%si,base_id=cfg['base_id'],split='TRAIN'if b<512 else'VALIDATION',variant=VARIANTS[vi])
            assert all(ir[k]==gr[k]==value for k,value in identity.items())
            assert ir['raw_hw']==[512,512]and gr['permCF_to_canonical']==perm and gr['camera']==cfg['camera']and gr['placement']==placement
            assert np.array_equal(np.asarray(gr['native8_CF_xy']),references[perm])
            assert gr['geometry']==checkpoint['geometry']and gr['foreign_geometry']==checkpoint['foreign_geometry']
            assert gr['target_layers_immutable']and gr['target_layer_digest']==checkpoint['target_layers_before']and gr['same_target_surface_RGB_exact']
            assert not gr['training_admitted']and gr['constructed_registration_not_legacy_or_real_native_gold']
            buf=arrays(check(gr['raw_buffers']),keysbuf);assert ir['full_RGB']==gr['raw_buffers']and ir['full_RGB_key']=='RGB_uint8'
            assert buf['RGB_uint8'].dtype==np.uint8 and buf['RGB_uint8'].shape==(512,512,3)
            assert hashlib.sha256(np.ascontiguousarray(buf['RGB_uint8']).tobytes()).hexdigest()==ir['decoded_RGB_sha256']
            assert np.array_equal(buf['native8_CF_xy'],references[perm])and np.array_equal(buf['native8_canonical_xy'],references)
            assert np.array_equal(buf['permCF_to_canonical'],perm)and buf['state8_CF'].tolist()==gr['state8_CF']
            for key,ck in(('camera_K','K'),('camera_R','R'),('camera_t','t')):assert np.array_equal(buf[key],np.asarray(cfg['camera'][ck]))
            assert gr['foreign_pixels']==int((buf['part_id']==999).sum())and gr['ambiguity_pixels']==int(buf['ambiguity'].sum())
            if vi==0:
                beforebuf=buf;before_records=gr['records_canonical'];assert gr['same_target_surface_pixels']==int((buf['part_id']>=0).sum())
                assert not(buf['part_id']==999).any()
            else:
                mask=(buf['part_id']>=0)&(buf['part_id']!=999)&(buf['part_id']==beforebuf['part_id'])&(buf['global_triangle_id']==beforebuf['global_triangle_id'])&(buf['camera_depth']==beforebuf['camera_depth'])
                assert np.array_equal(buf['RGB_uint8'][mask],beforebuf['RGB_uint8'][mask])and int(mask.sum())==gr['same_target_surface_pixels'];target_preserved+=int(mask.sum())
            targetbox=mask_box(beforebuf['part_id']>=0);foreignbox=mask_box(buf['part_id']==999);activebox=gr['records_CF'][cfg['selected_CF_index']]['projected_active_witness_bbox_xyxy']
            resolution.append(dict(scene_index=si,base_index=b,split=cfg['split'],variant=VARIANTS[vi],selected_CF_index=cfg['selected_CF_index'],selected_teacher_state=gr['state8_CF'][cfg['selected_CF_index']],
                BEFORE_target_bbox_xyxy=targetbox,BEFORE_target_wh_px=[targetbox[2]-targetbox[0],targetbox[3]-targetbox[1]]if targetbox else[0,0],BEFORE_target_pixels=int((beforebuf['part_id']>=0).sum()),
                foreign_bbox_xyxy=foreignbox,foreign_pixels=int((buf['part_id']==999).sum()),selected_active_witness_wh_native_px=[activebox[2]-activebox[0],activebox[3]-activebox[1]]))
            assert len(gr['records_canonical'])==len(gr['records_CF'])==8
            for k,record in enumerate(gr['records_canonical']):
                assert record['native_canonical_index']==k
                state,reason=canonical_rule(record,xyz,data['native_witness_points'],cfg['camera'],core);reasons[(cfg['split'],reason)]+=1
                assert record['target_witness_states']==before_records[k]['target_witness_states']and record['active_witness_indices']==before_records[k]['active_witness_indices']
                if vi==0:assert record['combined_witness_states']==record['target_witness_states']
                cf=gr['records_CF'][perm.index(k)];assert cf['canonical_index']==k and cf['CF_index']==perm.index(k)and cf['canonical_name']==record['name']
                for key,val in record.items():
                    if key!='name':assert cf[key]==val,(b,vi,k,key)
                assert gr['state8_CF'][perm.index(k)]==state;slots+=1;teacher_copy_exact+=1
                counts[(cfg['split'],VARIANTS[vi],state)]+=1
                for factor in('geometry_family','aspect','material','foreign_family','signed_quarter'):factorcounts[(cfg['split'],factor,str(cfg['factors'][factor]))][state]+=1
        if(b+1)%64==0:print(json.dumps(dict(bases_audited=b+1,slots=slots,seconds=round(time.monotonic()-started,2))),flush=True)
    assert slots==teacher_copy_exact==20480 and len(geomcache)==40
    expected_counts={fold:{s:sum(counts[(fold,variant,s)]for variant in VARIANTS)for s in'DESU'}for fold in('TRAIN','VALIDATION')}
    assert receipt['state_counts_before_query_expansion']==expected_counts
    math_audit=core.metadata_audit(configs,read(p['old_regression']['base_configs']['path'])['rows']);assert math_audit['PASS']
    for key in('policy','base_configs','render_complete','source_math','renderer'):check(p['old_regression'][key])
    result=dict(completed=True,PASS=True,code=bind(__file__),completed_render=bind(COMPLETE),root_entry=receipt['entry'],bases=640,scenes=2560,slots=20480,geometry_groups=40,
        declared_output_bindings_verified=len(receipt['outputs']),unique_saved_files_hashed=len(checked),all640_base_checkpoints_exact=True,all2560_aggregate_and_perbase_rows_exact=True,
        all_registered_arrays_positive_scale_exact=True,all8anchors_literal_designated_part_vertices=True,all640normal_rule_exact=True,minimum_normal_top_cosine_margin=min(front_margins),
        all20480_teacher_rule_reductions_and_CFcanonical_copies_exact=True,teacher_recompute_scope='State reduction, active-index/in-frame geometry and saved37ray-classification consistency. No new raycasting or teacher generation executed.',
        target_alone_ray_classifications_and_active_sets_preserved=True,target_RGB_surviving_same_triangle_depth_bitexact=True,target_preserved_pixels_across_foreign_variants=target_preserved,
        minimum_near_sham_clearance_px=minimum_clear,minimum_far_additional_px=minimum_far,all_foreign_saved_vertices_match_analytic_placement_exact=True,
        state_counts=expected_counts,counts_by_fold_variant_state={str(k):v for k,v in counts.items()},teacher_reason_counts={str(k):v for k,v in reasons.items()},
        resolution_and_foreign_visibility=resolution,source_visual_limit='All rows retained. Small target resolution and subtle SAME_TARGET_MATERIAL/NEAR_SHAM contrast are measured facts, not an assumption every sham is visually obvious.',
        counts_by_factor_state={str(k):dict(v)for k,v in factorcounts.items()},source_split_and_recipe_audit=math_audit,old128_64_bindings_unchanged=True,
        no_GT_state_filtering=True,no_new_topology=True,new_source_VAL_not_human_application_gold=True,actual_teacher_calls=0,model_fit_private_V112_PnP_query_access=0,seconds=time.monotonic()-started)
    write(H/'SAVED_SOURCE_AUDIT_COMPLETE.json',result)
    for b in PREVIEW_BASES:preview_base(b,configs)
    print(json.dumps(dict(completed=True,PASS=True,audit=bind(H/'SAVED_SOURCE_AUDIT_COMPLETE.json'),state_counts=expected_counts,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--previews-if-ready',action='store_true');ap.add_argument('--full-audit',action='store_true');a=ap.parse_args()
    try:
        if a.previews_if_ready and not a.full_audit:previews()
        elif a.full_audit and not a.previews_if_ready:audit()
        else:raise SystemExit('Use --previews-if-ready or --full-audit')
    except BaseException:
        path=H/('FAILED_'+str(time.time_ns())+'.json');write(path,dict(code=bind(__file__),traceback=traceback.format_exc(),automatic_retry=False,source_files_mutated=False));raise
