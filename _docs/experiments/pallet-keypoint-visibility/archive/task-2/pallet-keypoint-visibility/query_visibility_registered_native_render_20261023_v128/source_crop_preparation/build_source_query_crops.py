"""Registered source q emulation + exact query-only crops. Root-entry execution only."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,math,sys,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OUT=V/'source_crops_v1';ROOT_POLICY=V/'ROOT_SOURCE_CROP_BASELINE_POLICY_BEFORE_INPUTS.json'
RENDER=V/'actual_render_v1/RENDER_COMPLETE.json'
OPERATOR=V/'review_preparation/SOURCE_QUERY_OPERATOR_CANDIDATE.json'
PROTOCOL=H/'QUERY_CROP_PROTOCOL_BEFORE_PAYLOADS.json';GENERATED=H/'GENERATED_CROP_TESTS.json'
OPS=dict(local=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py',context=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py',side=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py')
OFFSETS=[(dx,dy)for dy in(-3.,0.,3.)for dx in(-3.,0.,3.)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])=={k:b[k]for k in('path','sha256')},b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def pure(path,names):
    body=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert {n.name for n in body}==set(names);ns={'np':np};exec(compile(ast.Module(body=body,type_ignores=[]),str(path)+'::pure_only','exec'),ns);return ns
def operators():
    out=pure(OPS['local'],('crop','query_validity'));out.update(pure(OPS['context'],('query_extent','valid_box','target_context_grid','sample_target_context')));out.update(pure(OPS['side'],('native_grid',)));return out
def query_condition(ref8,j,offset,hw,op):
    q=np.array(ref8,np.float64,copy=True);good=np.isfinite(q).all(1)&~(q==[-1.,-1.]).all(1)
    q[~good]=[-1.,-1.]
    if good[j]:q[j]+=np.asarray(offset,np.float64)
    _,roi,L=op['native_grid'](q,hw);box=op['query_extent'](q)
    return q,L,box,bool(roi[j].any())
def validate_entry(path):
    e=read(path);allowed={'approved_actual_source_crop_build','source_only','model_allowed','fit_allowed','current_allowed','V112_allowed','PnP_allowed','restart_allowed','code','protocol','generated','render_complete','root_policy','output'}
    assert set(e)==allowed,'Strict crop entry schema'
    assert e['approved_actual_source_crop_build']is True and e['source_only']is True
    assert all(e[k]is False for k in('model_allowed','fit_allowed','current_allowed','V112_allowed','PnP_allowed','restart_allowed'))
    for k in('code','protocol','generated','render_complete','root_policy'):check(e[k])
    p=read(PROTOCOL);assert e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['generated']==p['generated']==bind(GENERATED)
    assert e['render_complete']==p['render_complete']==bind(RENDER)and e['root_policy']==p['root_policy']==bind(ROOT_POLICY)
    assert e['output']==p['output']==str(OUT)and not OUT.exists(),'Preserve prior completed/failed attempt'
    for k,b in p['operators'].items():assert b==bind(OPS[k])
    check(p['scene_manifest']);check(p['operator_candidate']);assert read(GENERATED)['PASS']and read(RENDER)['completed']and read(RENDER)['PASS']
    return e,p

def run(entry):
    e,p=validate_entry(entry) # All approval/bindings/output-absence before actual buffer/RGB reads.
    op=operators();manifest=read(p['scene_manifest']['path']);scenes=manifest['scenes'];assert manifest['completed']and len(scenes)==6
    assert len({(s['style'],s['variant'])for s in scenes})==6
    OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),query_emulation_only=True,RGB_reads=0,training_admission=False))
    try:
        # Read only reference coordinates first; no RGB/state/owner key accessed.
        references=[];style_ref={};base=[]
        for si,s in enumerate(scenes):
            bp=check(s['buffers']);assert bp.resolve().is_relative_to(RENDER.parent.resolve())
            with np.load(bp,allow_pickle=False)as z:ref=np.array(z['native8_xy_continuous'],copy=True)
            assert ref.shape==(8,2)and ref.dtype==np.float64
            if s['style']in style_ref:assert np.array_equal(ref,style_ref[s['style']],equal_nan=True)
            else:style_ref[s['style']]=ref.copy()
            references.append(ref)
            for j in range(8):
                for k,offset in enumerate(OFFSETS):
                    q8,L,box,roi_any=query_condition(ref,j,offset,[256,256],op);i=72*si+9*j+k;scene_id=s['style']+'__'+s['variant'];group=s['style']+f'__slot{j}__jitter{k}'
                    base.append(dict(row_id=i,occurrence_index=i,id=f'{scene_id}::slot{j}::jitter{k}',query_id=group+'::'+s['variant'],query_group_id=group,scene_id=scene_id,scene_index=si,style=s['style'],variant=s['variant'],source='NEW_REGISTERED_PROCEDURAL_PALLET',split='SOURCE_DIAGNOSTIC',raw_split='SOURCE_DIAGNOSTIC',source_frame_group=s['style'],query_index=j,logical_slot=j,carrier_slot=j,jitter_index=k,offset_xy=list(offset),query_xy=q8[j].tolist(),q8=q8.tolist(),initial_points=q8.tolist(),L=float(L),context_xyxy=box,context_box_xyxy=box,HW=[256,256],raw_hw=[256,256],ROI_any_valid=roi_any,raw_rgb=s['images']['RGB'],full_RGB=s['buffers'],full_RGB_format='NPZ_RGB_UINT8_HWC',full_RGB_key='RGB_uint8',source_query_emulation=True,actual_initializer_output=False))
        assert len(base)==432 and [x['row_id']for x in base]==list(range(432))
        lock=OUT/'QUERY_ROWS432_BEFORE_RGB.json';write(lock,dict(completed=True,rows=base,source_reference='Only native8_xy_continuous read to create source q; no RGB or teacher-state/owner keys read before this lock.',signed_yaw='YAW_0; camera-facing index equals canonical index for these fixed scenes.',protocol=e['protocol'],render_complete=e['render_complete'],new_labels_or_training_admission=False))
        from PIL import Image,ImageDraw,ImageFont
        (OUT/'local96').mkdir();(OUT/'figures').mkdir();rgbpack=np.empty((432,96,96,3),np.uint8);vpack=np.empty((432,96,96),bool);inputs=[];gt=[];images=[];scene_cache=[]
        for si,s in enumerate(scenes):
            with np.load(s['buffers']['path'],allow_pickle=False)as z:
                rgb=z['RGB_uint8'];states=z['continuous_region_states'];owner=z['part_id'];amb=z['ambiguity'];tie=z['tie_count']
            assert rgb.shape==(256,256,3)and rgb.dtype==np.uint8 and states.shape==(8,)and owner.shape==amb.shape==tie.shape==(256,256)
            assert np.isin(states,['D','E','S','U']).all();decoded=ash(rgb);check(s['images']['RGB']);images.append(rgb.copy());scene_cache.append(s)
            for r0 in base[si*72:(si+1)*72]:
                r=copy.deepcopy(r0);i=r['row_id'];j=r['query_index'];q=np.array(r['query_xy']);L=r['L'];crop,v=op['crop'](rgb,q,L,96)
                a=crop.transpose(1,2,0).copy();vv=v[0].copy();rgbpack[i]=a;vpack[i]=vv
                qgood,cont,near=op['query_validity'](q[None],[256,256]);central=bool(vv[47:49,47:49].all())
                context=False
                if r['context_xyxy']is not None:_,tv=op['sample_target_context'](rgb,r['context_xyxy']);context=bool(tv.any())
                factual=bool(qgood[0]and central);available=factual and context
                reason=[]
                if not cont[0]:reason.append('NATIVE_CONTINUOUS_QUERY_UNAVAILABLE')
                if not near[0]:reason.append('NATIVE_NEAREST_QUERY_UNAVAILABLE')
                if not central:reason.append('LOCAL96_CENTRAL4_UNAVAILABLE')
                if not context:reason.append('CONTEXT_UNAVAILABLE')
                f=OUT/'local96'/f'ROW_{i:04d}.png';Image.fromarray(a).save(f)
                r.update(decoded_RGB_sha256=decoded,local96=bind(f),rgb96_row=i,rgb96_sha256=ash(a),valid96_sha256=ash(vv),rgb96_layout='HWC',valid96_layout='HW',q_native_continuous_valid=bool(cont[0]),q_nearest_valid=bool(near[0]),local_central4_valid=central,factual_available=factual,factual_query_available=factual,context_available=context,query_available=available,unavailability_reasons=reason,original_input_availability=dict(continuous=bool(cont[0]),nearest=bool(near[0]),ROI_any_valid=r['ROI_any_valid'],central4=central,context=context))
                inputs.append(r)
                ref=references[si][j];rec=s['continuous_region_records'][j];assert rec['native_canonical_index']==j and rec['state']==str(states[j])
                assert np.array_equal(ref,np.array(rec['projected_reference_xy']),equal_nan=True)
                pix=np.floor(q+.5).astype(int);inside=bool(near[0]);xy=(int(pix[0]),int(pix[1]));state=str(states[j]);reference_finite=bool(np.isfinite(ref).all())
                gt.append(dict(row_id=i,occurrence_index=i,id=r['id'],scene_id=r['scene_id'],canonical_index=j,camera_facing_index=j,reference_xy=ref.tolist()if reference_finite else None,delta_xy=r['offset_xy'],error_px=float(np.linalg.norm(q-ref))if reference_finite else None,raw_state=state,external_target_D0_E1_else_minus1=0 if state == 'D' else 1 if state == 'E' else -1,known_geometric_S=state=='S',teacher_reason='Unchanged continuous registered region witness teacher; qpixel membership is diagnostic only.',witnessrefs=dict(file=p['scene_manifest'],scene_index=si,record_key='continuous_region_records',record_index=j,active_witness_indices=rec['active_witness_indices']),original_continuous_region_record=rec,q_nearest_pixel_owner_diagnostic=dict(nearest_xy=list(xy),in_frame=inside,part_owner_id=int(owner[xy[1],xy[0]])if inside else None,ambiguity=bool(amb[xy[1],xy[0]])if inside else None,tie_count=int(tie[xy[1],xy[0]])if inside else None),source_native_registration='Constructed exactly for these NEW procedural assets only; no real/legacy-human equivalence.',human_gold=False,training_admission=False))
        assert len(inputs)==len(gt)==432
        # Pair geometry/operator identities are invariant across the3 variants.
        bygroup={}
        for r in inputs:bygroup.setdefault(r['query_group_id'],[]).append(r)
        assert len(bygroup)==144 and all(len(x)==3 for x in bygroup.values())
        for rr in bygroup.values():
            assert {x['variant']for x in rr}=={'BEFORE','COVER','SHAM'}
            assert all(all(x[k]==rr[0][k]for k in('query_xy','q8','L','context_xyxy','query_available'))for x in rr)
        pack=OUT/'RGB96_INPUT_ONLY.npz';np.savez_compressed(pack,rgb96=rgbpack,valid96=vpack,occurrence_index=np.arange(432,dtype=np.int32))
        ip=OUT/'INPUT_ROWS.json';gp=OUT/'GT_ROWS.json'
        write(ip,dict(completed=True,rows=inputs,protocol=e['protocol'],query_lock=bind(lock),render_complete=e['render_complete'],stored_crops=bind(pack),source_query_emulation_not_actual_initializer=True,training_admission=False))
        write(gp,dict(completed=True,rows=gt,protocol=e['protocol'],input_row_identity=bind(ip),raw_S_known_geometry_outside_external_task=True,U_unresolved_not_hidden_negative=True,no_training_admission=True))
        font=ImageFont.load_default();contacts=[];overview=Image.new('RGB',(840,630),'#eef3f7');od=ImageDraw.Draw(overview)
        for si,s in enumerate(scenes):
            board=Image.new('RGB',(636,735),'#f8fafc');dr=ImageDraw.Draw(board);dr.text((10,8),s['style']+' '+s['variant']+' | native index1 | raw96',fill='#193b52',font=font)
            for k in range(9):
                i=si*72+9+k;r=inputs[i];im=Image.fromarray(rgbpack[i]).resize((192,192),Image.Resampling.NEAREST);x=12+(k%3)*208;y=45+(k//3)*228;board.paste(im,(x,y));dr.text((x,y+195),f'row{i} delta{r["offset_xy"]} L{r["L"]:.3f}',fill='#193b52',font=font);dr.text((x,y+209),f'available {r["query_available"]} | source emulation',fill='#193b52',font=font)
            cf=OUT/'figures'/f'NATIVE1_JITTER9_{si:02d}.png';board.save(cf);contacts.append(bind(cf));ox=(si%3)*280+12;oy=(si//3)*310+35;overview.paste(Image.fromarray(images[si]),(ox,oy));od.text((ox,oy-22),s['style']+' '+s['variant'],fill='#193b52',font=font)
        of=OUT/'figures/OVERVIEW_RAW6.png';overview.save(of)
        counts={s:sum(g['raw_state']==s for g in gt)for s in'DESU'}
        complete=dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),protocol=e['protocol'],render_complete=e['render_complete'],query_lock=bind(lock),input_manifest=bind(ip),GT_only=bind(gp),stored_crops=bind(pack),rows=432,query_groups=144,scenes=6,original_slots=8,jitters=9,raw_states=counts,input_available=sum(r['query_available']for r in inputs),factual_input_available=sum(r['factual_available']for r in inputs),source_reference_and_teacher_unchanged=True,yaw0_identity=True,raw_PNGs=432,contacts=contacts,overview=bind(of),files=[bind(ip),bind(gp),bind(pack),bind(lock)]+contacts+[bind(of)],training_admission=False,new_model_fit_current_V112_PnP=0)
        write(OUT/'CROP_COMPLETE.json',complete);print(json.dumps(dict(completed=True,receipt=bind(OUT/'CROP_COMPLETE.json'),rows=432,states=counts)),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_RESTART.json',dict(traceback=traceback.format_exc(),automatic_restart=False));raise

def generated():
    op=operators();tests={};q=np.array([[10.,10.],[110.,10.],[110.,60.],[10.,60.],[20.,20.],[100.,20.],[100.,50.],[20.,50.]])
    before=q.copy();qq,L,box,roi=query_condition(q,1,(3.,-3.),[128,128],op)
    assert np.array_equal(q,before)and np.array_equal(qq[np.arange(8)!=1],q[np.arange(8)!=1])and np.array_equal(qq[1],[113.,7.]);assert L==25.75 and box==[10.,7.,113.,60.]and roi;tests['one_slot_jitter_exact_side_and_unclipped_context']=True
    assert OFFSETS==[(-3.,-3.),(0.,-3.),(3.,-3.),(-3.,0.),(0.,0.),(3.,0.),(-3.,3.),(0.,3.),(3.,3.)];tests['six_eight_nine_order_and_shared_variant_group']=len({(si,j,k)for si in range(6)for j in range(8)for k in range(9)})==432
    im=np.full((128,128,3),73,np.uint8);a,v=op['crop'](im,np.array([64.,64.]),32.,96);assert a.shape==(3,96,96)and(a==73).all()and v.all();tests['exact_crop_constant_RGB_and_valid_layout']=True
    a,v=op['crop'](im,np.array([-5.,-5.]),16.,96);assert not a[:,~v[0]].any()and not v[:,47:49,47:49].all();tests['OOB_zeroing_and_factualU_separate_from_teacher']=True
    assert op['query_extent'](np.full((8,2),-1.))is None;tests['missing_query_context_no_GT_fallback']=True
    missing=q.copy();missing[1]=[-1.,-1.];sentinel,_,_,_=query_condition(missing,1,(3.,-3.),[128,128],op);assert np.array_equal(sentinel[1],[-1.,-1.]);tests['sentinel_query_never_jittered']=True
    rr=np.array([65.25,63.75]);query=np.array([64.,64.]);loc=(rr-query)*96/32+47.5;assert np.array_equal(query+((loc+.5)/96-.5)*32,rr);tests['query_phase_inverse_exact']=True
    try:validate_entry_json_fixture()
    except AssertionError:tests['unapproved_entry_stops_before_payload']=True
    assert all(tests.values());write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),operators={k:bind(p)for k,p in OPS.items()},tests=tests,actual_render_buffers_RGB_models_fit_current_V112=0));print(json.dumps(dict(generated=bind(GENERATED))))
def validate_entry_json_fixture():
    # Exercise the actual entry guard with invented metadata, before any binding/data reads.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='v128_unapproved_entry_')as d:
        e=dict(approved_actual_source_crop_build=False,source_only=True,model_allowed=False,fit_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,restart_allowed=False,code={},protocol={},generated={},render_complete={},root_policy={},output='invented')
        path=Path(d)/'UNAPPROVED.json';write(path,e);validate_entry(path)
def prepare():
    g=read(GENERATED);assert g['PASS']and g['code']==bind(__file__);rr=read(RENDER);assert rr['completed']and rr['PASS']and rr['scenes']==6
    sb=next(b for b in rr['outputs']if Path(b['path']).name=='SCENES6_AND_WITNESS_DIAGNOSTICS.json');check(sb)
    p=dict(completed=True,code=bind(__file__),generated=bind(GENERATED),operators={k:bind(p)for k,p in OPS.items()},root_policy=bind(ROOT_POLICY),render_complete=bind(RENDER),scene_manifest=sb,operator_candidate=bind(OPERATOR),output=str(OUT),order='scene manifest order then slot0..7 then jitterdy outer/dx inner {-3,0,3}; row=scene*72+slot*9+jitter',signed_yaw='YAW_0 identity only for these fixed scenes',data_layout=dict(rgb96=[432,96,96,3],valid96=[432,96,96],occurrence_index=[432]),input_GT_boundary='Freeze all q8/operator rows before accessing RGB/teacher-state/owner keys; actual source projection-derived q emulation explicitly recorded. GT/raw states and owner diagnostics remain separate.',availability='Factual=continuous&nearest&central4; context-any separate; v61 may independently use historicalROIany. No teacher-state availability gate.',model_fit_current_V112_PnP=0,training_admission=False)
    write(PROTOCOL,p);template=dict(approved_actual_source_crop_build=False,source_only=True,model_allowed=False,fit_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,restart_allowed=False,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),render_complete=p['render_complete'],root_policy=p['root_policy'],output=str(OUT));write(H/'ROOT_CROP_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_CROP_ENTRY_TEMPLATE.json'),command=f'{sys.executable} {Path(__file__).resolve()} --entry ROOT_ACTUAL_SOURCE_CROP_ENTRY.json',actual_source_payload_reads_or_model_calls=0)),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.generated:generated()
    elif a.prepare:prepare()
    elif a.entry:run(a.entry)
    else:raise SystemExit('Use --generated / --prepare / --entry ROOT_EXACT_ENTRY')
