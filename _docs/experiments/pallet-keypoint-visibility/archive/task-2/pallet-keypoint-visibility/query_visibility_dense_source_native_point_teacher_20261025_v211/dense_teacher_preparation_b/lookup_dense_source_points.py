"""Dense saved own-scene SOURCE point metadata; no model/array/projection code."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,os,stat,traceback
H=Path(__file__).parent;V=H.parent;P=V.parent
S=V/'ROOT_PROSPECTIVE_DENSE_SOURCE_POINT_TEACHER_LOOKUP_SCOPE.json'
S_SHA='a77f60c1a8a41a4738c4713be5ceb39d5653628963e3616b2021207d675af9cc'
OLD_PROTO=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'
OLD_PROTO_SHA='8df60dc146b5f026563341a1d959ecff20ef501333a14694c3b0794a2c0e07d8'
OUT=V/'dense_source_point_teacher_v1'
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
STATES=('D','E','S','U','UNKNOWN')
GT_ALLOWED={}

def raw(path,gt=False):
    p=Path(path)
    assert p.is_absolute()and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix in {'.json','.jsonl','.py'}and not p.name.startswith('BASE_CONFIGS')
    if p.name=='SCENES_GT_ONLY.json':assert gt and str(p)in GT_ALLOWED,'Root actual GT whitelist before filesystem'
    else:assert not gt
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def bind(path):return dict(path=str(path),sha256=hashlib.sha256(raw(path)).hexdigest())
def get(b,gt=False):
    if gt:assert GT_ALLOWED.get(b['path'])==b
    z=raw(b['path'],gt);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def lines(b):
    z=raw(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return [json.loads(s)for s in z.splitlines()if s.strip()]
def write(path,obj):
    assert path.parent in (H,OUT)and 'v112'not in str(path).casefold()
    with path.open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rowwrite(path,rows):
    assert path.parent in(H,OUT)and 'v112'not in str(path).casefold()
    with path.open('x',encoding='utf-8')as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def map_input(row,scene_map):
    """Image scene comes from original image identity; NEVER anchor/cache index."""
    k=row['native_slot'];image=row.get('source_image_original_global_index');anchor=row.get('anchor_original_global_index')
    base=dict(row,component_role='TRAIN',source_teacher_key=None,mapping_status='UNKNOWN_MISSING_EXACT_IMAGE_SOURCE_IDENTITY',new_training_admission=False)
    if type(image)is not int or type(anchor)is not int:return base
    assert image%8==anchor%8==k
    si=image//8;ai=anchor//8
    if si not in scene_map or ai not in scene_map:return base
    source=scene_map[si];anchor_scene=scene_map[ai]
    assert source['scene_index']==row['scene_index']and source['base_index']==row['base_index']and source['variant']==row['variant']
    assert row['anchor_scene_index']==ai
    own=source['original_input_metadata8'][k];anch=anchor_scene['original_input_metadata8'][k]
    assert own['original_global_index']==image and anch['original_global_index']==anchor
    assert row['query_xy']==anch['original_query_xy']and row['L']==anch['original_L']
    assert row['original_query_available']==anch['original_input_flags']['query_available']
    assert row['source_image_original_query_id']==own['original_occurrence_id']and row['anchor_original_query_id']==anch['original_occurrence_id']
    assert row['original_flags_ref']==[ai,k]and row['image_flags_ref']==[si,k]
    if row['family']=='SELF_SCENE15':assert si==ai and row['state_index']==0 and row['state_name']=='NOMINAL'
    else:assert row['family']=='BEFORE_ANCHOR4'and anchor_scene['variant']=='BEFORE'
    base.update(source_teacher_key=[si,k],mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE',actual_image_scene_index=si,actual_image_variant=source['variant'],
        actual_image_original_global_index=image,actual_image_original_occurrence_id=own['original_occurrence_id'],
        anchor_scene_index=ai,anchor_variant=anchor_scene['variant'],anchor_query_original_occurrence_id=anch['original_occurrence_id'],
        actual_image_source_binding_inert=source['source_RGB_binding'],original_input_flags=anch['original_input_flags'],
        actual_image_original_input_flags_context_only=own['original_input_flags'],declared_CF_to_canonical=own['declared_CF_to_canonical'])
    return base

def select_scene(doc,scene):
    if not isinstance(doc,dict)or not isinstance(doc.get('rows'),list):return None,'UNKNOWN_EXPLICIT_TOP_ROWS_SCHEMA'
    candidates=[r for r in doc['rows']if isinstance(r,dict)and r.get('variant')==scene['variant']and r.get('scene_index')==scene['scene_index']]
    assert len(candidates)<=1,'Ambiguous ownscene'
    if not candidates:return None,'UNKNOWN_MISSING_EXACT_OWN_SCENE'
    own=candidates[0];assert own['base_index']==scene['base_index']
    rr=own.get('records_CF');perm=own.get('permCF_to_canonical')
    if not isinstance(rr,list)or len(rr)!=8 or not all(isinstance(r,dict)and 'CF_index'in r and 'canonical_index'in r for r in rr):return None,'UNKNOWN_MISSING_EXPLICIT_RECORDS_CF'
    if not isinstance(perm,list)or len(perm)!=8:return None,'UNKNOWN_MISSING_EXPLICIT_SCENE_PERM'
    assert all(type(r['CF_index'])is int and type(r['canonical_index'])is int for r in rr)
    assert all(type(x)is int for x in perm)and sorted(perm)==list(range(8))
    records={r['CF_index']:r for r in rr};assert set(records)==set(range(8))and sorted(r['canonical_index']for r in rr)==list(range(8))
    for original in scene['native_rows']:
        k=original['native_slot'];r=records[k]
        assert r['canonical_index']==perm[k]==original['declared_CF_to_canonical']
        if 'native_canonical_index'in r:assert r['native_canonical_index']==perm[k]
    return records,None

def extract(record):
    def state(name,point=False):
        value=None if record is None else record.get(name)
        if point:value=value[0]if isinstance(value,list)and value else None
        return value if isinstance(value,str)and value in STATES[:-1]else'UNKNOWN'
    reference=None if record is None else record.get('projected_reference_xy')
    if not(isinstance(reference,list)and len(reference)==2 and all(type(v)in(int,float)and math.isfinite(v)for v in reference)):reference=None
    return dict(point0_target_alone_state=state('target_witness_states',True),point0_combined_state=state('combined_witness_states',True),
        region_state_context_only=state('state'),target_alone_region_state_context_only=state('target_alone_region_state'),saved_projected_reference_xy=reference)

def locality(q,L,available,reference,raw_hw):
    out=dict(original_input_available=available,original_input_U_preserved=not available,raw96_reference_index_xy=None,
        inside_sample_center_extent=None,inside_crop_canvas_extent=None,inside_original_frame_canvas_extent=None,
        not_coordinate_error_correction_or_pixel_visibility_proof=True)
    if reference is None:return out
    h,w=raw_hw;out['inside_original_frame_canvas_extent']=(-.5<=reference[0]<w-.5 and -.5<=reference[1]<h-.5)
    if not(isinstance(q,list)and len(q)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in q)and type(L)in(int,float)and math.isfinite(L)and L>0):return out
    u=[47.5+96*(reference[i]-q[i])/L for i in range(2)]
    out.update(raw96_reference_index_xy=u,inside_sample_center_extent=all(0<=v<=95 for v in u),inside_crop_canvas_extent=all(-.5<=v<95.5 for v in u))
    return out

def prepare():
    assert bind(S)['sha256']==S_SHA and bind(OLD_PROTO)['sha256']==OLD_PROTO_SHA
    scope=get(bind(S));proposal=get(scope['proposed_SOURCE_GT']);ready=get(scope['readiness']);old=get(bind(OLD_PROTO));b=old['context_input_bindings']
    assert ready['PASS']and proposal['TRAIN_bases']==110
    parents=proposal['parents'];base_ids={r['base_index']for r in parents};assert len(parents)==len(base_ids)==110
    train={r['row_id']:r for r in lines(b['SYNTH_AUG65536_roles'])if r['role']=='TRAIN'}
    aug=[r for r in lines(b['SYNTH_AUG65536'])if r['row_id']in train]
    assert {r['base_index']for r in aug}==base_ids
    scenes=[s for s in lines(b['original_scenes'])if s['base_index']in base_ids]
    smap={s['scene_index']:s for s in scenes};assert len(scenes)==len(smap)==440
    cache=get(old['old_cache_bindings']['augmentation']);assert cache['metadata']==b['SYNTH_AUG65536']
    parentmap={p['base_index']:p for p in parents};selected_scenes=[];native=[]
    for s in sorted(scenes,key=lambda x:x['scene_index']):
        parent=parentmap[s['base_index']];assert s['split']=='TRAIN'and s['variant']in VARIANTS
        ref=next(x for x in parent['selected_original_scenes']if x['scene_index']==s['scene_index'])
        assert ref['variant']==s['variant']and ref['source_RGB_binding_NOT_OPENED']==s['source_RGB_binding']
        assert [r['native_slot']for r in s['original_input_metadata8']]==list(range(8))
        assert sorted(r['declared_CF_to_canonical']for r in s['original_input_metadata8'])==list(range(8))
        ns=[]
        for o in s['original_input_metadata8']:
            assert o['scene_index']==s['scene_index']and o['base_index']==s['base_index']and o['variant']==s['variant']
            r=dict(teacher_key=[s['scene_index'],o['native_slot']],role='TRAIN',scene_index=s['scene_index'],base_index=s['base_index'],variant=s['variant'],
                native_slot=o['native_slot'],declared_CF_to_canonical=o['declared_CF_to_canonical'],original_global_index=o['original_global_index'],original_occurrence_id=o['original_occurrence_id'],
                query_xy=o['original_query_xy'],L=o['original_L'],original_query_available=o['original_input_flags']['query_available'],original_input_flags=o['original_input_flags'],
                raw_hw=s['raw_hw'],factors=s['factors'],source_components=parent['source_component_ids'],source_RGB_binding_inert=s['source_RGB_binding'],
                original_saved_raw96_binding_inert={k:o[k]for k in ('saved_crop_chunk','saved_crop_row','saved_crop_keys','saved_rgb96_sha256_inherited','saved_valid96_sha256_inherited')},
                new_training_admission=False)
            native.append(r);ns.append(r)
        selected_scenes.append(dict(scene_index=s['scene_index'],base_index=s['base_index'],variant=s['variant'],native_rows=ns))
    assert len(native)==3520 and len({tuple(r['teacher_key'])for r in native})==3520
    eligible=[r for r in aug if r['family']=='BEFORE_ANCHOR4'or(r['family']=='SELF_SCENE15'and r['state_index']==0 and r['state_name']=='NOMINAL')]
    assert len(eligible)==7040
    mapping=[]
    for r in eligible:
        role=train[r['row_id']]
        for key in ('native_slot','query_xy','L','original_query_available','source_image_original_global_index','original_global_index'):assert r[key]==role[key]
        m=map_input(r,smap);m.update(component_id=role['component_id'],input_cache_index=role['cache_index'],input_cache_binding_key='SYNTH_AUG65536')
        mapping.append(m)
    assert len({r['row_id']for r in mapping})==7040
    whitelist=[p['GT_json_binding_NOT_OPENED']for p in parents]
    assert len({x['path']for x in whitelist})==110
    for gt in whitelist:
        p=Path(gt['path']);assert p.name=='SCENES_GT_ONLY.json'and 'v112'not in str(p).casefold()and '..'not in p.parts and len(gt['sha256'])==64
    rowwrite(H/'ALL3520_ORIGINAL_OWN_SCENE_NATIVE_SELECTION.jsonl',native);rowwrite(H/'ALL7040_C61_ELIGIBLE_IMAGE_SOURCE_MAPPING.jsonl',mapping)
    write(H/'GT110_WHITELIST_NOT_OPENED.json',dict(completed=True,GT_json_whitelist=whitelist,GT_stat_hash_body_reads=0))
    sources=dict(scope=bind(S),readiness=scope['readiness'],proposed_GT=scope['proposed_SOURCE_GT'],old_protocol=bind(OLD_PROTO),
        source_scenes=b['original_scenes'],aug_rows=b['SYNTH_AUG65536'],roles=b['SYNTH_AUG65536_roles'],input_cache_complete=old['old_cache_bindings']['augmentation'])
    selection=dict(completed=True,PASS=True,parents=parents,scenes=selected_scenes,counts=dict(parents=110,scenes=440,own_native=3520,C61_eligible_inputs=7040),
        metadata_sources=sources,input_source_cache_binding_inert=cache['inputs'],input_mapping_rule='Use source_image_original_global_index//8 actual scene and slot, not augmented cache original_global_index or BEFORE anchor. q/L/flags remain anchor inputs.',
        metadata_counts=dict(own_inputU=sum(not r['original_query_available']for r in native),mapped_inputU=sum(not r['original_query_available']for r in mapping),mapping_status=dict(Counter(r['mapping_status']for r in mapping))),
        GT_original_arrays_headers_read=0,new_labels_or_admission=False)
    write(H/'SELECTION_METADATA_ONLY.json',selection)
    print(json.dumps(dict(completed=True,PASS=True,counts=selection['counts'],metadata_counts=selection['metadata_counts'])))

def run(entry_path):
    e=get(bind(Path(entry_path)));assert e.get('approved_SOURCE_TRAIN_dense_saved_point_lookup')is True and e.get('new_training_admission')is False
    assert e.get('root_actual_scope')is not None;get(e['root_actual_scope']);assert Path(e['root_actual_scope']['path']).parent==V
    for k,name in [('code','lookup_dense_source_points.py'),('protocol','PROTOCOL_BEFORE_DENSE_TRAIN_GT.json'),('selection','SELECTION_METADATA_ONLY.json'),('whitelist','GT110_WHITELIST_NOT_OPENED.json'),('original_native_rows','ALL3520_ORIGINAL_OWN_SCENE_NATIVE_SELECTION.jsonl'),('input_mapping','ALL7040_C61_ELIGIBLE_IMAGE_SOURCE_MAPPING.jsonl')]:assert e[k]==bind(H/name)
    sel=get(e['selection']);whitelist=get(e['whitelist'])['GT_json_whitelist'];assert e['GT_json_whitelist']==whitelist
    assert e['output_directory']==str(OUT)and e.get('automatic_retry')is False
    for b in sel['metadata_sources'].values():assert bind(b['path'])==b # Opaque text closure only; do not re-analyze shared roles.
    assert len(whitelist)==110 and all(p['GT_json_binding_NOT_OPENED']in whitelist for p in sel['parents'])
    GT_ALLOWED.update({b['path']:b for b in whitelist})
    original=lines(e['original_native_rows']);mapping=lines(e['input_mapping']);assert len(original)==3520 and len(mapping)==7040
    assert original==[r for s in sel['scenes']for r in s['native_rows']],'Exact selected native metadata/order'
    OUT.mkdir(exist_ok=False);done=[];schemas=[]
    try:
        for parent in sel['parents']:
            receipt=get(parent['base_completion']);assert receipt['completed']and receipt['PASS']and receipt['GT_scenes']==parent['GT_json_binding_NOT_OPENED']
            doc=get(parent['GT_json_binding_NOT_OPENED'],True)
            own=[s for s in sel['scenes']if s['base_index']==parent['base_index']];assert len(own)==4 and {s['variant']for s in own}==set(VARIANTS)
            for scene in own:
                records,reason=select_scene(doc,scene);schemas.append(dict(base_index=scene['base_index'],scene_index=scene['scene_index'],variant=scene['variant'],UNKNOWN_schema_reason=reason))
                for o in scene['native_rows']:
                    extracted=extract(None if records is None else records[o['native_slot']])
                    done.append(dict(o,**extracted,UNKNOWN_schema_reason=reason,reference_locality=locality(o['query_xy'],o['L'],o['original_query_available'],extracted['saved_projected_reference_xy'],o['raw_hw']),authority='SAVED_PROCEDURAL_OWN_SCENE_POINT0_NOT_REAL_NATIVE_GOLD',new_native_target_array_created=False))
        teacher={tuple(r['teacher_key']):r for r in done};assert len(done)==len(teacher)==3520
        mapped=[]
        for r in mapping:
            t=teacher.get(tuple(r['source_teacher_key']))if r['source_teacher_key']is not None else None
            values=extract(None)if t is None else {k:t[k]for k in ('point0_target_alone_state','point0_combined_state','region_state_context_only','target_alone_region_state_context_only','saved_projected_reference_xy')}
            mapped.append(dict(r,**values,reference_locality_for_actual_input=locality(r['query_xy'],r['L'],r['original_query_available'],values['saved_projected_reference_xy'],[0,0]if t is None else t['raw_hw']),
                point_state_applies_to='ACTUAL_ORIGINAL_IMAGE_SCENE_NOT_ANCHOR_OR_CACHE_ID',new_native_target_array_created=False))
        assert len(mapped)==7040 and [r['row_id']for r in mapped]==[r['row_id']for r in mapping]
        rowwrite(OUT/'ALL3520_OWN_SCENE_NATIVE_POINT_TEACHER_METADATA.jsonl',done);rowwrite(OUT/'ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl',mapped)
        strata={}
        grouping={'variant':lambda r:r['variant'],'native':lambda r:str(r['native_slot']),'canonical':lambda r:str(r['declared_CF_to_canonical']),
            'material':lambda r:r['factors']['material'],'aspect':lambda r:r['factors']['aspect'],'geometry':lambda r:r['factors']['geometry_family'],
            'camera':lambda r:json.dumps(r['factors']['camera_pair_deg']),'sourcegroup':lambda r:str(r['base_index']),
            'factor_cell':lambda r:'|'.join(r['factors'][k]for k in ('geometry_family','aspect','material'))}
        for name,fn in grouping.items():
            groups=defaultdict(list)
            for r in done:groups[fn(r)].append(r)
            strata[name]={g:dict(N=len(rr),point_target={s:sum(r['point0_target_alone_state']==s for r in rr)for s in STATES},point_combined={s:sum(r['point0_combined_state']==s for r in rr)for s in STATES},region={s:sum(r['region_state_context_only']==s for r in rr)for s in STATES},inputU=sum(not r['original_query_available']for r in rr))for g,rr in sorted(groups.items())}
        summary=dict(completed=True,PASS=True,counts=sel['counts'],own_point_target_counts=dict(Counter(r['point0_target_alone_state']for r in done)),own_point_combined_counts=dict(Counter(r['point0_combined_state']for r in done)),
            own_region_counts=dict(Counter(r['region_state_context_only']for r in done)),mapped_point_counts=dict(Counter(r['point0_combined_state']for r in mapped)),strata=strata,schemas=schemas,
            own_inputU=sum(not r['original_query_available']for r in done),mapped_inputU=sum(not r['original_query_available']for r in mapped),mapping_status=dict(Counter(r['mapping_status']for r in mapped)),
            point_E_absent=not any(r['point0_combined_state']=='E'for r in done),new_training_admission=False,physical_real_gold_claim=False,model_array_RGB_config_projection_ray_or_PnP=0)
        write(OUT/'SUMMARY.json',summary)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=e['protocol'],entry=bind(Path(entry_path)),selection=e['selection'],GT_json_whitelist=whitelist,
            counts=sel['counts'],outputs={name:bind(OUT/name)for name in ('ALL3520_OWN_SCENE_NATIVE_POINT_TEACHER_METADATA.jsonl','ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl','SUMMARY.json')},original_qL_native_U_unchanged=True,new_training_admission=False,GT_JSON_files_read=110))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),point_counts=summary['own_point_combined_counts'],mapped=summary['mapped_point_counts'])))
    except Exception as error:
        write(OUT/'FAILURE.json',dict(completed=False,PASS=False,error=str(error),traceback=traceback.format_exc(),automatic_blind_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=parser.parse_args()
    prepare()if args.prepare else run(args.entry)
