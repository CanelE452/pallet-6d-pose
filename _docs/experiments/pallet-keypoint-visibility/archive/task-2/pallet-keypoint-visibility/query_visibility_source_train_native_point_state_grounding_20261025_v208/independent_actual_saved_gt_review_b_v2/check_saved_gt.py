"""Independent explicit BEFORE/records_CF scalar check; no producer import/3D math."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,os,stat,traceback

H=Path(__file__).parent;V=H.parent;P=V.parent
ENTRY=V/'ROOT_ACTUAL_SAME_TRAIN_GT_RECORDS_CF_V2_ENTRY.json'
ENTRY_SHA='6f318323747986f4751652cc78edccfb3d3f31d8d28831e35aed8e2d823f128b'
COMPLETE=V/'source_train_native_point_grounding_v2/COMPLETE.json'
COMPLETE_SHA='5b18efe0a89d31297d5bceff301e674d85ab2179199f506905733c16c650a6c6'
STATES={'D','E','S','U'}

def read(path):
    p=Path(path)
    assert p.is_absolute()and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix in {'.json','.jsonl','.py'},'No binary/header access'
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    with os.fdopen(final,'rb')as stream:return stream.read()

def binding(path):return dict(path=str(path),sha256=hashlib.sha256(read(path)).hexdigest())
def checked(b):
    raw=read(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256'];return json.loads(raw)
def lines(b):
    raw=read(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256']
    return [json.loads(x)for x in raw.splitlines()if x.strip()]
def put(name,obj):
    assert name in {'REVIEW.json','FAILURE.json'}
    with (H/name).open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def point(record,name):
    value=record.get(name)
    return value[0]if isinstance(value,list)and len(value)>0 and value[0]in STATES else'UNKNOWN'
def state(record,name):
    value=record.get(name);return value if isinstance(value,str)and value in STATES else'UNKNOWN'
def reference(record):
    value=record.get('projected_reference_xy')
    return list(map(float,value))if isinstance(value,list)and len(value)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in value)else None

def run(entry_path):
    assert Path(entry_path)==ENTRY and binding(ENTRY)['sha256']==ENTRY_SHA
    e=checked(dict(path=str(ENTRY),sha256=ENTRY_SHA));assert e['approved_selected_TRAIN_saved_GT_JSON_lookup']is True
    complete=checked(dict(path=str(COMPLETE),sha256=COMPLETE_SHA));assert complete['completed']and complete['PASS']
    assert complete['entry']==binding(ENTRY)and complete['selection']==e['selection']and complete['GT_json_whitelist']==e['GT_json_whitelist']
    scope=checked(e['scope']);selection=checked(e['selection'])
    assert selection['completed']and selection['PASS']and selection['GT_json_whitelist_NOT_OPENED']==e['GT_json_whitelist']
    whitelist={b['path']:b for b in e['GT_json_whitelist']};assert len(whitelist)==41
    parents=selection['parents'];original=selection['original_BEFORE_native8'];controls=selection['physical_control_rows']
    assert len(parents)==41 and len(original)==328 and len(controls)==1968
    assert all(r['role']=='TRAIN'for r in parents+original+controls)
    saved_rows=lines(complete['outputs']['ALL_ORIGINAL_BEFORE_NATIVE_POINT_STATE_AND_LOCALITY.jsonl'])
    saved_map=lines(complete['outputs']['ALL_CONSTRUCTED_TRAIN_CONTROL_ROWS_TO_BEFORE_PARENT.jsonl'])
    summary=checked(complete['outputs']['SUMMARY.json'])
    row_by={(r['base_index'],r['native_slot']):r for r in saved_rows}
    orig_by={(r['base_index'],r['native_slot']):r for r in original}
    assert len(row_by)==len(orig_by)==len(saved_rows)==328 and set(row_by)==set(orig_by)
    counts={k:Counter()for k in ('point0_target_alone_states','point0_combined_states','region_states','target_alone_region_states')}
    locality=Counter();point_region=Counter();native=defaultdict(Counter);canonical=defaultdict(Counter)
    max_index_difference=0.;sourceGT=[];used=set();displacements=[]
    for parent in parents:
        b=parent['GT_json_binding_NOT_OPENED'];assert whitelist.get(b['path'])==b and Path(b['path']).name=='SCENES_GT_ONLY.json'
        assert b['path']not in used;used.add(b['path']);doc=checked(b);sourceGT.append(b)
        # Explicit authoritative member only. Do not inspect old foreign or canonical/nearest alternatives.
        assert type(doc)is dict and isinstance(doc.get('rows'),list),'Explicit top rows scene-list'
        matches=[item for item in doc['rows']if type(item)is dict and item.get('variant')=='BEFORE'and item.get('scene_index')==parent['original_scene_index']]
        assert len(matches)==1,'Unique original BEFORE by unchanged scene_index'
        scene=matches[0]
        assert scene['scene_index']==parent['original_scene_index']and scene['base_index']==parent['base_index']
        records=scene['records_CF'];assert len(records)==8
        assert all(type(r['CF_index'])is int and type(r['canonical_index'])is int for r in records)
        declared={r['CF_index']:r for r in records};assert set(declared)==set(range(8))
        assert sorted(r['canonical_index']for r in records)==list(range(8))
        perm=scene['permCF_to_canonical'];assert len(perm)==8 and sorted(perm)==list(range(8))
        for k in range(8):
            key=(parent['base_index'],k);o=orig_by[key];s=row_by[key];r=declared[k]
            assert o['GT_json_binding_NOT_OPENED']==b and o['native_slot']==k and o['base_index']==parent['base_index']
            assert r['canonical_index']==r['native_canonical_index']==perm[k]==o['declared_CF_to_canonical']
            assert all(s[name]==value for name,value in o.items()),'All original q/L/U/native/bindings unchanged'
            expected=dict(point0_target_alone_state=point(r,'target_witness_states'),point0_combined_state=point(r,'combined_witness_states'),
                region_state=state(r,'state'),target_alone_region_state=state(r,'target_alone_region_state'),saved_anchor_projected_reference_xy=reference(r))
            assert all(s[name]==value for name,value in expected.items()),key
            assert s['source_variant']=='ORIGINAL_BEFORE_ONLY'and s['canonical_index']==perm[k]
            assert s['new_visibility_target_created']is False and s['training_admission']is False
            assert type(o['original_query_available'])is bool
            ref=expected['saved_anchor_projected_reference_xy'];saved_local=s['original_crop_reference_locality']
            if ref is not None:
                q=o['query_xy'];L=o['L'];assert all(math.isfinite(x)for x in q)and math.isfinite(L)and L>0
                # Independent arithmetic grouping, same F64 mathematical inverse; no projection/ray.
                u=[(ref[i]-q[i])*(96.0/L)+47.5 for i in range(2)]
                for actual,want in zip(saved_local['raw96_reference_index_xy'],u):
                    delta=abs(actual-want);max_index_difference=max(max_index_difference,delta);assert delta<1e-10
                in_centers=all(0<=x<=95 for x in u);in_canvas=all(-.5<=x<95.5 for x in u)
                assert saved_local['inside_sample_center_extent']==in_centers and saved_local['inside_canvas_extent']==in_canvas
                locality['saved_reference_rows']+=1;locality['inside_sample_center_extent']+=int(in_centers);locality['inside_canvas_extent']+=int(in_canvas)
                displacements.append(math.hypot(u[0]-47.5,u[1]-47.5))
            else:assert saved_local['raw96_reference_index_xy']is None
            assert saved_local['original_input_available']==o['original_query_available']
            if ref is not None:assert saved_local['original_input_U_preserved']==(not o['original_query_available'])
            for source,target in [('point0_target_alone_state','point0_target_alone_states'),('point0_combined_state','point0_combined_states'),('region_state','region_states'),('target_alone_region_state','target_alone_region_states')]:counts[target][expected[source]]+=1
            point_region[(expected['point0_combined_state'],expected['region_state'])]+=1
            native[k][expected['point0_combined_state']]+=1;canonical[perm[k]][expected['point0_combined_state']]+=1
    assert used==set(whitelist)
    for name in ('point0_target_alone_states','point0_combined_states','region_states'):assert dict(counts[name])==summary[name]
    assert summary['selection_counts']==selection['counts']and summary['original_inputU']==sum(not o['original_query_available']for o in original)
    schemas={r['base_index']:r for r in summary['schemas']};assert len(schemas)==len(parents)
    for parent in parents:
        s=schemas[parent['base_index']];assert s['saved_CF_records_found']is True and s['UNKNOWN_schema_reason']is None and s['original_source_GT']==parent['GT_json_binding_NOT_OPENED']
    # Key-based mapping comparison; order, IDs, all original fields and current variant statuses retained.
    own_keys={tuple(r['control_join_key']):r for r in controls};saved_keys={tuple(r['control_join_key']):r for r in saved_map}
    assert len(own_keys)==len(saved_keys)==len(saved_map)==1968 and set(own_keys)==set(saved_keys)
    assert [r['control_join_key']for r in controls]==[r['control_join_key']for r in saved_map]
    variants=Counter()
    for key,o in own_keys.items():
        s=saved_keys[key];assert all(s[name]==value for name,value in o.items())
        assert key==('SYNTH_FINITE_END',o['attempt_id'],o['variant'],o['native_slot'])
        assert s['parent_lookup_key']==[o['source_base_index'],o['native_slot']]
        assert s['saved_native_states_apply_to']=='ORIGINAL_BEFORE_PARENT_ONLY'
        assert s['changed_V189_COVER_SHAM_point_states']=='UNKNOWN_NOT_DERIVED_OR_ALIASED_FROM_FINITE_END'
        assert not any(name in s for name in ('point0_combined_state','point0_target_alone_state','region_state','native_visibility_target'))
        parent=orig_by[(o['source_base_index'],o['native_slot'])]
        for name in ('query_xy','L','original_query_available','original_input_flags','original_occurrence_id','original_global_index'):assert o[name]==parent[name]
        assert o['variant']in ('BEFORE','COVER','SHAM');variants[o['variant']]+=1
    assert complete['point_state_rows']==len(original)and complete['physical_TRAIN_control_rows_mapped']==len(controls)and complete['GT_json_files_read']==len(sourceGT)
    assert complete['new_point_or_region_labels_created']is False and complete['training_admission']is False
    # Original V1 remains independently sealed and unmodified; do not rewrite its conservative UNKNOWN.
    v1=checked(scope['v1_actual_complete']);v1_summary=checked(v1['outputs']['SUMMARY.json'])
    assert v1['completed']and v1['PASS']and v1_summary['no_new_labels_or_admission']is True
    result=dict(completed=True,PASS=True,code=binding(Path(__file__)),root_entry=binding(ENTRY),complete=binding(COMPLETE),selection=e['selection'],
        original_source_GT_whitelist=sourceGT,GT_JSON_files_checked=len(sourceGT),original_BEFORE_native_rows=len(original),control_rows=len(controls),
        original_inputU=summary['original_inputU'],states={k:dict(v)for k,v in counts.items()},locality=dict(locality),
        maximum_scalar_crop_index_difference=max_index_difference,reference_to_original_query_crop96_distance_min=min(displacements),
        reference_to_original_query_crop96_distance_max=max(displacements),point_combined_vs_region=[dict(point0=p,region=r,N=n)for(p,r),n in sorted(point_region.items())],
        by_native_point0={str(k):dict(v)for k,v in native.items()},by_canonical_point0={str(k):dict(v)for k,v in canonical.items()},
        mapping_variants=dict(variants),all1968_controlkey_original_fields_and_order_exact=True,all328_native_permutation_qL_U_reference_region_exact=True,
        changed_controls_point_states_remain_UNKNOWN=True,v1_preserved_complete=scope['v1_actual_complete'],
        v1_preserved_point0_target_counts=v1_summary['point0_target_alone_states'],v1_preserved_point0_combined_counts=v1_summary['point0_combined_states'],
        producer_or_evaluator_imports=0,NPZ_NPY_RGB_GT_reprojection_rays_models_fit_or_eval_calls=0,new_labels_or_admission=False,
        authority='Saved procedural source BEFORE point0 and separate region evidence. Locality is metadata F64 index algebra, not physical nativegold, observed RGB visibility, coordinate correction or transfer proof.')
    put('REVIEW.json',result)
    with (H/'KO.md').open('x',encoding='utf-8')as f:
        f.write('승인된 41 TRAIN 원 GT JSON의 BEFORE/records_CF만 독립 추출하여 328 native의 point0 target/combined·region·reference와 원 q/L/U를 대조했다. 1968 control-key 매핑과 원 순서·메타데이터가 일치하며, 새 COVER/SHAM point 상태는 UNKNOWN으로 유지된다.\n\n')
        f.write('원 점 상태와 복합 region teacher는 다른 권한이다. locality는 저장 reference와 원 q/L 사이의 crop96 좌표식 검산이며 물리 native gold·RGB 가시성·좌표오류나 재중심화 증거가 아니다. v1 보수적 UNKNOWN 결과는 수정하지 않았다. 3D 투영·ray·이미지/배열·모델·학습·실사평가 접근은 0이다.\n')
    print(json.dumps(dict(completed=True,PASS=True,review=binding(H/'REVIEW.json'),GT_files=len(sourceGT),rows=len(original),mappings=len(controls),states=result['states'],locality=result['locality'],max_error=max_index_difference)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.entry)
    except Exception as error:
        put('FAILURE.json',dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
