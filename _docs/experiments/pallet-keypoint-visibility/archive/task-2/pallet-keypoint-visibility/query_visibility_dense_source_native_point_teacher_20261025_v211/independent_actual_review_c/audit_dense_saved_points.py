"""Independent same110 GT JSON / generated-metadata audit. No producer import."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_dense_source_native_point_teacher_20261025_v211'
H=V/'independent_actual_review_c'
ENTRY=V/'ROOT_ACTUAL_DENSE_SOURCE_POINT_TEACHER_LOOKUP_ENTRY.json'
COMPLETE=V/'dense_source_point_teacher_v1/COMPLETE.json'
ELIG=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/ROOT_PROSPECTIVE_SOURCE_POINT_LOSS_ELIGIBILITY_AMENDMENT.json'
STATES=('D','E','S','U','UNKNOWN');ALLOWED_GT={};GT_READS=[]


def safe(path):
    p=Path(path);assert 'v112'not in str(p).lower()and p.is_absolute()and'..'not in p.parts
    assert p.suffix in('.json','.jsonl','.py','.md')
    assert not any(x.is_symlink()for x in(p,*p.parents))
    return p


def bytes_for(path,gt=False):
    p=Path(path)
    assert 'v112'not in str(p).lower() # before any filesystem call
    if p.name=='SCENES_GT_ONLY.json':assert gt and str(p)in ALLOWED_GT
    else:assert not gt
    return safe(p).read_bytes()


def binding(path):return dict(path=str(path),sha256=hashlib.sha256(bytes_for(path)).hexdigest())
def get(b,gt=False):
    if gt:assert ALLOWED_GT.get(b['path'])==b
    raw=bytes_for(b['path'],gt);assert hashlib.sha256(raw).hexdigest()==b['sha256']
    if gt:GT_READS.append(b)
    return json.loads(raw)
def lines(b):
    raw=bytes_for(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256']
    return [json.loads(x)for x in raw.splitlines()if x.strip()]
def save(path,value):
    assert Path(path).parent==H
    with Path(path).open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def finite_xy(x):return isinstance(x,list)and len(x)==2 and all(type(v)in(int,float)and math.isfinite(v)for v in x)
def state(record,key,point=False):
    value=record.get(key)
    if point:value=value[0]if isinstance(value,list)and value else None
    return value if isinstance(value,str)and value in STATES else'UNKNOWN'
def point_fields(r):
    ref=r.get('projected_reference_xy')
    return dict(point0_target_alone_state=state(r,'target_witness_states',True),point0_combined_state=state(r,'combined_witness_states',True),
        region_state_context_only=state(r,'state'),target_alone_region_state_context_only=state(r,'target_alone_region_state'),
        saved_projected_reference_xy=ref if finite_xy(ref)else None)
def locality(q,L,av,ref,hw):
    z=dict(original_input_available=av,original_input_U_preserved=not av,raw96_reference_index_xy=None,
        inside_sample_center_extent=None,inside_crop_canvas_extent=None,inside_original_frame_canvas_extent=None,
        not_coordinate_error_correction_or_pixel_visibility_proof=True)
    if ref is not None:
        h,w=hw;z['inside_original_frame_canvas_extent']=all(-.5<=v<upper-.5 for v,upper in zip(ref,(w,h)))
        if finite_xy(q)and type(L)in(int,float)and math.isfinite(L)and L>0:
            uv=[47.5+96*(p-a)/L for p,a in zip(ref,q)]
            z.update(raw96_reference_index_xy=uv,inside_sample_center_extent=all(0<=a<=95 for a in uv),inside_crop_canvas_extent=all(-.5<=a<95.5 for a in uv))
    return z


def audit():
    assert binding(ENTRY)['sha256']=='2772fa79e5c243a4390e859ed92057eb2cd76ee991f0c76fefcb759b66e8be57'
    assert binding(COMPLETE)['sha256']=='76f204d49bef1ec26cac0f8115cb9711a0a7a8ac7a8049c6c35969cfaca3fa79'
    assert binding(ELIG)['sha256']=='6cfeb4add539f43326e33b1b49398c22577379ee69f06f815be5a68d146ed47f'
    e=get(binding(ENTRY));complete=get(binding(COMPLETE));eligibility=get(binding(ELIG));scope=get(e['root_actual_scope'])
    assert e['approved_SOURCE_TRAIN_dense_saved_point_lookup']is True and e['new_training_admission']is False
    assert complete['completed']is True and complete['PASS']is True and complete['entry']==binding(ENTRY)
    assert complete['GT_json_whitelist']==e['GT_json_whitelist']and len(e['GT_json_whitelist'])==110
    assert eligibility['prospective_before_V211_actual_values']is True
    sel=get(e['selection']);original=lines(e['original_native_rows']);mapping=lines(e['input_mapping'])
    actual_own=lines(complete['outputs']['ALL3520_OWN_SCENE_NATIVE_POINT_TEACHER_METADATA.jsonl'])
    actual_mapped=lines(complete['outputs']['ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl']);summary=get(complete['outputs']['SUMMARY.json'])
    assert len(original)==len(actual_own)==3520 and len(mapping)==len(actual_mapped)==7040
    assert original==[r for s in sel['scenes']for r in s['native_rows']]
    parents=sel['parents'];assert len(parents)==110 and len(sel['scenes'])==440
    ALLOWED_GT.update({b['path']:b for b in e['GT_json_whitelist']});assert len(ALLOWED_GT)==110
    assert set(ALLOWED_GT)=={r['GT_json_binding_NOT_OPENED']['path']for r in parents}
    save(H/'AUDIT_ACCESS_ENTRY.json',dict(completed=True,root_authorized_independent_actual_audit=True,
        root_actual_entry=binding(ENTRY),root_actual_scope=e['root_actual_scope'],source_complete=binding(COMPLETE),
        prospective_eligibility=binding(ELIG),independent_code=binding(__file__),
        exact_GT110_whitelist=e['GT_json_whitelist'],RGB_NPZ_config_weights_models_or_scores_allowed=False))
    expected={};schemas=[]
    for parent in parents:
        doc=get(parent['GT_json_binding_NOT_OPENED'],True)
        assert isinstance(doc,dict)and isinstance(doc.get('rows'),list)
        scenes=[s for s in sel['scenes']if s['base_index']==parent['base_index']]
        assert len(scenes)==4 and {s['variant']for s in scenes}=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
        for s in scenes:
            matching=[r for r in doc['rows']if r.get('scene_index')==s['scene_index']and r.get('variant')==s['variant']]
            assert len(matching)==1;scene=matching[0];assert scene['base_index']==s['base_index']
            rr=scene['records_CF'];perm=scene['permCF_to_canonical']
            assert len(rr)==len(perm)==8 and all(type(x)is int for x in perm)and sorted(perm)==list(range(8))
            assert all(type(r['CF_index'])is int and type(r['canonical_index'])is int for r in rr)
            assert sorted(r['CF_index']for r in rr)==sorted(r['canonical_index']for r in rr)==list(range(8))
            lookup={r['CF_index']:r for r in rr}
            for o in s['native_rows']:
                k=o['native_slot'];r=lookup[k]
                assert r['canonical_index']==perm[k]==o['declared_CF_to_canonical']
                if 'native_canonical_index'in r:assert r['native_canonical_index']==perm[k]
                key=tuple(o['teacher_key']);assert key==(s['scene_index'],k)and key not in expected
                values=point_fields(r)
                expected[key]=dict(o,**values,UNKNOWN_schema_reason=None,
                    reference_locality=locality(o['query_xy'],o['L'],o['original_query_available'],values['saved_projected_reference_xy'],o['raw_hw']),
                    authority='SAVED_PROCEDURAL_OWN_SCENE_POINT0_NOT_REAL_NATIVE_GOLD',new_native_target_array_created=False)
            schemas.append(dict(base_index=s['base_index'],scene_index=s['scene_index'],variant=s['variant'],UNKNOWN_schema_reason=None))
    assert len(GT_READS)==len({b['path']for b in GT_READS})==110
    assert [expected[tuple(o['teacher_key'])]for o in original]==actual_own
    exp_mapped=[];eligible=[];excluded=Counter();family=Counter();different_anchor=0
    for m in mapping:
        si=m['source_image_original_global_index']//8;k=m['native_slot'];ai=m['anchor_original_global_index']//8
        assert m['component_role']=='TRAIN'and m['source_teacher_key']==[si,k]
        image=expected[si,k];anchor=expected[ai,k]
        assert m['query_xy']==anchor['query_xy']and m['L']==anchor['L']and m['original_query_available']==anchor['original_query_available']
        assert m['original_input_flags']==anchor['original_input_flags']and m['actual_image_variant']==image['variant']
        assert m['actual_image_original_occurrence_id']==image['original_occurrence_id']and m['declared_CF_to_canonical']==image['declared_CF_to_canonical']
        assert m['original_flags_ref']==[ai,k]and m['image_flags_ref']==[si,k]and m['state_name']=='NOMINAL'and m['state_index']==0
        fields={k:image[k]for k in('point0_target_alone_state','point0_combined_state','region_state_context_only','target_alone_region_state_context_only','saved_projected_reference_xy')}
        loc=locality(m['query_xy'],m['L'],m['original_query_available'],fields['saved_projected_reference_xy'],image['raw_hw'])
        row=dict(m,**fields,reference_locality_for_actual_input=loc,
            point_state_applies_to='ACTUAL_ORIGINAL_IMAGE_SCENE_NOT_ANCHOR_OR_CACHE_ID',new_native_target_array_created=False)
        exp_mapped.append(row);family[m['family']]+=1;different_anchor+=si!=ai
        predicates={'target_alone_not_D':fields['point0_target_alone_state']!='D',
            'combined_not_D_E':fields['point0_combined_state']not in('D','E'),
            'original_input_U':not m['original_query_available'],
            'invalid_original_q_L':not(finite_xy(m['query_xy'])and type(m['L'])in(int,float)and math.isfinite(m['L'])and m['L']>0),
            'missing_nonfinite_reference':not finite_xy(fields['saved_projected_reference_xy']),
            'reference_not_in_frame_canvas':loc['inside_original_frame_canvas_extent']is not True,
            'reference_not_in_actual_input_crop_canvas':loc['inside_crop_canvas_extent']is not True}
        excluded.update(k for k,v in predicates.items()if v)
        if not any(predicates.values()):eligible.append(row)
    assert exp_mapped==actual_mapped
    own_counts=Counter(r['point0_combined_state']for r in actual_own);region=Counter(r['region_state_context_only']for r in actual_own)
    assert own_counts=={'D':2957,'E':123,'S':440}and region=={'D':1639,'U':1326,'E':115,'S':440}
    assert len(eligible)==6096 and Counter(r['point0_combined_state']for r in eligible)=={'D':5853,'E':243}
    assert sum(not r['original_query_available']for r in actual_own)==32 and sum(not r['original_query_available']for r in actual_mapped)==64
    assert Counter(r['point0_combined_state']for r in actual_own if r['variant']=='NEAR_SHAM')['E']==11
    # Recompute every stored stratum directly from independently verified rows.
    functions={'variant':lambda r:r['variant'],'native':lambda r:str(r['native_slot']),'canonical':lambda r:str(r['declared_CF_to_canonical']),
        'material':lambda r:r['factors']['material'],'aspect':lambda r:r['factors']['aspect'],'geometry':lambda r:r['factors']['geometry_family'],
        'camera':lambda r:json.dumps(r['factors']['camera_pair_deg']),'sourcegroup':lambda r:str(r['base_index']),
        'factor_cell':lambda r:'|'.join(r['factors'][k]for k in('geometry_family','aspect','material'))}
    strata={};flat=[]
    for name,fn in functions.items():
        grouped=defaultdict(list)
        for r in actual_own:grouped[fn(r)].append(r)
        strata[name]={}
        for value,rows in sorted(grouped.items()):
            counts={axis:{s:sum(r[key]==s for r in rows)for s in STATES}for axis,key in [('point_target','point0_target_alone_state'),('point_combined','point0_combined_state'),('region','region_state_context_only')]}
            counts.update(N=len(rows),inputU=sum(not r['original_query_available']for r in rows));strata[name][value]=counts
            flat.append(dict(stratum=name,value=value,N=len(rows),inputU=counts['inputU'],**{'point_'+s:counts['point_combined'][s]for s in STATES},**{'region_'+s:counts['region'][s]for s in STATES}))
    assert strata==summary['strata']and schemas==summary['schemas']
    for key,computed in [('own_point_target_counts',Counter(r['point0_target_alone_state']for r in actual_own)),('own_point_combined_counts',own_counts),('own_region_counts',region),('mapped_point_counts',Counter(r['point0_combined_state']for r in actual_mapped)),('mapping_status',Counter(r['mapping_status']for r in actual_mapped))]:assert dict(computed)==summary[key]
    assert summary['new_training_admission']is False and summary['physical_real_gold_claim']is False
    assert summary['point_E_absent']is False and complete['original_qL_native_U_unchanged']is True
    with (H/'SOURCE_POINT_STRATA_INDEPENDENT.csv').open('x')as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    readout=dict(completed=True,PASS=True,GT110=110,own_native=3520,mapped_inputs=7040,own_point_combined=dict(own_counts),own_region=dict(region),
        mapped_point_combined=dict(Counter(r['point0_combined_state']for r in actual_mapped)),families=dict(family),different_image_and_anchor=different_anchor,
        own_inputU32=True,mapped_inputU64=True,NEAR_SHAM_point_E=11,
        prospective_eligibility=dict(eligible_rows=len(eligible),point_counts=dict(Counter(r['point0_combined_state']for r in eligible)),
            excluded_rows=7040-len(eligible),overlapping_exclusion_predicate_counts=dict(excluded),overlap_not_additive=True,
            source_amendment=binding(ELIG),no_admission_or_target_array_created=True),strata=strata)
    save(H/'INDEPENDENT_READOUT.json',readout)
    review=dict(completed=True,PASS=True,root_actual_entry=binding(ENTRY),root_actual_scope=e['root_actual_scope'],source_complete=binding(COMPLETE),
        code=binding(__file__),producer_code=complete['code'],producer_protocol=complete['protocol'],
        original_selection=e['selection'],original_rows=e['original_native_rows'],original_mapping=e['input_mapping'],
        actual_outputs=complete['outputs'],same_GT110_bindings=GT_READS,outputs={'readout':binding(H/'INDEPENDENT_READOUT.json')},
        independent_checks=dict(all3520_saved_fields_exact=True,all7040_saved_fields_exact=True,all9_strata_exact=True,
            all_original_q_L_native_U_role_flags_inert_bindings_preserved=True,CF_canonical_perm_exact=True,
            image_teacher_separate_from_query_anchor=True,all_frame_crop_localities_exact=True,prospective_6096_D5853_E243=True),
        counts=readout,original_GT_JSON_reads=110,RGB_NPZ_config_depth_masks_weights_models_or_protected_eval_reads=0,
        new_native_label_array_or_training_admission=False,
        limits=['Independent saved procedural GT/metadata verification, not independent render/ray/physical native gold certification.',
            'Point target/combined, region, original inputU and reference crop/frame algebraic locality are separate axes. No NEAR_SHAM->D rule.',
            '6096 rows are a prospective SOURCE-loss predicate census only; exclusions overlap and this receipt creates no target/admission.',
            'No source pixels, model, score or protected/HOLDOUT source bodies read; no producer import/rerun.'])
    save(H/'REVIEW.json',review)
    text='# V211 독립 실제 saved-GT·metadata 검산\n\nPASS. root와 동일한 TRAIN110 GT JSON만 읽어 own3520 point와 실제 image-source7040 대응, CF→canonical, 원 q/L/native/U/flags, frame/crop locality 및 9종 strata를 독립 검산했다. producer import·재실행은 없다.\n\n'
    text+='|구분|D|E|S|U|UNKNOWN|\n|---|---:|---:|---:|---:|---:|\n'
    for name,c in [('own point0',own_counts),('own region',region),('mapped point0',Counter(r['point0_combined_state']for r in actual_mapped))]:text+='|'+name+'|'+'|'.join(str(c[s])for s in STATES)+'|\n'
    text+='\nNEAR_SHAM에도 저장 point E11이 있다. variant 이름으로 D를 강제하지 않는다. 원 own inputU32와 mapped inputU64는 geometry state와 별도로 유지된다. 실제 image와 BEFORE query anchor가 다른 입력2640개도 teacher는 image scene에서, q/L/flags는 원 anchor에서 유지된다.\n\n'
    text+='V212에서 실제값 조회 전에 고정한 조건(target-alone D·combined D/E·원 available/finite qL·finite reference·원 frame 및 실제 입력 crop canvas 안)을 독립 대입하면6096행(D5853/E243)이다. 제외944행의 사유는 중첩되므로 항목 합계를 새 분모로 쓰지 않는다. 이는 source loss eligibility 수이며 새 target 배열·학습 admission이 아니다.\n\n'
    text+='[전 strata CSV](SOURCE_POINT_STRATA_INDEPENDENT.csv) · [독립 readout](INDEPENDENT_READOUT.json) · [QA receipt](REVIEW.json) · [원 source summary](../dense_source_point_teacher_v1/SUMMARY.json)\n\n'
    text+='실사 native gold·fresh blind 일반화·모델개선을 인증하지 않는다. RGB/NPZ/config/원 mask·depth/weights/forward/보호 평가 접근0이다. saved reference의 원 crop 좌표는 대수적 association이며 실제 pixel 가시성이나 keypoint 좌표 보정의 증거가 아니다.\n'
    with (H/'KO.md').open('x')as f:f.write(text)
    print(json.dumps(dict(completed=True,PASS=True,review=binding(H/'REVIEW.json'),KO=binding(H/'KO.md'),GT_reads=len(GT_READS),eligibility=readout['prospective_eligibility'])))


if __name__=='__main__':
    try:audit()
    except BaseException:
        with (H/'FAILED_NO_AUTOMATIC_RETRY.json').open('x')as f:json.dump(dict(completed=False,traceback=traceback.format_exc(),GT_files_read=len(GT_READS),automatic_retry=False),f,indent=2)
        raise
