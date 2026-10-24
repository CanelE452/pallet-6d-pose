"""Independent bounded code/selected-metadata preflight; no GT bindings followed."""
import ast
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_dense_source_native_point_teacher_20261025_v211')
H=V/'independent_preflight_c';B=V/'dense_teacher_preparation_b';P=V.parent


def safe(p):
    p=Path(p)
    assert 'v112'not in str(p).lower()and p.is_absolute()and'..'not in p.parts
    assert not any(x.is_symlink()for x in(p,*p.parents))
    return p


def bind(p):return {'path':str(p),'sha256':hashlib.sha256(safe(p).read_bytes()).hexdigest()}
def read(p):return json.loads(safe(p).read_text())
def lines(p):return [json.loads(x)for x in safe(p).read_text().splitlines()if x]


def main():
    scope=V/'ROOT_PROSPECTIVE_DENSE_SOURCE_POINT_TEACHER_LOOKUP_SCOPE.json'
    assert bind(scope)['sha256']=='a77f60c1a8a41a4738c4713be5ceb39d5653628963e3616b2021207d675af9cc'
    ready=read(B/'FINAL_READY.json');assert bind(B/'FINAL_READY.json')['sha256']=='b00736adfc7383dd8198a448300ac5318184a0cd1f96d197936cff835ec6018d'
    for key in ('code','protocol','selection','whitelist','original_native_rows','input_mapping','checks','test_code','false_entry','scope'):
        assert bind(ready[key]['path'])==ready[key]
    selection=read(ready['selection']['path']);white=read(ready['whitelist']['path'])['GT_json_whitelist']
    own=lines(ready['original_native_rows']['path']);mapped=lines(ready['input_mapping']['path'])
    protocol=read(ready['protocol']['path']);false=read(ready['false_entry']['path'])
    assert ready['completed']is True and ready['PASS']is True and read(ready['checks']['path'])['PASS']is True
    for key in ('code','selection','whitelist','original_native_rows','input_mapping','checks','test_code','scope'):
        assert protocol[key]==ready[key]
    assert false['approved_SOURCE_TRAIN_dense_saved_point_lookup']is False and false['root_actual_scope']is None
    assert false['new_training_admission']is False and false['automatic_retry']is False
    assert false['GT_json_whitelist']==white and len(white)==len({b['path']for b in white})==110
    # Validate literal names only. Never stat/hash/open a GT or inert binary path.
    for b in white:
        p=Path(b['path']);assert 'v112'not in str(p).lower()and'..'not in p.parts
        assert p.name=='SCENES_GT_ONLY.json'and len(b['sha256'])==64
    parents=selection['parents'];scenes=selection['scenes'];bases={r['base_index']for r in parents}
    assert len(parents)==len(bases)==110 and len(scenes)==440
    assert {r['GT_json_binding_NOT_OPENED']['path']for r in parents}=={r['path']for r in white}
    assert own==[r for s in scenes for r in s['native_rows']]
    assert len(own)==3520 and len(mapped)==len({r['row_id']for r in mapped})==7040
    assert Counter(r['variant']for r in own)==dict.fromkeys(('BEFORE','COVER','NEAR_SHAM','FAR_SHAM'),880)
    assert Counter(r['native_slot']for r in own)==dict.fromkeys(range(8),440)
    ownmap={tuple(r['teacher_key']):r for r in own};assert len(ownmap)==3520
    byscene={s['scene_index']:s for s in scenes};assert len(byscene)==440
    for base in bases:
        ss=[s for s in scenes if s['base_index']==base]
        assert len(ss)==4 and {s['variant']for s in ss}=={'BEFORE','COVER','NEAR_SHAM','FAR_SHAM'}
    for r in own:
        si,k=r['teacher_key'];assert r['role']=='TRAIN'and r['new_training_admission']is False
        assert si==r['scene_index']and k==r['native_slot']and r['original_global_index']==si*8+k
        assert r['original_query_available']==r['original_input_flags']['query_available']
        assert r['base_index']in bases and r['variant']==byscene[si]['variant']
    family=Counter();different_anchor=0
    for r in mapped:
        k=r['native_slot'];si=r['source_image_original_global_index']//8;ai=r['anchor_original_global_index']//8
        assert r['source_image_original_global_index']%8==r['anchor_original_global_index']%8==k
        assert r['component_role']=='TRAIN'and r['split']=='TRAIN'and r['new_training_admission']is False
        assert r['mapping_status']=='EXACT_ORIGINAL_IMAGE_OWN_SCENE'and r['source_teacher_key']==[si,k]
        image=ownmap[si,k];anchor=ownmap[ai,k]
        assert r['scene_index']==r['actual_image_scene_index']==si and r['actual_image_variant']==image['variant']
        assert r['actual_image_original_global_index']==image['original_global_index']
        assert r['actual_image_original_occurrence_id']==r['source_image_original_query_id']==image['original_occurrence_id']
        assert r['actual_image_source_binding_inert']==image['source_RGB_binding_inert']
        assert r['query_xy']==anchor['query_xy']and r['L']==anchor['L']
        assert r['original_query_available']==anchor['original_query_available']and r['original_input_flags']==anchor['original_input_flags']
        assert r['anchor_query_original_occurrence_id']==r['anchor_original_query_id']==anchor['original_occurrence_id']
        assert r['anchor_scene_index']==ai and r['original_flags_ref']==[ai,k]and r['image_flags_ref']==[si,k]
        assert r['declared_CF_to_canonical']==image['declared_CF_to_canonical']
        assert r['state_name']=='NOMINAL'and r['state_index']==0 and r['dx']==r['dy']==0 and r['L_scale']==1
        if r['family']=='SELF_SCENE15':assert ai==si
        else:assert r['family']=='BEFORE_ANCHOR4'and anchor['variant']=='BEFORE'
        different_anchor+=si!=ai;family[r['family']]+=1
    assert family=={'SELF_SCENE15':3520,'BEFORE_ANCHOR4':3520}and different_anchor==2640
    assert Counter(tuple(r['source_teacher_key'])for r in mapped)==dict.fromkeys(ownmap,2)
    assert sum(not r['original_query_available']for r in own)==32
    assert sum(not r['original_query_available']for r in mapped)==64
    # Execute only five pure functions and guarded I/O entry with mocked access.
    source=safe(ready['code']['path']).read_text();tree=ast.parse(source)
    names={'map_input','select_scene','extract','locality','raw','run'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    ns=dict(Path=Path,P=P,math=math,os=os,stat=stat,STATES=('D','E','S','U','UNKNOWN'),GT_ALLOWED={})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_B_selected_functions','exec'),ns)
    checks=[]
    def ok(name,value):assert value,name;checks.append(name)
    def bad(name,f):
        try:f()
        except (AssertionError,KeyError,TypeError,ValueError):checks.append(name);return
        raise AssertionError(name)
    perm=[(k+3)%8 for k in range(8)]
    records=[dict(CF_index=k,canonical_index=perm[k],combined_witness_states=['E'],target_witness_states=['D'],state='U',projected_reference_xy=[20.,30.])for k in range(8)]
    original=dict(scene_index=1,variant='COVER',base_index=0,native_rows=[dict(native_slot=k,declared_CF_to_canonical=perm[k])for k in range(8)])
    scene=dict(scene_index=1,variant='COVER',base_index=0,permCF_to_canonical=perm,records_CF=records,nearest_pixel_diagnostic_CF=copy.deepcopy(records))
    doc=dict(rows=[scene]);rr,reason=ns['select_scene'](doc,original)
    ok('explicit_CF_point_not_region_or_nearest',reason is None and ns['extract'](rr[2])['point0_combined_state']=='E'and ns['extract'](rr[2])['region_state_context_only']=='U')
    missing=copy.deepcopy(doc);del missing['rows'][0]['records_CF']
    ok('missing_CF_no_nearest_fallback',ns['select_scene'](missing,original)[0]is None)
    drift=copy.deepcopy(doc);drift['rows'][0]['permCF_to_canonical']=list(range(8))
    bad('perm_drift_rejected',lambda:ns['select_scene'](drift,original))
    bad('duplicate_own_scene_rejected',lambda:ns['select_scene']({'rows':[scene,scene]},original))
    ok('missing_point_not_region_copy',ns['extract']({'state':'D'})['point0_combined_state']=='UNKNOWN')
    loc=ns['locality']([0.,0.],96.,False,[48.,0.],[100,100])
    ok('inputU_preserved_canvas_boundary_separate',loc['original_input_U_preserved']and loc['raw96_reference_index_xy']==[95.5,47.5]and not loc['inside_crop_canvas_extent'])
    smap={}
    for si,var in [(0,'BEFORE'),(1,'COVER')]:
        mm=[dict(original_global_index=si*8+k,original_occurrence_id=f's{si}k{k}',original_query_xy=[k+si+10.,20.],original_L=96.,original_input_flags={'query_available':True},declared_CF_to_canonical=k)for k in range(8)]
        smap[si]=dict(scene_index=si,base_index=0,variant=var,original_input_metadata8=mm,source_RGB_binding={'inert':True})
    a=smap[0]['original_input_metadata8'][2];im=smap[1]['original_input_metadata8'][2]
    row=dict(native_slot=2,original_global_index=999999,source_image_original_global_index=10,anchor_original_global_index=2,scene_index=1,base_index=0,variant='COVER',anchor_scene_index=0,
        query_xy=a['original_query_xy'],L=96.,original_query_available=True,source_image_original_query_id=im['original_occurrence_id'],anchor_original_query_id=a['original_occurrence_id'],original_flags_ref=[0,2],image_flags_ref=[1,2],family='BEFORE_ANCHOR4')
    before=copy.deepcopy(row);m=ns['map_input'](row,smap)
    ok('image_teacher_separate_from_anchor_and_cache_id',m['source_teacher_key']==[1,2]and m['query_xy']==a['original_query_xy']and row==before)
    del row['source_image_original_global_index'];ok('missing_image_identity_UNKNOWN',ns['map_input'](row,smap)['source_teacher_key']is None)
    bad('V112_rejected_before_open',lambda:ns['raw'](P/'V112/never.json'))
    bad('binary_rejected_before_open',lambda:ns['raw'](P/'never.npz'))
    bad('GT_rejected_before_actual_whitelist',lambda:ns['raw'](P/'never/SCENES_GT_ONLY.json'))
    ns.update(bind=lambda _:None,get=lambda _:dict(approved_SOURCE_TRAIN_dense_saved_point_lookup=False,new_training_admission=False))
    bad('false_entry_rejected_before_GT',lambda:ns['run'](Path('/never_entry.json')))
    assert len(checks)==12 and not ns['GT_ALLOWED']
    review=dict(completed=True,PASS=True,scope=bind(scope),code=ready['code'],protocol=ready['protocol'],preparation=bind(B/'FINAL_READY.json'),
        independent_checker=bind(__file__),selected_metadata_bindings={k:ready[k]for k in('selection','whitelist','original_native_rows','input_mapping','checks','test_code','false_entry')},
        counts=dict(TRAIN_parents=110,own_scenes=440,own_native_rows=3520,C61_nominal_eligible_rows=7040,own_inputU=32,mapped_inputU=64,
            families=dict(family),input_rows_with_image_scene_different_from_BEFORE_anchor=different_anchor,each_own_point_mapped_twice=True),
        critical_findings=['Exact image-source scene teacher key; original BEFORE anchor q/L/flags copied separately. No cache-id or anchor state alias.',
            'Explicit top rows + exact ownscene variant/index/base + records_CF only; integer CF/canonical/perm agreement, no nearest/canonical/region fallback.',
            'Saved point0 D/E/S/U/UNKNOWN distinct from region and algebraic frame/crop locality. No array/new-label/training admission.',
            'Root approved false template and exact110 whitelist gate before GT; lexical V112 and O_NOFOLLOW paths before source open.'],
        invented_checks=checks,GT_source_payload_stat_hash_header_body_reads=0,RGB_NPZ_weights_models_protected_score_access=0,
        limits=['This is code/selected-metadata preflight, not actual GT state verification. Actual GT110 lookup requires separate root entry.',
            'Shared source metadata containers were used by B preparation; C recomputed selected3520/7040 identity and arithmetic, not a full 65536-role rescan.',
            'Saved procedural point authority is not physical-real/human native gold. Source future admission remains root decision.'],new_labels_or_training_admission=False)
    with (H/'REVIEW.json').open('x')as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
    ko='''# V211 독립 코드·선택 metadata preflight\n\nPASS. 고정 TRAIN110 parent ×4 scene ×8 native =3520 own point metadata와 C61 eligible7040 입력 대응을 독립 재계산했다. own inputU32 / mapped inputU64, SELF NOMINAL3520 + BEFORE_ANCHOR3520, 각 own teacher는 두 입력에 대응한다. 2640 입력은 실제 image scene과 BEFORE query anchor가 다르며, teacher key는 image 원 global ID//8와 native이고 q/L/flags는 원 anchor에서 별도 보존된다. 원 cache index로 point 상태를 붙이지 않는다.\n\nReader는 saved top `rows`에서 scene_index·variant·base를 유일하게 고르고 `records_CF`와 원 CF→canonical perm을 검사한다. point0의 target/combined 상태와 region, 원 inputU, frame/crop algebraic locality를 분리한다. 누락 point는 UNKNOWN이며 region/nearest/canonical list로 대체하지 않는다. 실제 GT 조회 전 root 승인·정확 whitelist·SHA가 필요하다.\n\n독립 가상12검사 PASS. 실제 GT/source payload의 stat/hash/header/body·RGB/NPZ·가중치·모델·보호 점수 접근은 모두0이다. C는 B의 선택3520/7040 metadata 정합을 확인했으며 원65536 역할 전수집계나 실제 GT 상태를 재검증했다고 주장하지 않는다. 새 label/학습 admission은 없고, 저장 procedural SOURCE point 권위는 실사 native gold가 아니다.\n'''
    with (H/'KO.md').open('x')as f:f.write(ko)
    print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),KO=bind(H/'KO.md'),checks=len(checks))))


if __name__=='__main__':main()
