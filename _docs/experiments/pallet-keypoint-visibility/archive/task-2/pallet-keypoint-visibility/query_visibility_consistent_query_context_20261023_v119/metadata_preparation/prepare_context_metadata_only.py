"""Original-q8 source context plan. Metadata only; no source tensor/pixel/model I/O."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import datetime
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
NAMES = {96:'query_visibility_query_anchored_cue',99:'query_visibility_diverse_expected_cue_source',
         107:'query_visibility_target_context_rgb',108:'query_visibility_target_relative_rgb_head',
         109:'query_visibility_ycb_real_surface_source',111:'query_visibility_spatial_target_query_features',
         113:'query_visibility_ycb_wood_foam_source',116:'query_visibility_same_material_foreground_source'}


def R(n):
    return P/(NAMES[n]+'_20261023_v'+str(n))


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def bind(p):
    p=Path(p)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def jsonl(p, data):
    with p.open('x') as f:
        for row in data:
            f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')


def main():
    assert not (H/'RESULT.json').exists(), 'No overwrite'
    paths=dict(definitions=R(107)/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl',
        old_contexts=R(108)/'context_feature_preparation/UNIT_CONTEXTS3652_INPUT_ONLY.jsonl',
        old_occurrences=R(96)/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl',
        synthetic_occurrences=R(99)/'matching_feature_pack/INPUT_LINEAGE2304_INPUT_ONLY.jsonl',
        C11_receipt=R(111)/'spatial_query_features/FEATURE_COMPLETE.json',
        C11_facts=R(111)/'spatial_query_features/INPUT_RGB_AND_SPATIAL_FACTS_INPUT_ONLY.jsonl',
        surface_inputs=R(109)/'query_preparation/INPUTS1000_QUERY_EMULATION_ONLY.jsonl',
        material_inputs=R(113)/'query_preparation/INPUTS400_QUERY_EMULATION_ONLY.jsonl',
        material_receipt=R(113)/'material_spatial_features/FEATURE_COMPLETE.json',
        material_lineage=R(113)/'material_spatial_features/NEW3200_INPUT_LINEAGE_ONLY.jsonl',
        material_facts=R(113)/'material_spatial_features/INPUT_RGB_QUERY_FACTS400_INPUT_ONLY.jsonl',
        samewood_inputs=R(116)/'corpus_build_v1/OCCURRENCES924_INPUT_ONLY.jsonl',
        samewood_units=R(116)/'feature_preparation_v1/UNIT152_INPUT_ONLY.jsonl',
        samewood_map=R(116)/'feature_preparation_v1/OCCURRENCE924_TO_UNIT_INPUT_ONLY.jsonl',
        samewood_receipt=R(116)/'spatial_features_v1/FEATURE_COMPLETE.json',
        current_box_code=R(111)/'current_evaluation_preparation/current_adapter.py')
    bindings={k:bind(p)for k,p in paths.items()}
    assert bindings['current_box_code']['sha256']=='1f1ed2341c7ea9edc05b5dd1b376c80e4b58f908fbdca759748fb2ad2c610d90'
    tree=ast.parse(paths['current_box_code'].read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='initial_context_box')
    ast_text=ast.unparse(fn)+'\n'
    ast_path=H/'EXACT_CURRENT_INITIAL_CONTEXT_BOX_AST.py'
    with ast_path.open('x')as f:f.write(ast_text)
    ns=dict(np=np,require=lambda ok,why:None if ok else (_ for _ in ()).throw(ValueError(why)))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(paths['current_box_code'])+'::initial_context_box_AST_only','exec'),ns)
    exact_box=ns['initial_context_box']

    def box(q8):
        # Preserve original JSON q8. Null serialization becomes NaN only for the exact8x2 math call.
        assert len(q8)==8
        a=np.array([[np.nan,np.nan]if q is None else q for q in q8],np.float64)
        out=exact_box(a)
        return None if out is None else out.tolist()

    fixture=[[-5.,-2.],[12.,9.],[-1.,-1.],None,None,None,None,None]
    assert box(fixture)==[-5.,-2.,12.,9.]
    assert box([None]*8)is None
    assert box([[2.,3.]]*8)is None
    assert box([[2.,3.],[2.,4.]]+[None]*6)is None
    checks=4
    definitions=rows(paths['definitions']);defs={d['context_id']:(i,d)for i,d in enumerate(definitions)}
    oldctx=rows(paths['old_contexts']);old=rows(paths['old_occurrences']);syn=rows(paths['synthetic_occurrences'])
    facts=rows(paths['C11_facts']);c11=read(paths['C11_receipt']);material=read(paths['material_receipt']);wood=read(paths['samewood_receipt'])
    assert len(oldctx)==3652 and len(old)==31680 and len(syn)==2304 and len(facts)==11652
    units=[];occurrences=[];groups={};geom_cache={}
    oldrep={}
    for r in old:oldrep.setdefault(r['input_unit_index'],r)
    assert set(oldrep)==set(range(1348))

    def support_fraction(b,hw):
        if b is None:return 0.0
        key=(tuple(b),tuple(hw))
        if key not in geom_cache:
            x0,y0,x1,y1=np.asarray(b,np.float64)
            xs=x0+(np.arange(224)+.5)*(x1-x0)/224
            ys=y0+(np.arange(224)+.5)*(y1-y0)/224
            Hh,W=hw
            geom_cache[key]=float(((xs>=0)&(xs<=W-1)).sum()*((ys>=0)&(ys<=Hh-1)).sum()/224**2)
        return geom_cache[key]

    def add_unit(index,family,input_id,slot,raw_split,split,q,L,hw,image,fmt,decoded,oldbox,q8,
                 q8prov,pack,pack_index,local_hash,valid_hash,canonical=None):
        nonlocal checks
        assert index==len(units)
        if q is not None:assert q8[slot]==q,(index,q,q8[slot])
        newbox=box(q8);equal=oldbox==newbox
        newfraction=support_fraction(newbox,hw);oldfraction=support_fraction(oldbox,hw)
        available=newbox is not None and newfraction>0
        canonical=index if canonical is None else canonical
        key=json.dumps(dict(decoded_RGB_sha256=decoded,raw_hw=hw,box=newbox),sort_keys=True,separators=(',',':'))
        if not decoded:key=json.dumps(dict(file=image,format=fmt,raw_hw=hw,box=newbox),sort_keys=True,separators=(',',':'))
        groupid='V119_CONTEXT_'+hashlib.sha256(key.encode()).hexdigest()[:20]
        u=dict(source_feature_unit_index=index,canonical_context_feature_unit_index=canonical,
            is_explicit_V116_BEFORE_alias=index!=canonical,source_family=family,input_id=input_id,
            logical_slot=slot,raw_split=raw_split,split=split,query_xy=q,L=L,raw_hw=hw,
            full_RGB=image,full_RGB_format=fmt,decoded_RGB_sha256=decoded,
            native_fullRGB_file_exists=Path(image['path']).is_file(),
            original_q8=q8,q8_provenance=q8prov,old_box_xyxy=oldbox,proposed_q8_box_xyxy=newbox,
            box_exact_equal=equal,box_changed=not equal,new_context_needed=not equal and available,
            context_unavailable=not available,old_context_available_geometry=oldfraction>0,
            proposed_context_available_geometry=available,
            old_context_valid224_fraction_geometry=oldfraction,
            proposed_context_valid224_fraction_geometry=newfraction,
            context_group_id=groupid,
            source_feature_ref=dict(file=pack,key='features',index=pack_index),
            original_local_RGB96_sha256=local_hash,original_valid96_sha256=valid_hash,
            local_F_C_P_features_unchanged=True,
            whole1928_exact_reuse_eligible_by_bound_metadata=equal,
            GT_reference_pose_mask_labels_in_input=False)
        if canonical!=index:
            orig=units[canonical]
            for k in('query_xy','L','raw_hw','full_RGB','decoded_RGB_sha256','old_box_xyxy','proposed_q8_box_xyxy','context_group_id','original_local_RGB96_sha256','original_valid96_sha256'):
                assert u[k]==orig[k],('Explicit BEFORE alias mismatch',index,k)
            u['new_context_needed']=False
            u['alias_target_context_update_owned_by_unit']=canonical
        units.append(u);checks+=1
        groups.setdefault(groupid,dict(context_group_id=groupid,full_RGB=image,format=fmt,
            decoded_RGB_sha256=decoded,raw_hw=hw,proposed_q8_box_xyxy=newbox,
            context_available_geometry=available,source_feature_unit_indices=[],canonical_units=[],
            any_canonical_unit_requires_new_context=False))
        gr=groups[groupid];gr['source_feature_unit_indices'].append(index)
        if canonical not in gr['canonical_units']:gr['canonical_units'].append(canonical)
        if index==canonical and u['new_context_needed']:gr['any_canonical_unit_requires_new_context']=True

    for i,c in enumerate(oldctx):
        di,d=defs[c['context_id']]
        if i<1348:
            r=oldrep[i];family=r['source'];slot=r['native_slot_logical_only'];inputid=r['input_id'];rawsplit=r['raw_split'];split=r['split']
        else:
            r=syn[i-1348];family='V99_SYNTH';slot=r['aux_corner_index'];inputid=r['base_id'];rawsplit=split=r['split']
        assert c['input_unit_index']==i and c['box_xyxy']==d['target_box_xyxy']
        add_unit(i,family,inputid,slot,rawsplit,split,c['query_xy'],c['L'],c['raw_hw'],c['full_RGB'],c['full_RGB_format'],c['decoded_RGB_sha256'],c['box_xyxy'],d['original_query8_xy'],dict(**bindings['definitions'],row_index=di,field='original_query8_xy'),c11['output'],i,facts[i]['RGB96_sha256'],facts[i]['valid96_sha256'])
    surface=rows(paths['surface_inputs'])
    assert len(surface)==1000
    for i,r in enumerate(surface):
        for k in range(8):
            u=3652+i*8+k;f=facts[u]
            assert f['input_unit_index']==u and f['input_id']==r['input_id'] and f['query_index']==k
            add_unit(u,'V109_YCB',r['input_id'],k,r['split'],r['split'],r['initial_points'][k],r['roi_side'],r['raw_hw'],r['image'],'PNG_RGB',f['decoded_RGB_sha256'],r['query_box_xyxy'],r['initial_points'],dict(**bindings['surface_inputs'],row_index=i,field='initial_points'),c11['output'],u,f['RGB96_sha256'],f['valid96_sha256'])
    minputs=rows(paths['material_inputs']);mfacts=rows(paths['material_facts']);ml=rows(paths['material_lineage'])
    assert len(minputs)==len(mfacts)==400 and len(ml)==3200
    for i,r in enumerate(minputs):
        f=mfacts[i];assert f['input_id']==r['input_id']
        for k in range(8):
            local=i*8+k;u=11652+local;t=ml[local]
            assert t['global_input_unit_index']==u and t['query_xy']==r['initial_points'][k]
            add_unit(u,'V113_YCB',r['input_id'],k,r['original_split'],r['split'],r['initial_points'][k],r['roi_side'],r['raw_hw'],r['image'],'PNG_RGB',f['decoded_RGB_sha256'],r['query_box_xyxy'],r['initial_points'],dict(**bindings['material_inputs'],row_index=i,field='initial_points'),material['output'],local,f['queries'][k]['RGB96_sha256'],f['queries'][k]['valid96_sha256'])
    wunits=rows(paths['samewood_units']);wmapping=rows(paths['samewood_map']);wi=rows(paths['samewood_inputs'])
    assert len(wunits)==152 and len(wmapping)==len(wi)==924
    for i,r in enumerate(wunits):
        di,d=defs[r['source']+':'+r['input_id']]
        alias=r['old_C11_unit_index']if i<28 else None
        add_unit(14852+i,'V116_'+r['source'],r['input_id'],r['native_slot_logical_only'],r['raw_split'],'TRAIN',r['query_xy'],r['L'],r['raw_hw'],r['fullRGB'],r['format'],r['decoded_RGB_sha256'],r['box_xyxy'],d['original_query8_xy'],dict(**bindings['definitions'],row_index=di,field='original_query8_xy'),wood['output'],i,r['rgb96_sha256'],r['valid96_sha256'],alias)
    assert len(units)==15004

    def add_occ(global_i,stored_i,inputid,slot,q,L,rawsplit,split,variant,localid,origin):
        assert global_i==len(occurrences)
        u=units[stored_i]
        assert (inputid,slot,q,L,rawsplit,split)==(u['input_id'],u['logical_slot'],u['query_xy'],u['L'],u['raw_split'],u['split'])
        occurrences.append(dict(global_occurrence_index=global_i,original_occurrence_id=localid,
            source_feature_unit_index=stored_i,canonical_context_feature_unit_index=u['canonical_context_feature_unit_index'],
            input_id=inputid,logical_slot=slot,query_xy=q,L=L,raw_split=rawsplit,split=split,
            variant=variant,source_family=u['source_family'],context_group_id=u['context_group_id'],
            context_unavailable=u['context_unavailable'],input_lineage_provenance=origin))

    for i,r in enumerate(old):
        add_occ(i,r['input_unit_index'],r['input_id'],r['native_slot_logical_only'],r['query_xy'],r['L'],r['raw_split'],r['split'],r['variant'],r['occurrence_id'],dict(**bindings['old_occurrences'],row_index=i))
    for j,r in enumerate(syn):
        add_occ(31680+j,1348+j,r['base_id'],r['aux_corner_index'],r['query_xy'],r['L'],r['split'],r['split'],r['variant'],r['image_id']+'::AUX'+str(r['aux_corner_index']),dict(**bindings['synthetic_occurrences'],row_index=j))
    for i,r in enumerate(surface):
        for k in range(8):
            j=i*8+k;add_occ(33984+j,3652+j,r['input_id'],k,r['initial_points'][k],r['roi_side'],r['split'],r['split'],'NATURAL',r['input_id']+'::q'+str(k),dict(**bindings['surface_inputs'],row_index=i,logical_slot=k))
    for j,r in enumerate(ml):
        add_occ(41984+j,11652+j,r['input_id'],r['logical_aux_slot'],r['query_xy'],r['L'],r['original_split'],r['split'],'NATURAL',r['query_id'],dict(**bindings['material_lineage'],row_index=j))
    for j,r in enumerate(wi):
        stored=14852+wmapping[j]['input_unit_index']
        assert wmapping[j]['occurrence_index']==r['occurrence_index']==j
        add_occ(45184+j,stored,r['input_id'],r['native_slot_logical_only'],r['query_xy'],r['L'],r['raw_split'],r['split'],r['variant'],r['occurrence_id'],dict(**bindings['samewood_inputs'],row_index=j))
    assert len(occurrences)==46108
    assert {r['source_feature_unit_index']for r in occurrences}==set(range(15004))
    assert sum(r['is_explicit_V116_BEFORE_alias']for r in units)==28
    for r in occurrences:
        if r['source_feature_unit_index']>=14852 and r['source_feature_unit_index']<14880:
            assert r['canonical_context_feature_unit_index']<960
    up=H/'UNIT15004_CONSISTENT_CONTEXT_INPUT_ONLY.jsonl';op=H/'OCCURRENCE46108_TO_UNIT_INPUT_ONLY.jsonl';gp=H/'NATIVE_FULLRGB_CONTEXT_GROUPS_INPUT_ONLY.jsonl'
    jsonl(up,units);jsonl(op,occurrences);jsonl(gp,list(groups.values()))
    families={}
    for family in sorted({r['source_family']for r in units}):
        rr=[r for r in units if r['source_family']==family]
        families[family]=dict(stored_units=len(rr),canonical_units=sum(not r['is_explicit_V116_BEFORE_alias']for r in rr),
            box_exact_equal=sum(r['box_exact_equal']for r in rr),box_changed=sum(r['box_changed']for r in rr),
            new_context_needed=sum(r['new_context_needed']for r in rr),context_unavailable=sum(r['context_unavailable']for r in rr),
            missing_native_files=sum(not r['native_fullRGB_file_exists']for r in rr))
    result=dict(completed=True,PASS=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Input-only source metadata/stat and exact current box-function AST only; no source pixels/NPZfeature arrays/weights/labels/current bodies/models/fit.',
        rationale='Source-input code-contract coherence; no current outcome used to select this candidate.',
        actual_feature_extraction_or_fit_authorized_by_this_metadata=False,
        counts=dict(stored_feature_units=15004,canonical_planned_units=14976,occurrences=46108,explicit_V116_BEFORE_aliases=28,
            exact_same_box_units=sum(u['box_exact_equal']for u in units),changed_box_units=sum(u['box_changed']for u in units),
            canonical_units_needing_new_context=sum(u['new_context_needed']for u in units),
            context_unavailable_units=sum(u['context_unavailable']for u in units),context_unavailable_occurrences=sum(r['context_unavailable']for r in occurrences),
            native_fullRGB_context_groups=len(groups),new_context_encoder_groups=sum(g['any_canonical_unit_requires_new_context']for g in groups.values()),
            missing_native_fullRGB_paths=sum(not u['native_fullRGB_file_exists']for u in units),
            units_with_unknown_decoded_RGB_sha256=sum(not u['decoded_RGB_sha256']for u in units)),
        by_family=families,
        boundaries=['Original q8 coordinates are copied verbatim. JSON null points are represented as NaN solely in the exact8x2 function call; finite OOB and(-1,-1)sentinel semantics unchanged.',
            'Missing/degenerate q8 or no native bilinear224 support remains explicit context-unavailable; no selected-box/GT/reference fallback. Original factual query validity remains separately owned by frozen source packets.',
            'Only T/Q descriptors, dependent cosine scalars, targetfraction and Qinside may change. F/C/P and local validity fractions remain exact old input/feature references.',
            'V11628 BEFORE feature references explicitly alias their updated original C11 baseunit; all15004 stored-unit identities and46108 occurrence identities remain.',
            'Same-native-fullRGB groups require matching decodedRGBSHA/nativeHW/newbox. File presence is stat-only; metadata SHA/pixel/format claims are inherited, not redecoded or rehashed here.',
            'Old1928 exact reuse is permitted only when oldbox==exactnewbox by metadata; no YCB/synthetic assumption. Actual feature producer must validate packet hashes/bytes separately.',
            'No head/teacher/query/fold/sampler/threshold/selector/gate or current array is changed by this plan.'],
        original_q8_function=bindings['current_box_code'],exact_AST=bind(ast_path),
        context_availability_geometry='Exact target_context_grid224 lo+(j+.5)*span/224; positive-weight bilinear-neighbour support iff x in[0,W-1],y in[0,H-1]. Metadata arithmetic, no pixels.',
        generated_math_checks=checks,outputs=dict(units=bind(up),occurrences=bind(op),native_context_groups=bind(gp)),
        source_metadata_bindings=bindings,
        source_feature_packets_bound_only_not_read=[c11['output'],material['output'],wood['output']],
        native_file_format_counts=dict(Counter(u['full_RGB_format']for u in units)),
        own_code=bind(__file__))
    (H/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (H/'SEAL.json').write_text(json.dumps(dict(completed=True,PASS=True,result=bind(H/'RESULT.json'),outputs=result['outputs'],exact_AST=bind(ast_path)),indent=2)+'\n')
    print(json.dumps(dict(result=bind(H/'RESULT.json'),counts=result['counts'],by_family=families),indent=2))


if __name__=='__main__':
    main()
