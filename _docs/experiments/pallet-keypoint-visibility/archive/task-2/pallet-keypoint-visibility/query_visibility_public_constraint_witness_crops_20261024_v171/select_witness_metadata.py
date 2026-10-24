"""All critical tie rows, metadata only after fixed V171 design. Never opens pixels."""
from pathlib import Path
import hashlib,json
import numpy as np

V=Path(__file__).resolve().parent;P=V.parent
V170=P/'query_visibility_source_threshold_feasibility_20261024_v170'
RULES=V/'ROOT_SOURCE_ONLY_DESIGN_BEFORE_WITNESS_SCORE_GT_OR_RGB.json'
CONDITIONS=('NATIVE::VALIDATION::foreign_family::SAME_TARGET_MATERIAL::D','NATIVE::VALIDATION::foreign_family::SAME_TARGET_MATERIAL::E')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'];return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def present_only(b):
    # Deliberately neither opens nor hashes original RGB/crop NPZ bodies.
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    return dict(binding_inherited=b,path_exists=p.is_file(),body_opened=False,body_SHA_reverified=False,
                status='PATH_PRESENT_INHERITED_SHA_NOT_YET_REVERIFIED' if p.is_file() else 'UNKNOWN_PATH_ABSENT')
def run():
    design=read(RULES);assert design['selection_cap'] is None and design['no_replacement']
    feasibility=read(V170/'FEASIBILITY.json');inputs=read(V170/'INPUT_BINDINGS_BEFORE_PUBLIC_BODIES.json')
    assert feasibility['PASS'];parent=read(checked(inputs['parent_report_protocol']))
    # This GT is public synthetic metadata; no mixed/image NPZ is read.
    gt_binding=parent['native'][0]['GT'];all_gt=lines(checked(gt_binding));gt=all_gt[0::2]
    assert len(gt)==6144 and all(x['convention']=='CF_PRIMARY_DECLARED' for x in gt)
    feature_seal_binding=parent['native'][0]['feature_seal'];feature_seal=read(checked(feature_seal_binding))
    facts_binding=feature_seal['input_crop_facts'];facts=read(checked(facts_binding))['rows']
    queries_binding=feature_seal['initializer_queries'];queries=lines(checked(queries_binding))
    assert len(facts)==len(queries)==6144
    bindings=[bind(RULES),bind(V170/'FEASIBILITY.json'),bind(V170/'INPUT_BINDINGS_BEFORE_PUBLIC_BODIES.json'),inputs['parent_report_protocol'],gt_binding,feature_seal_binding,facts_binding,queries_binding]
    memberships=[];dedup={};packets={};census=[]
    for model,fm in zip(inputs['models'],feasibility['models']):
        assert model['arm']==fm['arm'];b=model['outputs']['OLD6144'];bindings.append(b)
        with np.load(checked(b),allow_pickle=False) as z:
            wanted=('raw_three_logits','query_available','global_index','occurrence_id','scene_index','base_index','native_slot','query_xy','L','source_occurrence_index','corpus')
            packet={k:z[k].copy() for k in wanted}
        packets[model['arm']]=packet
        assert np.array_equal(packet['global_index'],np.arange(6144))
        score=packet['raw_three_logits'][:,0];assert score.dtype==np.float32
        for name in CONDITIONS:
            c=next(x for x in fm['conditions'] if x['condition']==name);state=c['state'];assert not c['empty']
            mask=np.array([r['split']=='VALIDATION' and r['factors']['foreign_family']=='SAME_TARGET_MATERIAL' and r['teacher_state']==state for r in gt],bool)
            assert int(mask.sum())==c['N'] and c['N']==(190 if state=='D' else 16)
            valid=mask&packet['query_available'];values=np.sort(score[valid].astype(np.float64));m=c['required_correct']
            assert m==(189 if state=='D' else 13)
            bound=float(values[m-1] if state=='D' else values[-m]);assert bound==c['bound']
            ids=np.flatnonzero(valid&(score.astype(np.float64)==bound)).tolist()
            census.append(dict(arm=model['arm'],condition=name,score_bound=bound,rank=m,direction='lowest' if state=='D' else 'highest',group_N=c['N'],valid_support=int(valid.sum()),all_tie_global_indices=ids,tie_count=len(ids),status='ALL_TIES_RETAINED' if ids else 'UNKNOWN_NO_MATCHING_TIE'))
            for i in ids:
                r=gt[i];f=facts[i];q=queries[i]
                assert f['occurrence_index']==i and f['occurrence_id']==q['query_id']==r['query_id']==str(packet['occurrence_id'][i])
                assert r['native_slot']==f['native_slot']==q['native_slot']==int(packet['native_slot'][i])
                assert r['query_xy']==f['query_xy']==q['query_xy']==packet['query_xy'][i].tolist()
                assert r['roi_side']==f['L']==q['roi_side']==float(packet['L'][i])
                assert f['raw_RGB_PNG']==r['raw_RGB_PNG'] and f['query_available']==bool(packet['query_available'][i])
                membership=dict(arm=model['arm'],condition=name,state=state,global_index=i,raw_E_logit_FP32_as_float64=float(score[i]),bound_float64=bound,rank=m,group_N=c['N'])
                memberships.append(membership)
                if i not in dedup:
                    dedup[i]=dict(global_index=i,occurrence_id=f['occurrence_id'],query_id=r['query_id'],native_slot=r['native_slot'],scene_index=r['scene_index'],base_index=r['base_index'],source_id=r['id'],base_id=r['base_id'],split=r['split'],variant=r['variant'],query_xy=f['query_xy'],L=f['L'],raw_hw=f['raw_hw'],query_available=f['query_available'],raw_RGB_PNG_binding_inherited=f['raw_RGB_PNG'],decoded_RGB_sha256_inherited=f['decoded_RGB_sha256'],rgb96_crop_binding_inherited=f['rgb96_chunk'],rgb96_row=f['rgb96_row'],rgb96_array_sha256_inherited=f['rgb96_sha256'],valid96_array_sha256_inherited=f['valid96_sha256'],original_context_box_xyxy=f['context_box_xyxy'],original_initial_points=f['initial_points'],source_GT_metadata=r,original_crop_facts_metadata=f,original_initializer_query_metadata=q,bound_memberships=[])
                dedup[i]['bound_memberships'].append(membership)
    witnesses=[];readiness=[]
    for i,row in sorted(dedup.items()):
        row['both_fixed_head_raw_E_logits']={arm:float(packet['raw_three_logits'][i,0]) for arm,packet in packets.items()}
        witnesses.append(row)
        readiness.append(dict(global_index=i,query_id=row['query_id'],raw_RGB=present_only(row['raw_RGB_PNG_binding_inherited']),actual_RGB96_crop_chunk=present_only(row['rgb96_crop_binding_inherited']),rgb96_row=row['rgb96_row'],rgb96_array_sha256_inherited=row['rgb96_array_sha256_inherited'],valid96_array_sha256_inherited=row['valid96_array_sha256_inherited'],ready_for_root_actual_RGB_and_crop_integrity_checks=True,original_qL_unchanged=True,
            next_actual_checks='Reverify inherited PNG/chunk SHA; decode PNG and compare inherited decoded RGB SHA; load chunk rgb96/valid96/crop_constructed/global_index and compare selected row/array SHA. Display original q and saved crop; nominal reference is a separate uncertain diagnostic, never model recentering.'))
    for b in bindings:checked(b)
    out=V/'WITNESS_SELECTION_BEFORE_RGB.json'
    write(out,dict(completed=True,PASS=True,phase='V171',design=bind(RULES),code=bind(__file__),bindings=bindings,selection_census=census,all_bound_memberships=memberships,unique_witnesses=witnesses,total_memberships=len(memberships),deduplicated_rows=len(witnesses),selection_cap=None,no_substitution=True,unknown_selections=sum(x['tie_count']==0 for x in census),RGB_crop_feature_weight_bodies_read=0,current_V169_private_V112_access=0,threshold_or_model_selection=False,label_qL_or_reference_correction=False,training_admission=False,success_or_solution_claimed=False))
    ready=V/'INPUT_PATH_READINESS.json'
    write(ready,dict(completed=True,PASS=True,phase='V171',selection=bind(out),unique_rows=len(witnesses),rows=readiness,all_original_RGB_paths_present=all(r['raw_RGB']['path_exists'] for r in readiness),all_actual_crop_paths_present=all(r['actual_RGB96_crop_chunk']['path_exists'] for r in readiness),asset_hashes_inherited_only=True,asset_bodies_opened_or_hashed=0,diagnostic_only_not_label_gold=True,next_actual_execution_owned_by_root=True))
    print(json.dumps(dict(selection=bind(out),readiness=bind(ready),memberships=len(memberships),unique_rows=len(witnesses)),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
