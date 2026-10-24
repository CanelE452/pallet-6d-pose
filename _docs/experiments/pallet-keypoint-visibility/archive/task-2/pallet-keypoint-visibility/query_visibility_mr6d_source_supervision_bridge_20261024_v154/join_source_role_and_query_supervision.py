"""Join independently reviewed public roles and original query correspondence.

No nearest-GT matching, feature/weight bodies, or learned visibility scores.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math

V=Path(__file__).resolve().parent
S=V.parent/'query_visibility_mr6d_full_public_source_20261024_v153'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def check(b):
    assert '_v112' not in b['path'].lower()
    assert sha(b['path'])==b['sha256']
    return Path(b['path'])
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_lines(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def key(r):return r['frame_index'],r['gt_instance_index'],r['native_slot']

def main():
    policy=read(V/'ROOT_POLICY_BEFORE_SOURCE_BRIDGE_REVIEW.json')
    assert policy['goal_status']=='active_unsolved' and policy['fixed_source_scope']['frames']==230
    basefile=S/'native_reference_preparation/provided_reference_v1/ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'
    base=lines(basefile);assert len(base)==2424 and len({key(r) for r in base})==2424
    source_before=bind(basefile)
    seals=[(V/'development_native_review_scene0_2_agent/SOURCE_ROLE_REVIEW_COMPLETE.json','roles_packet'),
           (V/'development_native_review_scene5_6_agent/SEMANTIC_REVIEW_COMPLETE.json','roles')]
    reviewed=[];inputs=[]
    for f,role_key in seals:
        seal=read(f);assert seal['completed'] and seal['PASS'] and not seal['human_gold']
        records=lines(check(seal[role_key]));reviewed.extend(records);inputs.extend([bind(f),seal[role_key]])
    valfile=S/'native_reference_preparation/source_validation_visual_review_agent/SOURCE_VALIDATION360_NATIVE_ROLE_PROPOSALS_AGENT.jsonl'
    adjudication=read(V/'ROOT_SOURCE_VALIDATION_WEAK_ROLE_ADJUDICATION.json')
    assert adjudication['source_weak_role_proposals_accepted_for_pre_fit_diagnostic_join'] and not adjudication['train_allowed']
    reviewed.extend(lines(valfile));inputs.extend([bind(valfile),bind(V/'ROOT_SOURCE_VALIDATION_WEAK_ROLE_ADJUDICATION.json')])
    assert len(reviewed)==2424 and len({key(r) for r in reviewed})==2424
    lookup={key(r):r for r in reviewed};role_rows=[]
    for original in base:
        r=lookup[key(original)]
        for k,value in original.items():assert r[k]==value,(key(original),k)
        assert r['proposal_state'] in ('D','E','S','OOF','U') and not r['human_gold']
        item=dict(r)
        item.update(root_adjudicated_weak_role=True,exact_real_point_gold=False,
            role_proposal_usable_if_actual_query_correspondence=bool(r['proposal_state'] in ('D','E')),
            source_validation_training_allowed=False,weak_role_not_automatic_query_label=True)
        role_rows.append(item)
    candidates=defaultdict(list)
    for r in role_rows:candidates[(r['frame_index'],r['native_slot'])].append(r)
    association_file=V/'actual_query_association_agent/ALL1840_ORIGINAL_QUERY_ENTITY_AND_NATIVE_ROLE_PROPOSALS.jsonl'
    association=lines(association_file)
    assoc_summary=read(V/'actual_query_association_agent/REVIEW_SUMMARY.json')
    assert assoc_summary['completed'] and assoc_summary['PASS'] and assoc_summary['rows']==1840
    factsfile=S/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
    facts=lines(factsfile);assert len(association)==len(facts)==1840
    joined=[];counts=defaultdict(Counter)
    for i,(a,f) in enumerate(zip(association,facts)):
        assert a['global_index']==f['global_index']==i and a['frame_index']==f['frame_index']==i//8
        assert a['native_slot']==f['native_slot']==i%8
        assert a['id']==f['frame_id'] and a['query_id']==f['query_id'] and a['source_id']==f['source_id']
        assert a['split']==f['split'] and a['query_xy']==f['query_xy'] and a['L']==f['L']
        assert a['original_input_valid']==f['native_continuous_valid'] and a['nearest_valid']==f['nearest_valid']
        assert not a['coordinate_error_used_for_decision'] and not a['GT_nearest_or_permutation_used']
        assert a['point_quality_target']==-1 and not a['exact3px_point_quality_known']
        entity=a['entity_target'];supported=a['role_correspondence_supported'];gt=a['selected_gt_instance_index']
        assert entity in (-1,0,1)
        selected=lookup.get((i//8,gt,i%8)) if gt is not None else None
        if entity==1:assert selected is not None
        if supported:assert entity==1 and selected is not None
        state=selected['proposal_state'] if supported else 'U'
        target=0 if state=='D' else 1 if state=='E' else -1
        if target>=0:assert entity==1 and supported
        r=dict(global_index=i,frame_index=i//8,native_slot=i%8,id=a['id'],query_id=a['query_id'],
            source_id=a['source_id'],split=a['split'],query_xy=a['query_xy'],L=a['L'],
            original_input_valid=a['original_input_valid'],nearest_valid=a['nearest_valid'],
            feature_constructed=f['feature_constructed'],no_detection=a['no_detection'],
            frame_association_state=a['frame_association_state'],frame_entity_target=a['frame_entity_target'],
            entity_target=entity,selected_gt_instance_index=gt,role_correspondence_supported=supported,
            role_state=state,role_target=target,point_quality_label=-1,
            occluder_or_uncertainty=selected.get('occluder_or_uncertainty','UNKNOWN') if supported else 'ASSOCIATION_UNKNOWN_OR_WRONG_ENTITY',
            source_GT_proposal_state=selected['proposal_state'] if selected is not None else None,
            source_GT_proposal_confidence=selected.get('proposal_confidence') if selected is not None else None,
            association_reason=a['reason'],association_confidence=a['confidence'],
            source_role_review_reason=selected.get('review_reason_KO') if selected is not None else None,
            partial_clear_role_with_frame_entity_U=a['partial_clear_role_with_frame_entity_U'],
            candidate_GT_same_native_states=[dict(gt_instance_index=c['gt_instance_index'],proposal_state=c['proposal_state']) for c in candidates[(i//8,i%8)]],
            original_query_evidence=a['original_query_evidence'],provided_reference_overlay=a['provided_reference_overlay'],
            root_adjudicated_weak=True,human_gold=False,exact_real_point_gold=False,
            source_visibility_scores_used=False,source_only=True,coordinates_or_native_order_changed=False)
        joined.append(r)
        for group in ('ALL',r['split'],r['split']+f'::native{i%8}'):
            counts[group]['rows']+=1
            counts[group]['entity_'+str(entity)]+=1
            counts[group]['role_'+state]+=1
            counts[group]['role_supported']+=int(supported)
            counts[group]['input_valid']+=int(r['original_input_valid'])
            counts[group]['quality_unknown']+=1
            if r['original_input_valid']:
                counts[group]['entity_available_'+str(entity)]+=1
                counts[group]['role_available_'+state]+=1
    assert counts['ALL']['rows']==1840 and counts['ALL']['input_valid']==1538
    assert sum(r['nearest_valid'] for r in joined)==1535
    assert sum(r['role_correspondence_supported'] for r in joined)==972
    assert sum(r['partial_clear_role_with_frame_entity_U'] for r in joined)==44
    assert Counter(str(r['entity_target']) for r in joined)==Counter({'1':1076,'0':392,'-1':372})
    assert len({(r['source_id'],r['native_slot']) for r in joined})==1840
    assert all(r['role_target']==-1 for r in joined if r['entity_target']!=1)
    assert source_before==bind(basefile)
    master=V/'ROOT_ALL2424_NATIVE_WEAK_ROLE_PROPOSALS.jsonl';labels=V/'ROOT_ALL1840_SEPARATE_QUERY_SUPERVISION.jsonl'
    write_lines(master,role_rows);write_lines(labels,joined)
    inputs.extend([source_before,bind(association_file),bind(V/'actual_query_association_agent/REVIEW_SUMMARY.json'),
        bind(V/'actual_query_association_agent/ACTUAL_VISUAL_REVIEW_LOG.json'),bind(factsfile)])
    result=dict(completed=True,PASS=True,code=bind(__file__),root_policy=bind(V/'ROOT_POLICY_BEFORE_SOURCE_BRIDGE_REVIEW.json'),
        source_only=True,frames=230,rows=1840,GT_native_roles=2424,development_frames=185,validation_frames=45,
        source_fit_admitted=True,admission_scope='Assistant weak source supervision for fixed matched source experiment; no deployment/private admission.',
        labels=bind(labels),all2424_role_proposals=bind(master),inputs=inputs,
        counts={k:dict(c) for k,c in counts.items()},GT_role_counts=dict(Counter(r['proposal_state'] for r in role_rows)),
        all_original_queries_retained=True,original_q_L_flags_source_ids_and_splits_unchanged=True,
        human_gold=False,exact_real_point_gold=False,visibility_scores_used_for_labels=False,
        wrong_entity_never_E=True,partial44_clear_rear_roles_retained=True,
        real_point_quality_labels_created=0,no_GT_nearest_permutation_or_coordinate_correction=True,
        validation_never_admitted_for_gradient=True,new_model_or_fit_private_calls=0,
        weak_annotation_limits_KO=['역할의 가림은 원본 구조를 본 assistant 약한 감독이며 물리 키포인트 실측 gold가 아니다.',
            '같은 팔레트라도 앞면/역할 불명확13프레임의 native대응은U로 유지했다. 해당 개체 지지 라벨은 별도로 보존했다.',
            '각 프레임·instance·native의 모든 원행은 남기고 불명확한 주석만U로 유지했다. 일부 명확한44개 상단 팔레트 후방 점을 전프레임 모호함 때문에 버리지 않았다.',
            'source scene1 검증은 동일 공간/물리 물체가 반복될 수 있고 사람 gold·blind test가 아니다. WOOD/SQUARE 전체목표는 기존합성35검증과 함께 유지한다.'])
    write(V/'ROOT_SUPERVISION_SEAL.json',result)
    print(json.dumps(dict(completed=True,seal=bind(V/'ROOT_SUPERVISION_SEAL.json'),counts=result['counts']['ALL'])),flush=True)

if __name__=='__main__':main()
