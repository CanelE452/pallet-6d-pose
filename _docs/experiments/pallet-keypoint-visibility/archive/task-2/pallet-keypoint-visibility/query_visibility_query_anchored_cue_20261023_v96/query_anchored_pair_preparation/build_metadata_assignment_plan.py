"""Metadata only: freeze all960x16 donor ranks and all31680 planned rows."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
import query_anchored_pair_kernel as kernel

HERE=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def rows(b):
    assert bind(b['path'])['sha256']==b['sha256']
    return [json.loads(x)for x in Path(b['path']).read_text().splitlines()if x]
def write(name,x):
    with(HERE/name).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(name,x):
    with(HERE/name).open('x')as f:
        for r in x:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')

def main():
    protocol=read(HERE/'PAIR_PROTOCOL_BEFORE_ASSIGNMENT_AND_SOURCE_PIXELS.json')
    assert protocol['K']==kernel.K==16 and protocol['donor_ranking']['salt']==kernel.SALT
    inventory=rows(protocol['inputs']['inventory']);original=rows(protocol['inputs']['rows960']);pool=rows(protocol['inputs']['donor_pool'])
    ip=read(protocol['inputs']['inventory_protocol']['path'])
    loco=rows(ip['inputs']['loco_root160']);pallox=rows(ip['inputs']['pallox800']);cross=rows(ip['inputs']['all_cue_cross4832'])
    byq=defaultdict(list)
    for x in cross:byq[x['query_id']].append(x)
    assert len(inventory)==len(original)==960
    targets=[];assignments=[]
    for i,(inv,row)in enumerate(zip(inventory,original)):
        assert inv['row_index']==row['slot_index']==i and inv['original_q']==row['query_xy']and inv['original_L']==row['L']
        y=inv['query_anchored_AUX_label_D0_E1_Uminus1'];group=inv['matched_certified_group_GT_only'];regions=[]
        if y in [0,1]:
            assert group is not None
            if inv['source']=='LOCO':
                root=loco[i];assert y==0 and root['before_aux_state']=='D'
                regions=[dict(kind='LOCO_HALFOPEN_SUPPORT',bbox=root['support_bbox_xyxy']),dict(kind='LOCO_HALFOPEN_UNCERTAINTY',bbox=root['uncertainty_bbox_xyxy'])]
            else:
                old=pallox[i-160];members=[x for x in byq[old['query_id']]if x['certified_duplicate_group']==group]
                assert members and all(x['teacher_label']==y and not x['reference_null']for x in members)
                regions=[dict(kind='PALLOX_CLOSED_NATIVE_UNCERTAINTY',bbox=x['cue_uncertainty_bbox_unchanged'],cue_id=x['cue_id'],cue_view_index=x['cue_view_index'],
                    original_reference_xy=x['cue_reference_xy_unchanged'],original_coverage=x['coverage'])for x in members]
        targets.append(dict(base_row_index=i,source_family=inv['source'],raw_split=inv['raw_split'],original_q=row['query_xy'],original_L=row['L'],
            query_anchored_label=y,matched_certified_group=group,original_regions=regions,central4_native_stencil=inv['central4_positive_bilinear_native_cells'],
            construction_eligible=y==0,natural_E_before_only=y==1,original_C1_V95_label=inv['original_C1_V95_label_D0_E1_Uminus1'],
            inventory_reason_unchanged=inv['all_reasons'],native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False))
        ranked=kernel.fixed_donors(row,pool)
        assignments.append(dict(base_row_index=i,input_id=row['input_id'],native_slot_logical_only=row['native_slot_logical_only'],raw_split=row['split'],
            donor_pool=protocol['inputs']['donor_pool'],distinct_donor_ids=16,replacement_allowed=False,success_filter_allowed=False,
            ranks=[dict(donor_rank=j,donor_id=d['id'],ranking_SHA256=kernel.rank_key(row,d),scene_id=d['scene_id'],frame_id=d['frame_id'],instance_id=d['instance_id'],
                object_id=d['object_id'],split=d['split'],image_sha256=d['image_sha256'])for j,d in enumerate(ranked)]))
        if(i+1)%240==0:print(json.dumps(dict(stage='METADATA_DONOR_LOCK',done=i+1,all=960)),flush=True)
    assert Counter(t['query_anchored_label']for t in targets)=={0:61,1:4,-1:895}
    plan=kernel.occurrence_plan(original,targets)
    jsonl('MATCHED_TARGET960_GT_ONLY.jsonl',targets);jsonl('DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl',assignments)
    inputs=[];gt=[]
    for o in plan:
        i=o['base_row_index'];r=original[i];t=targets[i];before=o['variant']=='BEFORE';label=t['query_anchored_label']if before else -1
        inputs.append(dict(**o,input_id=r['input_id'],source=t['source_family'],raw_split=r['split'],image_sha256_before=r['image_sha256'],raw_hw=r['raw_hw'],
            original_q=r['query_xy'],original_L=r['L'],native_slot_logical_only=r['native_slot_logical_only'],original_input_valid=r['original_input_valid'],
            image_status='UNCHANGED_BOUND_BEFORE'if before else'PENDING_ROOT_SOURCE_CONSTRUCTION',model_forward_fields_only=['p3_roi','rgb96','crop_valid96']))
        gt.append(dict(**o,scalar_target_D0_E1_Uminus1=label,known=label>=0,query_anchored_before_label=t['query_anchored_label'],
            original_C1_V95_label=t['original_C1_V95_label'],eligible_attempt=(not before and t['construction_eligible']),
            success_label_if_certified=(1 if o['variant']=='OPAQUE_COVER'else 0)if not before and t['construction_eligible']else None,
            reason='ORIGINAL_QUERY_ANCHORED_BEFORE_LABEL'if before else'PENDING_CONSTRUCTION_AFTER_U',pair_ready=False,native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False))
    jsonl('INPUT_PLAN31680_INPUT_ONLY.jsonl',inputs);jsonl('GT_PLAN31680_GT_ONLY.jsonl',gt)
    stats=dict(completed=True,PASS=True,metadata_preparation_only=True,base_rows=960,occurrences=31680,before_occurrences=960,after_occurrences=30720,
        K=16,assigned_rank_jobs=15360,eligible_D_rows=61,eligible_D_pair_attempts=976,natural_E_before_only=4,
        initial_plan_labels=dict(Counter(str(x['scalar_target_D0_E1_Uminus1'])for x in gt)),
        by_source_raw_split={s+'|'+sp:dict(rows=sum(t['source_family']==s and t['raw_split']==sp for t in targets),
            before_D=sum(t['source_family']==s and t['raw_split']==sp and t['query_anchored_label']==0 for t in targets),
            before_E=sum(t['source_family']==s and t['raw_split']==sp and t['query_anchored_label']==1 for t in targets),
            planned_pair_attempts=16*sum(t['source_family']==s and t['raw_split']==sp and t['query_anchored_label']==0 for t in targets))for s,sp in sorted({(t['source_family'],t['raw_split'])for t in targets})},
        complete_original_matched_group_boxes_retained=True,all_failures_and_unknowns_will_remain=True,no_rank_replacement_or_success_selection=True,
        limitations=['Large original LOCO support/uncertainty may make all controls infeasible; do not shrink boxes. Record protected area/L² on every actual attempt.','Pallox-only successful controls would not prove flat-pallet-family E transfer. Natural LOCO D remain separately typed.','All after labels are currently U pending actual certificate; no actual source images/edits/features were created.','Query-specific physical cue AUX is a new target, not a rewrite of V95 known-set labels or native/human semantic gold.'],
        actual_source_RGB_mask_features_weights_models_fit_current_PnP=0,training_admission=False)
    write('METADATA_ASSIGNMENT_PLAN_SUMMARY.json',stats)
    names=['PAIR_PROTOCOL_BEFORE_ASSIGNMENT_AND_SOURCE_PIXELS.json','query_anchored_pair_kernel.py','test_generated_query_anchored_pairs.py',
        'GENERATED_QUERY_ANCHORED_PAIR_TESTS.json','build_metadata_assignment_plan.py','MATCHED_TARGET960_GT_ONLY.jsonl','DONOR_ASSIGNMENT960_K16_BEFORE_PIXELS.jsonl',
        'INPUT_PLAN31680_INPUT_ONLY.jsonl','GT_PLAN31680_GT_ONLY.jsonl','METADATA_ASSIGNMENT_PLAN_SUMMARY.json']
    write('PREPARATION_AND_METADATA_LOCK_SEAL.json',dict(completed=True,PASS=True,files=[bind(HERE/n)for n in names],actual_source_prototype_authorized=False,training_admission=False,current_allowed=False))
    print(json.dumps(stats))

if __name__=='__main__':main()
