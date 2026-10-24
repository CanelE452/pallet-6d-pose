import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'query_visibility_matched_query_scalar_20261023_v94'
O=R/'aux_cue_review_comparison'
B=R/'reviewer_B_actual_cues';C=R/'reviewer_C_actual_cues'
S=P/'query_visibility_flat_pallet_cue_bridge_20261023_v92/rgb96_review'
STATES=['YES','NO','UNKNOWN']

def bind(p):
    b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
    assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):return json.loads(p.read_text())
def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def check_binding(v):
    actual=bind(Path(v['path']));assert actual['sha256']==v['sha256']

def box_diag(a,b):
    if a is None or b is None:return None
    assert len(a)==len(b)==4
    for x in [a,b]:assert 0<=x[0]<x[2]<=96 and 0<=x[1]<x[3]<=96
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);ab=(b[2]-b[0])*(b[3]-b[1])
    centre_distance=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)
    return dict(exact_equal=a==b,area_B=aa,area_C=ab,intersection_area=inter,positive_area_overlap=inter>0,IoU=inter/(aa+ab-inter),centre_distance_px96=centre_distance,union_bbox_xyxy96=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])])

def kind_diag(a,b):
    def tokens(s):return set(re.findall(r'[A-Z0-9]+',(s or '').upper()))
    aa,bb=tokens(a),tokens(b)
    # Broad role components only. This is NOT a synonym/gold matching rule.
    def roles(t):
        out=set()
        if 'DECK' in t and t.intersection({'BLOCK','SUPPORT','FOOT'}):out.add('DECK_SUPPORT')
        if 'FLOOR' in t and t.intersection({'SUPPORT','FOOT','BLOCK'}):out.add('SUPPORT_FLOOR')
        if t.intersection({'FORK','OPENING','RIM'}):out.add('OPENING_RIM')
        if t.intersection({'SIDE','SIDEWOOD','PLANK','END'}):out.add('SIDE_OR_PLANK_END')
        return out
    ar,br=roles(aa),roles(bb)
    return dict(exact_string_equal=a==b,shared_role_components=sorted(ar&br),B_role_components=sorted(ar),C_role_components=sorted(br),shared_structure_words=sorted((aa&bb)-{'PLASTIC','WOOD','WOODEN','CANDIDATE','NONE','WITH','AND','OR','THE','ONLY','LEFT','RIGHT'}),compatibility_requires_visual_review=True)

def owner_bucket(s):
    if s in ['HIGH','CONTEXT_COMPATIBLE_SINGLE_PALLET_AI']:return 'CONTEXT_COMPATIBLE'
    if s in ['UNKNOWN_NO_VALID_QUERY','NO_QUERY','OOB_OR_INVALID_QUERY']:return 'FACTUAL_QUERY_U'
    if s in ['NOT_PALLET','NO_PALLET_STRUCTURE_OBSERVED_IN_CROP']:return 'NOT_PALLET_OR_NO_CUE_OBSERVED'
    return 'UNRESOLVED_OWNERSHIP_OR_STRUCTURE'

def crosstab(rows):
    m=[[0]*3 for _ in range(3)]
    for r in rows:m[STATES.index(r['B_state'])][STATES.index(r['C_state'])]+=1
    return dict(rows_B=STATES,columns_C=STATES,matrix=m,N=len(rows))

def generated():
    assert box_diag([0,0,10,10],[10,0,20,10])['intersection_area']==0
    assert box_diag([0,0,10,10],[5,0,15,10])['IoU']==1/3
    assert box_diag([0,0,10,10],[0,0,10,10])['IoU']==1
    assert box_diag(None,[0,0,1,1]) is None
    assert crosstab([dict(B_state='YES',C_state='UNKNOWN')])['matrix']==[[0,0,1],[0,0,0],[0,0,0]]
    assert kind_diag('DECK_SUPPORT_OPENING_JUNCTION','DECK_BLOCK_JUNCTION')['shared_role_components']==['DECK_SUPPORT']
    write(O/'GENERATED_COMPARISON_MATH_FIXTURES.json',dict(PASS=True,fixtures=6,threshold_or_gold_admission_rule=False,new_pixels_models_fit_current_PnP=0))

def main():
    generated()
    bseal=load(B/'REVIEW_SEAL.json');cseal=load(C/'FINAL_REVIEW_SEAL.json')
    assert bseal['completed'] and bseal['PASS'] and cseal['completed'] and cseal['PASS']
    for p in [bseal['observations'],bseal['receipts'],bseal['review'],cseal['observations'],cseal['summary'],cseal['all20_figures_and_raw_sheets_viewed']]:check_binding(p)
    br=jsonl(Path(bseal['observations']['path']));cr=jsonl(Path(cseal['observations']['path']));source=jsonl(S/'ROWS_INPUT_ONLY.jsonl')
    assert len(br)==len(cr)==len(source)==160
    bviews=load(B/'ACTUAL_VIEW_RECEIPTS.json')['all_frames'];cviews=load(C/'ACTUAL_ALL20_FIGURE_AND_SAVED96_VIEW_RECEIPTS.json')['receipts'];cgallery=load(C/'ACTUAL_CUE_REVIEW_SUMMARY_V2_KO.json')['gallery']
    assert len(bviews)==len(cviews)==len(cgallery)==20
    records=[];review_ids=[];both_yes=[];by_frame=defaultdict(list);by_split=defaultdict(list)
    for i,(b,c,s) in enumerate(zip(br,cr,source)):
        assert b['input_id']==c['input_id']==s['input_id']
        assert b['native_slot']==c['native_slot']==s['native_slot']==i%8
        assert b['frame_index']==c['frame_index']==s['frame_index']==i//8
        assert b['original_q']==c['original_q']==s['query_xy'] and b['original_L']==c['original_L']==s['L']
        assert b['split']==c['split']==s['split'] and b['image_sha256']==c['image_sha256']==s['image_sha256']
        assert b['source_validity']=={k:s[k] for k in b['source_validity']}
        assert c['source_validity']==s['input_status']
        valid=s['query_center_valid'];bs=b['visible_aux_structuralcue'];cs=c['visible_aux_structuralcue']
        assert bs in STATES and cs in STATES
        if not valid:assert bs==cs=='UNKNOWN'
        assert b['native_expected_part_state']==c['native_expected_part_state']=='UNKNOWN'
        assert b['native_human_gold']==c['native_human_gold']==False and not b['training_admitted'] and not c['training_admitted']
        bd=b['crop_support_bbox_xyxy96_or_null'];cd=c['crop_support_bbox_xyxy96_or_null']
        overlap=box_diag(bd,cd);kd=kind_diag(b['cue_kind'],c['cue_kind'])
        ob,oc=owner_bucket(b['target_ownership_confidence']),owner_bucket(c['target_ownership_confidence'])
        is_both=bs==cs=='YES';xor=(bs=='YES')!=(cs=='YES')
        uncertainties=[v for v in [b['support_uncertainty_px'],c['support_uncertainty_px']] if v is not None]
        for u in uncertainties:assert isinstance(u,(int,float)) and math.isfinite(u) and u>=0
        union_proposal=None
        if is_both:
            assert overlap is not None and len(uncertainties)==2
            ub=overlap['union_bbox_xyxy96'];u=max(uncertainties)
            raw=[ub[0]-u,ub[1]-u,ub[2]+u,ub[3]+u]
            union_proposal=dict(union_support_bbox_xyxy96=ub,max_visual_uncertainty_px=u,expanded_uncertainty_bbox_unclipped96=raw,expanded_uncertainty_bbox_clipped96=[max(0,raw[0]),max(0,raw[1]),min(96,raw[2]),min(96,raw[3])],extends_beyond_saved_crop=any(z<0 or z>96 for z in raw),proposal_only=True,root_visual_decision_required=True,no_native_ref_defined=True,training_admitted=False)
        flags=[]
        if bs!=cs:flags.append('YES_NO_U_DISAGREEMENT')
        if ob!=oc:flags.append('OWNERSHIP_BUCKET_DISAGREEMENT')
        if b['multiplecue_flag']!=c['multiplecue_flag']:flags.append('MULTIPLE_CUE_FLAG_DISAGREEMENT')
        if is_both and overlap and not overlap['positive_area_overlap']:flags.append('BOTH_YES_DISJOINT_SUPPORT')
        if is_both and not kd['shared_role_components']:flags.append('BOTH_YES_NO_SHARED_BROAD_ROLE')
        if is_both and (b['multiplecue_flag'] or c['multiplecue_flag']):flags.append('BOTH_YES_MULTIPLE_CUE_REQUIRES_ROOT_REVIEW')
        rid=f'{s["input_id"]}:{s["native_slot"]}'
        r=dict(row_index=i,input_id=s['input_id'],frame_index=s['frame_index'],native_slot=s['native_slot'],split=s['split'],image_sha256=s['image_sha256'],original_q=s['query_xy'],original_L=s['L'],source_validity=s['input_status'],query_center_valid=valid,B_state=bs,C_state=cs,B_appearance=b['appearance_structuralcue_observed'],C_appearance=c['visible_pallet_structure_appearance'],B_ownership=b['target_ownership_confidence'],C_ownership=c['target_ownership_confidence'],B_owner_bucket=ob,C_owner_bucket=oc,B_multiplecue=b['multiplecue_flag'],C_multiplecue=c['multiplecue_flag'],B_cue_kind=b['cue_kind'],C_cue_kind=c['cue_kind'],B_support_bbox=bd,C_support_bbox=cd,B_uncertainty=b['support_uncertainty_px'],C_uncertainty=c['support_uncertainty_px'],box_overlap_diagnostic=overlap,kind_diagnostic=kd,both_YES=is_both,one_reviewer_YES=xor,review_flags=flags,union_plus_max_uncertainty_proposal=union_proposal,root_final_aux_decision=None,native_expected_part_state='UNKNOWN',human_gold=False,training_admitted=False,B_record_unchanged=b,C_record_unchanged=c)
        records.append(r);by_frame[s['frame_index']].append(r);by_split[s['split']].append(r)
        if bs=='YES' or cs=='YES' or flags:review_ids.append(rid)
        if is_both:both_yes.append(rid)
    assert sum(r['query_center_valid'] for r in records)==122
    p=O/'ALL160_B_C_LOCKED_METADATA_COMPARISON.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records))
    flat=[]
    for r in records:
        d={k:r[k] for k in ['row_index','input_id','frame_index','native_slot','split','query_center_valid','B_state','C_state','B_ownership','C_ownership','B_multiplecue','C_multiplecue','B_cue_kind','C_cue_kind','both_YES','one_reviewer_YES']}
        d['support_IoU']=r['box_overlap_diagnostic']['IoU'] if r['box_overlap_diagnostic'] else ''
        d['shared_broad_roles']='|'.join(r['kind_diagnostic']['shared_role_components'])
        d['review_flags']='|'.join(r['review_flags']);flat.append(d)
    with (O/'ALL160_B_C_SUMMARY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    links=[]
    for i in range(20):
        assert bviews[i]['frame_index']==cviews[i]['frame_index']==cgallery[i]['frame_index']==i
        assert bviews[i]['context']['sha256']==cviews[i]['original_context_figure']['sha256']
        links.append(dict(frame_index=i,input_id=source[i*8]['input_id'],split=source[i*8]['split'],original_raw_context=bviews[i]['context'],B_exact96_sheet=bviews[i]['exact96_sheet'],C_exact96_sheet=cviews[i]['raw96_sheet'],C_raw_marked_gallery=cgallery[i]['figure'],original_slots=list(range(8)),proposed_or_disagreement_slots=[r['native_slot'] for r in by_frame[i] if r['both_YES'] or r['one_reviewer_YES'] or r['review_flags']],all_original_views_preserved=True,image_body_read_or_rehashed_in_comparison=False))
    write(O/'ALL20_ACTUAL_IMAGE_REVIEW_LINKS.json',dict(frames=links,all20=True,all160=True,root_must_view_all20_and_all_proposals_disagreements=True))
    valid=[r for r in records if r['query_center_valid']]
    summary=dict(schema='V94_LOCKED_B_C_AUX_CUE_REVIEW_COMPARISON_KO',completed=True,PASS_metadata_parity=True,frames=20,slots=160,valid=122,factual_U38_unchanged=True,
      state_tables=dict(all160=crosstab(records),valid122=crosstab(valid),by_split={k:dict(all=crosstab(v),valid=crosstab([x for x in v if x['query_center_valid']])) for k,v in by_split.items()},by_frame={str(k):dict(all=crosstab(v),valid=crosstab([x for x in v if x['query_center_valid']])) for k,v in by_frame.items()}),
      counts=dict(B_YES=sum(r['B_state']=='YES' for r in records),C_YES=sum(r['C_state']=='YES' for r in records),both_YES=len(both_yes),either_YES=sum(r['B_state']=='YES' or r['C_state']=='YES' for r in records),one_YES=sum(r['one_reviewer_YES'] for r in records),all_state_disagreements=sum(r['B_state']!=r['C_state'] for r in records),ownership_bucket_disagreements=sum(r['B_owner_bucket']!=r['C_owner_bucket'] for r in records),multiplecue_flag_disagreements=sum(r['B_multiplecue']!=r['C_multiplecue'] for r in records),bothYES_positive_overlap=sum(r['both_YES'] and r['box_overlap_diagnostic']['positive_area_overlap'] for r in records),bothYES_disjoint=sum(r['both_YES'] and not r['box_overlap_diagnostic']['positive_area_overlap'] for r in records),bothYES_shared_broad_role=sum(r['both_YES'] and bool(r['kind_diagnostic']['shared_role_components']) for r in records),bothYES_any_multiplecue_flag=sum(r['both_YES'] and (r['B_multiplecue'] or r['C_multiplecue']) for r in records)),
      both_YES_ids=both_yes,root_review_ids=review_ids,bothYES_support_diagnostics=[dict(input_id=r['input_id'],native_slot=r['native_slot'],frame_index=r['frame_index'],IoU=r['box_overlap_diagnostic']['IoU'],kind=r['kind_diagnostic'],union_proposal=r['union_plus_max_uncertainty_proposal']) for r in records if r['both_YES']],
      interpretation_KO='약한 AI 관찰 비교이며 accuracy/CM/gold consensus가 아니다. IoU/역할 문자열 공통점은 같은 cue의 증명이 아니다. BothYES와 union+maxunc도 root의 실제 원본검토 전에는 제안일 뿐이며 training/native admission false를 유지한다. NO/U는 E/S negative가 아니다. Multiplecue flag는 보이는 복수 구조와 미해결 identity를 구분하기 위해 root가 원문/이미지를 확인해야 하며 자동 제외·채택 문턱으로 쓰지 않았다.',
      input_parity=dict(all160_id_q_L_order_split_sha_valid_exact=True,all38_factual_U_both_UNKNOWN=True,native_expected_part_all_UNKNOWN=True,original_records_embedded_unchanged=True),
      policy=bind(O/'POLICY_BEFORE_B_C_LEDGER_COMPARISON.json'),B_seal=bind(B/'REVIEW_SEAL.json'),C_seal=bind(C/'FINAL_REVIEW_SEAL.json'),comparison=bind(p),image_links=bind(O/'ALL20_ACTUAL_IMAGE_REVIEW_LINKS.json'),
      new_pixels_models_forward_fit_current_PnP=0,training_admission=False,human_native_gold_added=0,goal_status='active',fully_solved=False)
    write(O/'COMPARISON_SUMMARY_KO.json',summary)
    write(O/'COMPARISON_SEAL.json',dict(completed=True,PASS=True,policy=bind(O/'POLICY_BEFORE_B_C_LEDGER_COMPARISON.json'),inputs=[bind(B/'REVIEW_SEAL.json'),bind(C/'FINAL_REVIEW_SEAL.json'),bind(S/'ROWS_INPUT_ONLY.jsonl')],outputs=[bind(p),bind(O/'ALL160_B_C_SUMMARY.csv'),bind(O/'ALL20_ACTUAL_IMAGE_REVIEW_LINKS.json'),bind(O/'COMPARISON_SUMMARY_KO.json'),bind(O/'GENERATED_COMPARISON_MATH_FIXTURES.json')],code=bind(Path(__file__)),new_pixels_models_fit_current_PnP=0,original_modified=0,training_admission=False))
    print(json.dumps(dict(summary=bind(O/'COMPARISON_SUMMARY_KO.json'),seal=bind(O/'COMPARISON_SEAL.json'),counts=summary['counts'],valid_table=summary['state_tables']['valid122']),ensure_ascii=False))

if __name__=='__main__':main()
