"""Freeze AI RGB-only cue geometry BEFORE any new ownership sanity read."""
from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE/'RGB_ONLY_WORKING_LOCALIZATION/WORKING_PROPOSALS.json'
# RGB-only refinement after viewing unmarked64 crops and finer RGB grids.
REFINEMENTS={
 0:([246,344],[241,338,252,350],'RGB fine-grid: center the actual gold front-rim turn, rather than the adjacent dark side.'),
 4:([281,300],[277,295,286,306],'RGB fine-grid: blade tip extends right of the first rough20px reading.'),
 6:([184,130],[179,124,190,137],'RGB fine-grid: top/front/left edge junction is below the initial rough reading.'),
 8:([231,383],[225,377,238,391],'RGB fine-grid: wood top/front/left junction lies right and above the initial rough point.')}


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def ref(f):return dict(path=str(Path(f).resolve()),sha256=sha(f))


def main():
    out=HERE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json';assert not out.exists()
    working=json.loads(WORK.read_text());source=[json.loads(s)for s in (HERE/'SOURCE_LINEAGE_GT_ONLY.jsonl').read_text().splitlines()]
    proposals=[]
    for i,row in enumerate(working['proposals']):
        r=source[i]['original'];q8=np.asarray(r['query'],float);xy=row['reference_xy'];bb=row['uncertainty_bbox_xyxy'];change=None
        if i in REFINEMENTS:xy,bb,change=REFINEMENTS[i]
        k=int(np.argmin(np.linalg.norm(q8-np.asarray(xy),axis=1)));q=q8[k].tolist();L=r['side'];h,w=r['raw_hw']
        crop=[max(-.5,q[0]-L/2),max(-.5,q[1]-L/2),min(w-.5,q[0]+L/2),min(h-.5,q[1]+L/2)]
        full=crop[0]<=bb[0] and crop[1]<=bb[1] and crop[2]>=bb[2] and crop[3]>=bb[3]
        intersects=min(crop[2],bb[2])>max(crop[0],bb[0])and min(crop[3],bb[3])>max(crop[1],bb[1])
        availability='FULL'if full else'PARTIAL'if intersects else'MISS'
        proposals.append(dict(cue_refID=r['id']+':AI_PHYSICAL_CUE_V1',target_id=r['id'],split=r['split'],source_RGB=r['image'],reference_xy=xy,uncertainty_bbox_xyxy=bb,uncertainty_semantics=row['uncertainty_semantics'],cue_description_ko=row['cue_description_ko'],AI_localization_confidence=row['AI_localization_confidence'],reference_visibility_proposal='D_AI_OBSERVATIONAL_PROPOSAL',native8_semantic_ID='UNKNOWN',source_teacher_tier='AI_RGB_PHYSICAL_CUE_PROPOSAL_NOT_HUMAN_GOLD_NOT_NATIVE8',RGB_evidence='Unmarked native RGB64 crop + raw full image/native20px grid inspected; distinguishable physical rim/edge-turn/corner, not mask-only membership.',prior_mask_exposure=True,new_GT_mask_read_before_geometry_freeze=False,rough_working_reference_xy=row['reference_xy'],RGB_only_refinement=change,GT_only_existing_query_correspondence=dict(rule='Nearest Euclidean existing available q to this newly visually localized cue, after cue localization; row identity is lineage only.',native_query_row_lineage=k,query_xy=q,query_minus_reference_xy=(q8[k]-np.asarray(xy)).tolist(),side_L_unchanged=L,input_box_unchanged=r['box'],observed_query_crop_xyxy=crop,uncertainty_crop_availability=availability,GT_reference_or_offset_is_not_model_input=True),training_target_status='U_PENDING_OWNERSHIP_SANITY_AND_ROOT_REVIEW',cropMISS_orPARTIAL_is_not_semantic_E_or_physical_absence=True))
    report=dict(schema='V71_FROZEN_AI_YCB_PHYSICAL_CUE_PROPOSALS_V1',completed=True,geometry_frozen_before_ownership_sanity=True,code=ref(__file__),source_selection_lock=ref(HERE/'SELECTION_LOCK.json'),working_RGB_only_proposals=ref(WORK),RGB_native20_grid=ref(HERE/'RAW_NATIVE20_LOCALIZATION/MANIFEST.json'),fine_RGB_only_grid=ref(HERE/'RGB_ONLY_WORKING_LOCALIZATION/FINE_RGB_ONLY_5PX_GRID.png'),proposals=proposals,prior_mask_exposure=True,not_blind_annotation=True,not_human_or_canonical8_GT=True,semantic_training_admission=False,refined_after_mask_sanity=False,scope='One actual RGB-localized operational physical cue per selected YCB target; hidden-reference identifiability remains unresolved, not solved by nearest-q correspondence.',new_model_forward_training_current_PnP=0)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (HERE/'CUE_GEOMETRY_SEAL.json').write_text(json.dumps(dict(completed=True,proposal=ref(out),geometry_frozen_before_ownership_sanity=True,GT_mask_reads_in_this_stage=0,all_query8_box_L_and_native_lineage_unchanged=True),indent=2)+'\n');print(json.dumps(ref(out)))


if __name__=='__main__':main()
