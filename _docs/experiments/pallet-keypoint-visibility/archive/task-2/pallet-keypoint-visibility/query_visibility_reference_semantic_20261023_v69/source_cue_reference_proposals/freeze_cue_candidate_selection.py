"""Freeze8 D-candidate IDs, metadata only, before finer cue localization."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent;V69=OUT.parent;BASE=V69.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=OUT/'CUE_PROPOSAL_SELECTION_LOCK.json';assert not dest.exists()
    reviewpath=V69/'source_anchor_review/EIGHT_FRAME_REFERENCE_VISUAL_REVIEW.json'
    review=json.loads(reviewpath.read_text());inputs=BASE/'query_visibility_public_real_20261022_v63/sam_source/INPUTS.jsonl'
    meta={r['id']:r for r in map(json.loads,inputs.read_text().splitlines()) if r.get('source')=='SYNTH' and r['split']=='train'}
    pool=[]
    for f in review['frames']:
        assert f['split']=='train'
        for a in f['observations8']:
            if a['AI_observation_tag']!='D_candidate':continue
            key=f['frame_id']+':'+str(a['native_corner'])
            pool.append((hashlib.sha256(key.encode()).hexdigest(),key,f,a))
    picked=sorted(pool,key=lambda x:(x[0],x[1]))[:8];assert len({x[2]['frame_id'] for x in picked})>=4
    rows=[]
    for rank,(h,key,f,a) in enumerate(picked):
        m=meta[f['frame_id']];k=a['native_corner']
        assert m['query'][k]==a['saved_initial_query_xy'] and m['image_sha256']==f['image_sha256']
        rows.append({'selection_rank':rank,'candidate_point_ID':key,'selection_SHA256_UTF8_ID':h,
            'frame_id':f['frame_id'],'split':'train','source_asset':f['source_asset'],
            'original_renderer_r_xy':a['reference_xy'],'original_query_xy':m['query'][k],
            'original_query8_xy':m['query'],'original_side_L':m['side'],'original_target_box_xyxy':m['box'],
            'original_native_index':k,'original_writer_index':a['original_writer_corner'],
            'raw_hw':f['raw_hw'],'original_image':f['image'],'original_image_sha256':f['image_sha256'],
            'original_annotation':f['annotation'],'original_annotation_sha256':f['annotation_sha256'],
            'prior_observation_tag':a['AI_observation_tag'],'prior_AI_observation':a['observation_ko'],
            'prior_AI_observation_confidence':a['AI_observation_confidence_only'],
            'raw96_sheet':f['raw_crop_sheet'],'marked_sheet':{'path':f['marked_sheet']['frame_sheet'],'sha256':f['marked_sheet']['sha256']}})
    result={'schema':'AI_OPERATIONAL_CUE_REFERENCE_SELECTION_V1','status':'FROZEN_BEFORE_FINER_PIXEL_LOCALIZATION',
        'selection':'First8 ascending SHA256(UTF8 frameID+":"+decimal native index) among prior D30. First8 already span5 frames; no forced substitution. Pool itself came from earlier deterministic8 TRAIN RGB review.',
        'pool_D_candidate_count':len(pool),'selected_count':8,'selected_distinct_frames':len({x['frame_id'] for x in rows}),
        'scope':'Existing marked/raw96 review rasters only; no new original RGB decode, mask/model/performance/current/VAL/fit/forward/PnP. Input JSONL metadata supplies unchanged q/side/box only.',
        'new_reference_semantics':'New cue_refID identifies one visually localized2D target pallet corner/edge cue. Associated cue xy and explicit visual uncertainty are proposals, distinct from old renderer cuboid r and native8 truth. Old r/q/L/native index remain immutable lineage; no nearest-foreground reassignment.',
        'association_admission':'If the physical cue cannot be unambiguously localized or associated with the named prior native corner, set U and do not invent a point. Native index remains lineage only, not admitted canonical8 semantic gold.',
        'uncertainty_policy':'Per-cue visual localization range at native pixel precision, including blur/bevel/edge feature extent. No common tuned radius and no model score. New cue reference xy must be inside its explicit half-open uncertainty rectangle. Do not force old renderer r into the new cue region.',
        'coverage_policy':'Use original q-centered square side L unchanged, intersected with image cell footprint[-.5,W-.5)x[-.5,H-.5). FULL covers entire uncertainty rectangle, PARTIAL some, MISS none; unresolved U cue gets INVALID/UNRESOLVED, not invented coverage or E/O. Follow bound V69 pure draft geometry, not a new crop recipe.',
        'teacher_status':'AI_OBSERVATIONAL_PROPOSAL_ONLY_ROOT_REVIEW_REQUIRED; NO_HUMAN_GOLD; NO_TRAINING_ADMISSION; no natural E/S admitted',
        'composites_or_jitters_generated':False,'training_authorized':False,
        'bindings':[{'path':str(p),'sha256':sha(p)} for p in [reviewpath,V69/'source_anchor_review/VISUAL_REVIEW_SEAL.json',inputs,V69/'DRAFT_PROTOCOL.json',V69/'reference_visibility_contract.py',Path(__file__)]],
        'candidates':rows}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest),'candidates':[r['candidate_point_ID'] for r in rows]}))
if __name__=='__main__':main()
