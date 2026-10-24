"""Independent metadata/native-raster replay; no builder/helper imports or models."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/ycb_cue_review'


def binding(p):
    p = Path(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def load(name):
    return json.loads((DATA / name).read_text())


def check_bound(item):
    assert binding(item['path'])['sha256'] == item['sha256'], item['path']


def main():
    proposals = load('FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json')
    sanity = load('FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json')
    seal = load('CUE_GEOMETRY_SEAL.json')
    lock = load('SELECTION_LOCK.json')
    lines = [json.loads(x) for x in (DATA/'SOURCE_LINEAGE_GT_ONLY.jsonl').read_text().splitlines()]
    lineage = {x['id']: x for x in lines}
    inputs = [json.loads(x) for x in (DATA/'INPUTS_RGB_ONLY.jsonl').read_text().splitlines()]
    original_inputs = {x['id']: x for x in inputs if x['source'] == 'YCB_ORIGINAL'}
    assert len(proposals['proposals']) == len(sanity['cases']) == len(lineage) == 12
    assert len(inputs) == 24
    assert all(set(x) == {'id','source','split','image','image_sha256','raw_hw','query8','side','box'} for x in inputs)
    check_bound(seal['proposal'])
    check_bound(sanity['frozen_proposals'])
    check_bound(sanity['geometry_seal'])
    for x in lock['outputs']:
        check_bound(x)
    assert proposals['geometry_frozen_before_ownership_sanity']
    assert proposals['prior_mask_exposure'] and proposals['not_blind_annotation']
    assert not proposals['refined_after_mask_sanity']
    cases = []
    read_bindings = []
    for prop, reported in zip(proposals['proposals'], sanity['cases']):
        ident = prop['target_id']
        assert ident == reported['target_id']
        lin = lineage[ident]
        inp = original_inputs[ident]
        orig, ctrl = lin['original'], lin['controlled']
        assert prop['split'] == lin['split'] == inp['split'] == orig['split'] == ctrl['split']
        assert ctrl['occluder_original']['split'] == orig['split']
        assert ctrl['occluder_original']['image']['sha256'] != orig['image']['sha256']
        assert inp['query8'] == orig['query'] == ctrl['query']
        assert inp['side'] == orig['side'] == ctrl['side']
        assert inp['box'] == orig['box'] == ctrl['box']
        assert prop['native8_semantic_ID'] == reported['native8_semantic_ID'] == 'UNKNOWN'
        corr = prop['GT_only_existing_query_correspondence']
        q8 = np.array(inp['query8'], dtype=np.float64)
        r = np.array(prop['reference_xy'], dtype=np.float64)
        k = int(np.argmin(((q8-r)**2).sum(axis=1)))
        q = q8[k]
        assert k == corr['native_query_row_lineage'] == reported['unchanged_query_row']
        assert q.tolist() == corr['query_xy'] == reported['unchanged_query_xy']
        L = inp['side']
        assert L == corr['side_L_unchanged'] == reported['unchanged_L']
        assert inp['box'] == corr['input_box_unchanged']
        assert prop['reference_xy'] == reported['reference_xy']
        assert prop['uncertainty_bbox_xyxy'] == reported['uncertainty_bbox_xyxy']
        crop = [q[0]-L/2, q[1]-L/2, q[0]+L/2, q[1]+L/2]
        assert crop == corr['observed_query_crop_xyxy']
        x0,y0,x1,y1 = prop['uncertainty_bbox_xyxy']
        full = crop[0] <= x0 and crop[1] <= y0 and crop[2] >= x1 and crop[3] >= y1
        overlap = min(crop[2],x1) > max(crop[0],x0) and min(crop[3],y1) > max(crop[1],y0)
        availability = 'FULL' if full else 'PARTIAL' if overlap else 'MISS'
        assert availability == corr['uncertainty_crop_availability'] == reported['observed_crop_availability']
        for b in [orig['image'], orig['visible'], orig['full_proxy'], ctrl['labels']]:
            check_bound(b)
            read_bindings.append(b)
        rgb = cv2.imread(orig['image']['path'], cv2.IMREAD_COLOR)
        vis = cv2.imread(orig['visible']['path'], cv2.IMREAD_GRAYSCALE) > 0
        fullmask = cv2.imread(orig['full_proxy']['path'], cv2.IMREAD_GRAYSCALE) > 0
        assert list(rgb.shape[:2]) == inp['raw_hw']
        # The uncertainty bbox denotes native pixel centers in a half-open rectangle.
        xx = np.arange(math.ceil(x0), math.ceil(x1))
        yy = np.arange(math.ceil(y0), math.ceil(y1))
        bbox_vis = vis[np.ix_(yy,xx)]
        ri = np.floor(r+.5).astype(int)
        qi = np.floor(q+.5).astype(int)
        with np.load(ctrl['labels']['path'], allow_pickle=False) as z:
            assert np.array_equal(z['original_visible'].astype(bool),vis)
            assert np.array_equal(z['original_full'].astype(bool),fullmask)
            paste = z['paste_mask'].astype(bool)
            after_vis = z['visible_positive'].astype(bool)
        center_vis = bool(vis[ri[1],ri[0]])
        assert center_vis == reported['reference_pixel_visible_mask']
        assert bool(fullmask[ri[1],ri[0]]) == reported['reference_pixel_full_proxy']
        assert int(bbox_vis.sum()) == reported['uncertainty_visible_pixels']
        assert bbox_vis.size == reported['uncertainty_bbox_native_pixels']
        cover = int(paste[np.ix_(yy,xx)].sum())
        assert cover == reported['existing_controlled_paste_covered_uncertainty_pixels'] == 0
        assert bool(vis[qi[1],qi[0]]) == reported['query_pixel_original_target_visible']
        assert bool(after_vis[qi[1],qi[0]]) == reported['query_pixel_after_target_visible']
        # This replays the conservative candidate rule; pixel membership is not semantic proof.
        teacher = 'D_CONDITIONAL_AI_PHYSICAL_CUE' if center_vis else 'U'
        assert teacher == reported['original_cue_teacher_candidate']
        train = teacher if availability == 'FULL' else 'U_CROP_'+availability
        assert train == reported['original_training_teacher_candidate']
        cases.append({'target_id':ident,'split':inp['split'],'native8_semantic_ID':'UNKNOWN',
                      'reference_xy':r.tolist(),'native_query_row_lineage':k,
                      'reference_distance_to_q_px':float(np.linalg.norm(q-r)),
                      'crop_availability':availability,'visible_raster_reference_pixel':center_vis,
                      'uncertainty_target_visible_pixels':int(bbox_vis.sum()),
                      'uncertainty_native_pixels':int(bbox_vis.size),'existing_paste_covers_pixels':cover,
                      'conditional_reference_teacher_candidate':teacher,
                      'conditional_original_training_candidate':train})
    counts = {s: {'targets':sum(c['split']==s for c in cases),
                  'availability':dict(Counter(c['crop_availability'] for c in cases if c['split']==s)),
                  'reference_D_conditional_candidates':sum(c['split']==s and c['conditional_reference_teacher_candidate'].startswith('D_') for c in cases),
                  'FULL_D_conditional_training_candidates':sum(c['split']==s and c['conditional_original_training_candidate'].startswith('D_') for c in cases)}
              for s in ['TRAIN','VALIDATION']}
    report = {
      'schema':'V71_INDEPENDENT_YCB_FROZEN_CUE_REVIEW_V1','completed':True,'status':'PASS_REPLAY_WITH_LIMITED_CONDITIONAL_TEACHER_SUPPORT',
      'source_only':True,'builder_or_sanity_helper_imports':0,
      'bindings':[binding(DATA/x) for x in ['FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json','CUE_GEOMETRY_SEAL.json','SELECTION_LOCK.json','INPUTS_RGB_ONLY.jsonl','SOURCE_LINEAGE_GT_ONLY.jsonl','FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json']]+[binding(__file__)],
      'source_raster_bindings':read_bindings,'replayed_cases':12,'input_rows':24,'mismatches':0,
      'original_q8_L_box_unchanged':True,'nearest_query_row_is_lineage_only':True,
      'GT_or_reference_in_input_whitelist':False,'native8_semantic_gold_count':0,'human_gold_count':0,
      'counts_by_split':counts,'total_reference_D_conditional_candidates':7,
      'total_FULL_D_conditional_original_candidates':2,'total_FULL_D_conditional_controlled_candidates':2,
      'controlled_E_candidates':0,'existing_paste_uncertainty_coverage_all_zero':True,
      'visual_review':{
        'reviewed':['WORKING_RAW64_1.png','WORKING_RAW64_2.png','WORKING_RAW64_3.png','FINE_RGB_ONLY_5PX_GRID.png','FROZEN_CUE_NATIVE64_SANITY_1.png','FROZEN_CUE_NATIVE64_SANITY_2.png','FROZEN_CUE_NATIVE64_SANITY_3.png'],
        'conclusion':'Localized RGB rims, edge turns and junctions are operational generic physical cues. The two FULL compatible VAL cues (pitcher lip and bottle foot turn) are visually observable conditional teacher candidates. This does not establish any canonical/native8 correspondence.',
        'prior_mask_exposure':True,'blind_gold':False,
        'mask_zero_limit':'Five ownership/boundary conflicts remain U. A mask-zero reference pixel does not prove semantic E/O/S; RGB proposals are preserved and no geometry was moved after sanity.',
        'mask_positive_limit':'The seven raster-compatible references are not established D by mask membership alone; RGB localization is the separate observational evidence.'},
      'execution_limit':'At unchanged q0, TRAIN has zero FULL compatible D teachers and VAL has two; neither a trained semantic dataset nor full goal resolution follows.',
      'separate_grid_feasibility':{
        'feasible_as_data_preparation':True,'actual_generation_in_this_audit':False,
        'requirements':['Freeze deterministic 9x9 q coordinates/order/hash from box and unchanged L before reference/mask/state access; preserve all 81 candidates.',
                        'Keep original q0/q8 and results immutable. Label availability only in GT-only records; keep PARTIAL/MISS/U and effective-input reference collisions.',
                        'Maintain RGB conditional-teacher tier; FULL is availability, not visibility proof. Disclose GT-jitter source-box lineage.',
                        'Call it a source-emulated proposal distribution, not actual native initializer performance; canonical/native8 identity and deployment association remain unresolved.']},
      'cases':cases,'new_model_forward_training_current_PnP':0,'features_or_weights_read':0,
      'old_artifacts_modified':False,'semantic_training_or_current_admission':False}
    out = ROOT/'source_audit/INDEPENDENT_YCB_CUE_REVIEW.json'
    assert not out.exists(),out
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':binding(out),'counts':counts,'mismatches':0},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
