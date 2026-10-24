"""Independent saved source-grid replay; no builder/helper/model imports."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / 'data/ycb_proposal_grid'
CUE = ROOT / 'data/ycb_cue_review'


def bind(p):
    p = Path(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def bound(b):
    assert bind(b['path'])['sha256'] == b['sha256'], b['path']


def scalar_crop(rgb, q, side):
    """Explicit independent four-neighbor scalar interpolation at each output site."""
    h,w = rgb.shape[:2]
    out = np.zeros((3,96,96),np.uint8)
    valid = np.zeros((1,96,96),bool)
    for oy in range(96):
        sy = q[1] + ((oy+.5)/96-.5)*side
        for ox in range(96):
            sx = q[0] + ((ox+.5)/96-.5)*side
            if not (-.5 <= sx < w-.5 and -.5 <= sy < h-.5):
                continue
            valid[0,oy,ox] = True
            ix,iy = math.floor(sx),math.floor(sy)
            dx,dy = sx-ix,sy-iy
            accum = np.zeros(3,np.float64)
            for xx,yy,weight in [(ix,iy,(1-dx)*(1-dy)),(ix+1,iy,dx*(1-dy)),(ix,iy+1,(1-dx)*dy),(ix+1,iy+1,dx*dy)]:
                if 0 <= xx < w and 0 <= yy < h:
                    accum += rgb[yy,xx].astype(np.float64)*weight
            out[:,oy,ox] = np.floor(np.clip(accum,0,255)+.5).astype(np.uint8)
    return out,valid


def main():
    result = read(GRID/'GRID_RESULT.json')
    for b in result['bindings'].values():
        bound(b)
    lock = read(GRID/'GRID_INPUT_LOCK.json')
    extraction = read(GRID/'INPUT_EXTRACTION_SEAL.json')
    protocol = read(GRID/'PROTOCOL.json')
    index = lines(GRID/'INPUT_INDEX_LINEAGE_ONLY.jsonl')
    frozen_grid = lines(GRID/'FROZEN_QUERY_GRID_INPUT_METADATA.jsonl')
    truth = lines(GRID/'GT_ONLY_GRID_TEACHERS.jsonl')
    source = lines(CUE/'INPUTS_RGB_ONLY.jsonl')
    props = {r['target_id']:r for r in read(CUE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json')['proposals']}
    sanity = {r['target_id']:r for r in read(CUE/'FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json')['cases']}
    assert lock['query_derived_from_box_only'] and lock['cue_GT_mask_state_not_parsed']
    assert lock['actual_RGB_reads_before_grid_lock'] == extraction['GT_arrays_read_before_input_seal'] == 0
    assert len(index) == len(frozen_grid) == len(truth) == 1944 and len(source) == 24
    arrays = {}
    with np.load(GRID/'INPUT_ONLY.npz',allow_pickle=False) as z:
        assert set(z.files) == {'rgb96','crop_valid96','query_valid','query_xy','box_xyxy','L'}
        for k in z.files:
            arrays[k] = z[k]
    assert arrays['rgb96'].shape == (1944,3,96,96) and arrays['rgb96'].dtype == np.uint8
    assert arrays['crop_valid96'].shape == (1944,1,96,96) and arrays['crop_valid96'].dtype == bool
    assert arrays['query_valid'].shape == arrays['L'].shape == (1944,)
    assert arrays['query_xy'].shape == (1944,2) and arrays['box_xyxy'].shape == (1944,4)
    groups_full,groups_crop = defaultdict(list),defaultdict(list)
    replay = []
    for i,(r,g,t) in enumerate(zip(index,frozen_grid,truth)):
        assert r['input_index'] == g['input_index'] == t['input_index'] == i
        assert {k:v for k,v in r.items() if k != 'decoded_fullRGB_sha256'} == g
        view = source[i//81]
        row,col = divmod(i%81,9)
        assert r['view_index'] == i//81 and row == r['grid_row'] == t['grid_row'] and col == r['grid_col'] == t['grid_col']
        assert r['view_id'] == view['id'] == t['view_id']
        assert r['source_lineage_only'] == view['source'] == t['source_lineage_only']
        assert r['split'] == view['split'] == t['split']
        assert r['image'] == view['image'] and r['image_sha256'] == view['image_sha256']
        x0,y0,x1,y1 = view['box']
        q = [x0+(col+.5)*(x1-x0)/9, y0+(row+.5)*(y1-y0)/9]
        side = view['side']
        assert q == r['query_xy'] == arrays['query_xy'][i].tolist()
        assert side == r['L'] == arrays['L'][i]
        assert view['box'] == r['box_xyxy'] == arrays['box_xyxy'][i].tolist()
        h,w = view['raw_hw']
        px,py = math.floor(q[0]+.5),math.floor(q[1]+.5)
        valid = all(math.isfinite(x) for x in q) and 0 <= q[0] < w and 0 <= q[1] < h and 0 <= px < w and 0 <= py < h
        assert valid == bool(arrays['query_valid'][i]) == r['native_query_valid_from_metadata']
        p,s = props[r['original_target_id']],sanity[r['original_target_id']]
        assert p['reference_xy'] == t['reference_xy'] and p['uncertainty_bbox_xyxy'] == t['uncertainty_bbox_xyxy']
        assert p['cue_refID'] == t['cue_refID'] and t['native8_semantic_ID'] == 'UNKNOWN'
        assert not t['model_input_contains_this_GT']
        bb = p['uncertainty_bbox_xyxy']
        crop = [max(-.5,q[0]-side/2), max(-.5,q[1]-side/2), min(w-.5,q[0]+side/2),min(h-.5,q[1]+side/2)]
        contained = all([crop[0]<=bb[0],crop[1]<=bb[1],crop[2]>=bb[2],crop[3]>=bb[3]])
        intersects = max(crop[0],bb[0]) < min(crop[2],bb[2]) and max(crop[1],bb[1]) < min(crop[3],bb[3])
        avail = 'INVALID' if not valid else 'FULL' if contained else 'PARTIAL' if intersects else 'MISS'
        base = 'D' if s['original_cue_teacher_candidate'].startswith('D_') else 'U'
        target = 'D' if base == 'D' and avail == 'FULL' else 'U'
        assert s['existing_controlled_paste_covered_uncertainty_pixels'] == 0
        assert avail == t['crop_availability'] and base == t['source_conditional_state'] and target == t['training_state']
        replay.append({'split':view['split'],'source':view['source'],'state':target,'base':base,'availability':avail})
        params = arrays['query_xy'][i].tobytes()+arrays['box_xyxy'][i].tobytes()+np.asarray(side,np.float64).tobytes()
        groups_full[hashlib.sha256(bytes.fromhex(r['decoded_fullRGB_sha256'])+params).hexdigest()].append(i)
        groups_crop[hashlib.sha256(arrays['rgb96'][i].tobytes()+arrays['crop_valid96'][i].tobytes()+np.asarray(valid,bool).tobytes()+params).hexdigest()].append(i)
    def summarize(rr):
        return {'N':len(rr),'source_conditional_states':dict(Counter(r['base'] for r in rr)),
                'availability':dict(Counter(r['availability'] for r in rr)),
                'training':dict(Counter(r['state'] for r in rr))}
    counts = summarize(replay)
    splits = {s:summarize([r for r in replay if r['split']==s]) for s in ['TRAIN','VALIDATION']}
    sources = {s:summarize([r for r in replay if r['source']==s]) for s in ['YCB_ORIGINAL','YCB_CONTROLLED']}
    assert counts == result['counts'] and splits == result['by_split'] and sources == result['by_view_source']
    identities = {}
    for name,groups in [('same_fullRGB_query_L_box',groups_full),('same_effectivecrop_valid_query_L_box',groups_crop)]:
        stats = dict(groups=len(groups),duplicate_groups=sum(len(v)>1 for v in groups.values()),duplicate_rows=sum(len(v) for v in groups.values() if len(v)>1))
        old = result['identity_audits'][name]
        assert all(stats[k] == old[k] for k in stats)
        conflicts = sum({'D','E'} <= {truth[i]['training_state'] for i in ids} for ids in groups.values())
        mixed = sum(len({truth[i]['training_state'] for i in ids})>1 for ids in groups.values())
        assert conflicts == mixed == 0 and old['strict_D_E_conflicts'] == old['D_U_mixed_knowledge_groups'] == []
        identities[name] = dict(**stats,strict_D_E_conflicts=conflicts,D_U_mixed_knowledge_groups=mixed)
    # Fixed source-metadata-only sample: two SHA-smallest targets per split, same grid sites in both views.
    orig_views = [v for v in source if v['source']=='YCB_ORIGINAL']
    selected = []
    for split in ['TRAIN','VALIDATION']:
        selected += sorted([v for v in orig_views if v['split']==split],key=lambda v:hashlib.sha256(v['id'].encode()).hexdigest())[:2]
    crops = []
    for j,v in enumerate(selected):
        k = [0,40,80,18][j]
        for kind in ['YCB_ORIGINAL','YCB_CONTROLLED']:
            i = next(r['input_index'] for r in index if r['original_target_id']==v['id'] and r['source_lineage_only']==kind and r['grid_row']*9+r['grid_col']==k)
            meta = index[i]
            assert bind(meta['image'])['sha256'] == meta['image_sha256']
            rgb = np.asarray(Image.open(meta['image']).convert('RGB'))
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == meta['decoded_fullRGB_sha256']
            out,val = scalar_crop(rgb,meta['query_xy'],meta['L'])
            rgb_bad = int(np.count_nonzero(out != arrays['rgb96'][i]))
            valid_bad = int(np.count_nonzero(val != arrays['crop_valid96'][i]))
            assert rgb_bad == valid_bad == 0
            crops.append({'input_index':i,'target_id':v['id'],'source':kind,'split':v['split'],'grid_index':k,'RGB_channels_mismatched':rgb_bad,'valid_cells_mismatched':valid_bad,'RGB_binding':bind(meta['image'])})
    report = {'schema':'V71_INDEPENDENT_YCB_GRID_AUDIT_V1','completed':True,'status':'PASS','mismatches':0,
      'bindings':[bind(GRID/'GRID_RESULT.json'),bind(__file__)]+list(result['bindings'].values()),
      'all_rows_replayed':1944,'input_keys':sorted(arrays),'NPZ_shapes':{k:list(v.shape) for k,v in arrays.items()},
      'GT_input_fields':0,'input_index_and_grid_and_GT_identity_mismatches':0,'GT_independent_q_formula_replay_mismatches':0,
      'original_L_and_box_preserved':True,'original_q8_results_unchanged':{'TRAIN_D':0,'VALIDATION_D':2},
      'coverage_and_source_teacher_replay':counts,'by_split':splits,'by_view_source':sources,'identity_replay':identities,
      'RGB_crop_sample_rule':'Two SHA256(target_id)-smallest original targets per split; grid index [0,40,80,18] in selected order, both original and controlled; no score/GT selection.',
      'scalar_RGB_crop_samples':crops,'sample_RGB_crops':8,'sample_RGB_pixel_centers':8*96*96,'sample_RGB_channels':8*96*96*3,
      'scalar_RGB_channel_mismatches':0,'scalar_crop_valid_mismatches':0,
      'guard_code_review':'freeze uses input box/L and hashes later GT files without parsing content; extract validates lock then reads only input metadata/RGB; label opens fixed proposals/sanity only after verifying INPUT_EXTRACTION_SEAL. Historical phase ordering is supported by frozen code/seals, not claimed as independent system-call tracing.',
      'interpretation':'D42 includes original and neutral controlled views (21 each), not 42 independent cues/images; TRAIN22 are 11 distinct q placements over four admitted source cues, VAL20 are ten q placements over three admitted source cues. They are operational generic cue teachers with native8 UNKNOWN, not actual native initializer performance, human gold, causal-E dataset, or completed semantic task.',
      'remaining_limitations':['Seven admitted generic real object cues are not canonical pallet native8 annotations.','No E examples are present; existing cutout never covers any frozen uncertainty footprint.','All 81 queries/view remain in the denominator; FULL/conditional-D enrichment must not be called real initialization accuracy.','Source box/q provenance includes the original approximate-box proposal construction; grid is a new source-emulated distribution.'],
      'builder_or_GT_helper_imports':0,'source_feature_or_weights_read':0,'new_model_forward_fit_current_PnP':0,'existing_artifacts_modified':False}
    dest=ROOT/'source_audit/INDEPENDENT_YCB_GRID_AUDIT.json'
    assert not dest.exists(),dest
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':bind(dest),'counts':counts,'by_split':splits,'sample_RGB_crops':8,'mismatches':0},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
