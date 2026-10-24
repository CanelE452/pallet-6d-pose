"""Independent actual-source data replay, without producer/model/helper imports."""
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
DATA=ROOT/'data/ycb_reference_cover'
V71=PARENT/'query_visibility_reference_jitter_20261023_v71'
OLD=V71/'data/ycb_proposal_grid'
CUE=V71/'data/ycb_cue_review'


def bind(p):
    p=Path(p)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']


def load_rgb(b):
    check(b)
    return np.asarray(Image.open(b['path']).convert('RGB'))


def maximal_square_dp(mask):
    """Different algorithm from producer's integral-image binary search."""
    h,w=mask.shape
    dp=np.zeros((h+1,w+1),np.int32)
    largest=0
    for y in range(h):
        for x in range(w):
            if mask[y,x]:
                dp[y+1,x+1]=1+min(dp[y,x+1],dp[y+1,x],dp[y,x])
                largest=max(largest,int(dp[y+1,x+1]))
    ends=np.argwhere(dp==largest)
    choices=[]
    for ey,ex in ends:
        x,y=int(ex-largest),int(ey-largest)
        distance=(y+(largest-1)/2-(h-1)/2)**2+(x+(largest-1)/2-(w-1)/2)**2
        choices.append((distance,y,x))
    _,y,x=min(choices)
    return [x,y,largest]


def region_count(mask,left,top,bb):
    n=0
    for y in range(math.ceil(bb[1]),math.ceil(bb[3])):
        for x in range(math.ceil(bb[0]),math.ceil(bb[2])):
            if 0<=y-top<mask.shape[0] and 0<=x-left<mask.shape[1]:
                n+=int(mask[y-top,x-left])
    return n


def scalar_crop(rgb,q,L):
    h,w=rgb.shape[:2]
    out=np.zeros((3,96,96),np.uint8);valid=np.zeros((1,96,96),bool)
    for oy in range(96):
        sy=q[1]+((oy+.5)/96-.5)*L
        for ox in range(96):
            sx=q[0]+((ox+.5)/96-.5)*L
            if not(-.5<=sx<w-.5 and -.5<=sy<h-.5):continue
            valid[0,oy,ox]=True
            x,y=math.floor(sx),math.floor(sy);dx,dy=sx-x,sy-y;value=np.zeros(3,np.float64)
            for xx,yy,weight in [(x,y,(1-dx)*(1-dy)),(x+1,y,dx*(1-dy)),(x,y+1,(1-dx)*dy),(x+1,y+1,dx*dy)]:
                if 0<=xx<w and 0<=yy<h:value+=rgb[yy,xx].astype(np.float64)*weight
            out[:,oy,ox]=np.floor(np.clip(value,0,255)+.5).astype(np.uint8)
    return out,valid


def main():
    receipt=read(DATA/'DATASET_RECEIPT.json');protocol=read(DATA/'COMPOSITION_PROTOCOL.json')
    check(receipt['protocol']);check(receipt['code'])
    for b in receipt['outputs']+receipt['galleries']+protocol['bindings']:check(b)
    certs=read(DATA/'INTERVENTION_CERTIFICATES_GT_ONLY.json')['variants']
    rows=lines(DATA/'INPUT_INDEX_LINEAGE_ONLY.jsonl');gt=lines(DATA/'GT_ONLY_TEACHERS.jsonl')
    oi=lines(OLD/'INPUT_INDEX_LINEAGE_ONLY.jsonl');og=lines(OLD/'GT_ONLY_GRID_TEACHERS.jsonl')
    lineage={r['id']:r for r in lines(CUE/'SOURCE_LINEAGE_GT_ONLY.jsonl')}
    props={r['target_id']:r for r in read(CUE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json')['proposals']}
    sanity={r['target_id']:r for r in read(CUE/'FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json')['cases']}
    pool={r['id']:r for r in lines(PARENT/'query_visibility_controlled_real_20261022_v66/provenance/OCCLUDER_POOL.jsonl')}
    keys={'rgb96','crop_valid96','query_valid','query_xy','box_xyxy','L'}
    with np.load(DATA/'MODEL_INPUT_ONLY.npz',allow_pickle=False) as z:
        assert set(z.files)==keys
        data={k:z[k] for k in z.files}
    with np.load(OLD/'INPUT_ONLY.npz',allow_pickle=False) as z:
        assert set(z.files)==keys
        assert all(np.array_equal(data[k][:1944],z[k]) for k in keys)
    assert len(rows)==len(gt)==3645 and len(certs)==21
    assert data['rgb96'].shape==(3645,3,96,96) and data['rgb96'].dtype==np.uint8
    assert data['crop_valid96'].shape==(3645,1,96,96) and data['crop_valid96'].dtype==bool
    assert len({r['row_id'] for r in rows})==3645
    for i in range(1944):
        assert all(rows[i][k]==v for k,v in oi[i].items())
        assert all(gt[i][k]==v for k,v in og[i].items())
    certified={};rgb_cache={};donor_cache={};raster_checks=[]
    for c in certs:
        ident=c['target_id'];p=props[ident];orig=lineage[ident]['original'];d=c['donor']
        assert d==pool[d['id']]
        assert d['split']==orig['split']==c['split']
        assert d['object_id']!=orig['object_id'] and d['image']['sha256']!=orig['image']['sha256']
        assert d['scene_id'] in (range(8) if c['split']=='TRAIN' else [8,9])
        assert c['reference_xy']==p['reference_xy'] and c['uncertainty_bbox_xyxy']==p['uncertainty_bbox_xyxy']
        assert sanity[ident]['original_cue_teacher_candidate'].startswith('D_')
        assert not c['native_donor_resize'] and c['source_queries_L_box_unchanged']
        if ident not in rgb_cache:rgb_cache[ident]=load_rgb(orig['image'])
        original=rgb_cache[ident];h,w=original.shape[:2]
        if d['id'] not in donor_cache:
            drgb=load_rgb(d['image']);check(d['visible'])
            dm=np.asarray(Image.open(d['visible']['path']))>0
            yy,xx=np.where(dm);db=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            dc=drgb[db[1]:db[3],db[0]:db[2]];mc=dm[db[1]:db[3],db[0]:db[2]]
            donor_cache[d['id']]=(dc,mc,db,maximal_square_dp(mc))
        dc,mc,db,square=donor_cache[d['id']]
        assert db==c['donor_crop_native_bbox'] and square==c['donor_largest_opaque_square_xyK']
        bb=p['uncertainty_bbox_xyxy'];uw=math.ceil(bb[2])-math.ceil(bb[0]);uh=math.ceil(bb[3])-math.ceil(bb[1])
        sx,sy,K=square;assert K>=max(uw,uh)+2
        fl=math.ceil(bb[0])-sx-(K-uw)//2;ft=math.ceil(bb[1])-sy-(K-uh)//2
        variant=c['variant'];left,top=c['certificate']['integer_translation_xy'];dh,dw=mc.shape
        if variant=='FULL_COVER':assert (left,top)==(fl,ft)
        elif variant=='PARTIAL_COVER':
            choices=[]
            for dx in range(-(dw+uw),dw+uw+1):
                n=region_count(mc,fl+dx,ft,bb)
                if 0<n<uw*uh:choices.append((abs(2*n-uw*uh),abs(dx),dx))
            chosen=min(choices)[2]
            assert (left,top)==(fl+chosen,ft)
        else:
            candidates=[(l,t) for t in [0,(h-dh)//2,h-dh] for l in [0,(w-dw)//2,w-dw] if region_count(mc,l,t,bb)==0]
            assert (left,top)==candidates[0]
        # Reconstruct every output pixel by scattering the native visible donor.
        alpha=np.zeros((h,w),bool);expected=original.copy()
        dy,dx=np.where(mc);ty,tx=dy+top,dx+left;inside=(ty>=0)&(ty<h)&(tx>=0)&(tx<w)
        alpha[ty[inside],tx[inside]]=True
        expected[ty[inside],tx[inside]]=dc[dy[inside],dx[inside]]
        actual=load_rgb(c['image']);check(c['GT_alpha'])
        stored=np.asarray(Image.open(c['GT_alpha']['path']))
        assert set(np.unique(stored)).issubset({0,255})
        assert np.array_equal(stored>0,alpha) and np.array_equal(actual,expected)
        covered=int(alpha[math.ceil(bb[1]):math.ceil(bb[3]),math.ceil(bb[0]):math.ceil(bb[2])].sum())
        assert covered==c['certificate']['uncertainty_opaque_cells']
        assert uw*uh==c['certificate']['uncertainty_total_cells']
        assert int(np.any(actual!=original,axis=2).sum())==c['certificate']['changed_RGB_pixels']>0
        assert covered==uw*uh if variant=='FULL_COVER' else covered==0 if variant=='NEUTRAL_D' else 0<covered<uw*uh
        state={'FULL_COVER':'E','PARTIAL_COVER':'U','NEUTRAL_D':'D'}[variant]
        assert state==c['reference_state']
        assert len(c['input_indices'])==81
        certified[c['image']['path']]=c
        raster_checks.append(dict(target_id=ident,split=c['split'],variant=variant,donor_id=d['id'],donor_object_id=d['object_id'],
                                  alpha_native_pixels=int(alpha.sum()),uncertainty_covered=covered,uncertainty_total=uw*uh,
                                  full_RGB_and_alpha_replay_mismatches=0,donor_square_algorithm='independent dynamic programming',
                                  image=c['image'],GT_alpha=c['GT_alpha']))
        rgb_cache[c['image']['path']]=actual
    count_rows=[];pair_supervised=Counter();fullgroups=defaultdict(list);effectivegroups=defaultdict(list)
    for i,(r,t) in enumerate(zip(rows,gt)):
        assert i==r['input_index']==t['input_index'] and r['row_id']==t['row_id']
        p=props[r['original_target_id']];orig=lineage[r['original_target_id']]['original']
        row,col=r['grid_row'],r['grid_col'];b=orig['box'];L=orig['side']
        q=[b[0]+(col+.5)*(b[2]-b[0])/9,b[1]+(row+.5)*(b[3]-b[1])/9]
        assert q==r['query_xy']==data['query_xy'][i].tolist() and L==r['L']==data['L'][i]
        assert b==r['box_xyxy']==data['box_xyxy'][i].tolist()
        assert p['reference_xy']==t['reference_xy'] and p['uncertainty_bbox_xyxy']==t['uncertainty_bbox_xyxy']
        assert p['cue_refID']==r['cue_group']==t['cue_refID']
        assert r['split']==t['split']==orig['split']
        assert t['native8_semantic_ID']=='UNKNOWN' and not t['model_input_contains_this_GT'] and t['reference_or_GT_is_not_model_input']
        h,w=r['raw_hw'];px,py=math.floor(q[0]+.5),math.floor(q[1]+.5)
        valid=all(math.isfinite(v) for v in q) and 0<=q[0]<w and 0<=q[1]<h and 0<=px<w and 0<=py<h
        assert valid==r['native_query_valid_from_metadata']==bool(data['query_valid'][i])
        bb=p['uncertainty_bbox_xyxy'];crop=[max(-.5,q[0]-L/2),max(-.5,q[1]-L/2),min(w-.5,q[0]+L/2),min(h-.5,q[1]+L/2)]
        full=crop[0]<=bb[0] and crop[1]<=bb[1] and crop[2]>=bb[2] and crop[3]>=bb[3]
        overlap=min(crop[2],bb[2])>max(crop[0],bb[0]) and min(crop[3],bb[3])>max(crop[1],bb[1])
        availability='INVALID' if not valid else 'FULL' if full else 'PARTIAL' if overlap else 'MISS'
        assert availability==t['availability']==t['crop_availability']
        state=og[i]['source_conditional_state'] if i<1944 else certified[r['image']]['reference_state']
        target=state if state in ['D','E'] and availability=='FULL' else 'U'
        assert state==t['reference_state']==t['source_conditional_state'] and target==t['training_state']
        if i>=1944:
            c=certified[r['image']];assert r['image_sha256']==c['image']['sha256']
            assert i in c['input_indices'] and r['variant']==c['variant'] and r['split']==c['split']
            paired=rows[r['paired_original_input_index']]
            assert paired['variant']=='BASE_ORIGINAL' and paired['original_target_id']==r['original_target_id']
            assert paired['query_xy']==q and paired['L']==L and paired['box_xyxy']==b and paired['split']==r['split']
            if target in ['D','E']:pair_supervised[(r['split'],r['variant'])]+=1
        count_rows.append(dict(reference_state=state,availability=availability,training_state=target,split=r['split'],variant=r['variant']))
        params=data['query_xy'][i].tobytes()+data['box_xyxy'][i].tobytes()+np.asarray(L,np.float64).tobytes()
        fullgroups[hashlib.sha256(bytes.fromhex(r['decoded_fullRGB_sha256'])+params).hexdigest()].append(i)
        effectivegroups[hashlib.sha256(data['rgb96'][i].tobytes()+data['crop_valid96'][i].tobytes()+np.asarray(valid,bool).tobytes()+params).hexdigest()].append(i)
    def counts(rr):
        return dict(N=len(rr),reference_states={k:sum(r['reference_state']==k for r in rr) for k in ['D','E','U']},
                    availability={k:sum(r['availability']==k for r in rr) for k in ['FULL','PARTIAL','MISS','INVALID']},
                    training={k:sum(r['training_state']==k for r in rr) for k in ['D','E','U']})
    total=counts(count_rows);assert total==receipt['counts']
    bysplit={s:counts([r for r in count_rows if r['split']==s]) for s in ['TRAIN','VALIDATION']}
    assert bysplit==receipt['by_split']
    byvariant={s:counts([r for r in count_rows if r['variant']==s]) for s in receipt['by_variant']}
    assert byvariant==receipt['by_variant']
    identity={};reported=read(DATA/'INPUT_CONFLICT_AUDIT.json')['audits']
    for name,groups in [('full_RGB_query_L_box',fullgroups),('effective_crop_valid_query_L_box',effectivegroups)]:
        sem=sum({'D','E'}<={gt[i]['reference_state'] for i in ix} for ix in groups.values())
        strict=sum({'D','E'}<={gt[i]['training_state'] for i in ix} for ix in groups.values())
        identity[name]=dict(groups=len(groups),duplicate_groups=sum(len(ix)>1 for ix in groups.values()),reference_D_E_collision_groups=sem,supervised_D_E_conflicts=strict)
        assert len(groups)==reported[name]['groups'] and identity[name]['duplicate_groups']==reported[name]['duplicate_groups']
        assert sem==len(reported[name]['reference_D_E_collision_groups']) and strict==len(reported[name]['supervised_D_E_conflicts'])==0
    # One actual FULL crop per new variant, chosen deterministically before looking at scores (none exist).
    crop_checks=[]
    for c in certs:
        candidates=[i for i in c['input_indices'] if gt[i]['crop_availability']=='FULL']
        i=min(candidates);r=rows[i]
        actual=rgb_cache[c['image']['path']]
        assert hashlib.sha256(actual.tobytes()).hexdigest()==r['decoded_fullRGB_sha256']
        out,valid=scalar_crop(actual,r['query_xy'],r['L'])
        assert np.array_equal(out,data['rgb96'][i]) and np.array_equal(valid,data['crop_valid96'][i])
        crop_checks.append(dict(input_index=i,target_id=c['target_id'],variant=c['variant'],split=c['split'],training_state=gt[i]['training_state'],RGB_channel_mismatches=0,valid_mismatches=0))
    # No source/donor photograph crosses TRAIN/VAL in this dataset.
    hashes=defaultdict(set)
    for r in rows:hashes[r['split']].add(r['image_sha256'])
    for c in certs:hashes[c['split']].add(c['donor']['image']['sha256'])
    assert not hashes['TRAIN'] & hashes['VALIDATION']
    report=dict(schema='V72_INDEPENDENT_ACTUAL_YCB_REFERENCE_COVER_AUDIT_V1',completed=True,status='PASS',mismatches=0,
      bindings=[bind(DATA/'DATASET_RECEIPT.json'),bind(DATA/'COMPOSITION_PROTOCOL.json'),bind(__file__)]+receipt['outputs'],
      old1944_all_six_arrays_exact_prefix=True,old1944_index_GT_fields_preserved=True,model_input_whitelist=sorted(keys),GT_reference_mask_alpha_ID_source_offset_model_fields=0,
      all_rows_geometry_label_availability_replayed=3645,all81_queries_preserved_per_view=True,all21_native_RGB_alpha_and_placement_replayed=True,
      counts=total,by_split=bysplit,by_variant=byvariant,identity_audit=identity,native_donor_replays=raster_checks,
      scalar_RGB96_crop_checks=crop_checks,scalar_RGB96_count=21,scalar_RGB_channel_count=21*96*96*3,scalar_RGB_and_valid_mismatches=0,
      source_donor_split_crossing_RGB_SHAs=0,allFULL_DE_claims_recomputed=True,
      gallery_visual_review=dict(reviewed=receipt['galleries'],all7_viewed=True,
        conclusion='All7 galleries show a fixed RGB cue, opaque full cover, partial cover kept U, and neutral placement preserving the cue. No immediate semantic-label or GT-overlay leakage error observed in actual source rasters/crops.',
        limits=['Gallery r-centered native64 zooms are diagnostic and not model crops. Actual crop inputs were separately rebuilt from unmarked RGB.',
                'The donor introduces other visible rims/corners: observing some generic cue is not equivalent to observing the intended fixed target cue. Reference association remains unresolved.',
                'Tiny repeated cue/view examples and hard pasted-object/context artifacts permit memorization or shortcuts; this is not natural-occlusion or canonical native8 gold.']),
      effective_hidden_reference_collision_handling='Preserved as diagnostics; all identical-input supervised D/E conflicts zero. This finite audit does not establish global semantic identifiability.',
      human_or_native8_gold=False,conditional_AI_teacher_only=True,new_model_forward_fit_current_PnP=0,
      producer_or_model_or_sampler_imports=0,features_weights_read=0,immutable_producer_artifacts_modified=False,model_fit_or_current_authorization=False)
    dest=ROOT/'source_audit/INDEPENDENT_YCB_REFERENCE_COVER_AUDIT.json';assert not dest.exists()
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=bind(dest),counts=total,by_split=bysplit,identity=identity,RGB_crops=21,mismatches=0),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
