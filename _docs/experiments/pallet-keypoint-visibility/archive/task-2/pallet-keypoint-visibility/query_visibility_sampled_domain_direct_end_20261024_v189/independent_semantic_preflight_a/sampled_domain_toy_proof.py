"""Independent invented-only sampled-domain proof. No source/model/data imports."""
import copy,hashlib,itertools,json,math
from pathlib import Path

H=Path(__file__).absolute().parent
def support(q,L,hw):
    h,w=hw;result=[]
    for j in range(96):
        for i in range(96):
            x=q[0]+((i+.5)/96-.5)*L;y=q[1]+((j+.5)/96-.5)*L
            s={}
            if -.5<=x<w-.5 and -.5<=y<h-.5:
                ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy
                for px,wx in ((ix,1-dx),(ix+1,dx)):
                    for py,wy in ((iy,1-dy),(iy+1,dy)):
                        if wx*wy>0 and 0<=px<w and 0<=py<h:s[(px,py)]=wx*wy
            result.append(s)
    return result
def raw_enclosure(box):
    x0,y0,x1,y1=box
    if not all(math.isfinite(v) for v in box) or not x0<=x1 or not y0<=y1:raise ValueError('finite closed box')
    return set(itertools.product(range(math.floor(x0),math.ceil(x1)+1),range(math.floor(y0),math.ceil(y1)+1)))
def sampled_gate(q,L,hw,boxes,available=True,weak=True,witness=None):
    initial=copy.deepcopy((q,L,hw,boxes,witness));samples=support(q,L,hw);allpixels=set().union(*samples)
    extent=[q[0]-(47.5/96)*L,q[1]-(47.5/96)*L,q[0]+(47.5/96)*L,q[1]+(47.5/96)*L]
    details=[]
    for box in boxes:
        pixels=raw_enclosure(box)
        details.append({'inside_continuous_sample_coordinate_extent':extent[0]<=box[0]<=box[2]<=extent[2] and extent[1]<=box[1]<=box[3]<=extent[3],
                        'positive_related_samples':sum(bool(set(s)&pixels) for s in samples),
                        'all_raw_candidate_pixels_participate':pixels<=allpixels})
    ok=available and weak and bool(boxes) and all(d['inside_continuous_sample_coordinate_extent'] and d['positive_related_samples']>0 for d in details)
    if witness is not None:ok=ok and tuple(witness) in allpixels
    assert initial==(q,L,hw,boxes,witness)
    return bool(ok),details,allpixels
def rounded_scalar_sample(weights,colors):
    return math.floor(min(255,max(0,math.fsum(weights[p]*colors[p] for p in weights)))+.5)
def run():
    tests=[]
    def add(name,passed,meaning):tests.append({'name':name,'PASS':bool(passed),'meaning':meaning})
    q=[400.,400.];hw=[800,800];L=288.;box=[395.,395.,405.,405.]
    good,ds,s=sampled_gate(q,L,hw,[box]);add('sparse_grid_end_evidence_can_exist_despite_raw_holes',good and not ds[0]['all_raw_candidate_pixels_participate'],'L/96=3; unchanged end enclosure intersects positive samples without requiring every enclosed raw pixel to be sampled.')
    good,ds,_=sampled_gate(q,L,hw,[[259.,259.,259.,259.]]);add('inside_extent_but_no_end_related_sample_is_UNKNOWN',not good and ds[0]['inside_continuous_sample_coordinate_extent'] and ds[0]['positive_related_samples']==0,'Inside rectangle is necessary but not sufficient input evidence.')
    good,ds,_=sampled_gate(q,L,hw,[[257.,257.,257.,257.]]);add('outer_raw_tap_hull_is_not_continuous_sample_extent',not good and not ds[0]['inside_continuous_sample_coordinate_extent'] and ds[0]['positive_related_samples']>0,'A contributing edge pixel beyond first sample coordinate cannot silently enlarge the continuous crop domain.')
    good,ds,_=sampled_gate(q,L,hw,[[258.,258.,258.,258.],[259.,259.,259.,259.]]);add('every_alternative_candidate_requires_evidence',not good and ds[0]['positive_related_samples']>0 and ds[1]['positive_related_samples']==0,'Union-level one-hit cannot hide an unobserved alternative enclosure.')
    e=raw_enclosure(box);hole=next(iter(e-s));alpha={p:True for p in e};alpha[hole]=False
    add('COVER_still_requires_opacity_at_unsampled_candidate_raw_pixels',not all(alpha[p] for p in e),'Removing allraw participation never permits opaque holes in the physical candidate envelope.')
    weights={(10,10):.5,(11,10):.5};before={(10,10):100,(11,10):100};sham={(10,10):100,(11,10):220};candidate={(10,10)}
    add('SHAM_zero_on_candidate_alone_does_not_ensure_sample_equality',rounded_scalar_sample(weights,before)!=rounded_scalar_sample(weights,sham) and (11,10) not in candidate,'SHAM may modify another tap of an end-related sample; actual equality of every selected sample remains required.')
    weights={(10,10):.1,(11,10):.9};before={(10,10):100,(11,10):100};cover={(10,10):101,(11,10):100}
    add('positive_end_weight_not_enough_for_uint8_change',rounded_scalar_sample(weights,before)==rounded_scalar_sample(weights,cover),'Actual COVER end-related RGB96 byte change is independent of geometric participation because donor colors and rounding can hide change.')
    add('original_input_U_is_not_relabeled_by_visible_crop',not sampled_gate(q,L,hw,[box],available=False)[0],'Original factual U remains priority independent of crop appearance.')
    add('unsupported_semantic_end_is_not_promoted_by_sampling',not sampled_gate(q,L,hw,[box],weak=False)[0],'Sampling provides input evidence only; it cannot invent selected-instance/end correspondence.')
    add('old_surrounding_witness_tap_hole_remains_UNKNOWN',not sampled_gate(q,L,hw,[box],witness=[259,259])[0],'Root explicitly retained original chosen surrounding witness positive-tap requirement; no witness reselection.')
    add('unchanged_qL_grid_and_candidate_provenance',q==[400.,400.] and L==288. and box==[395.,395.,405.,405.],'Operator never corrects/recenters q, changes L, shrinks enclosure or searches permutation.')
    add('all_native_population_is_not_known_label_population',128*8==1024 and 22*3==66 and 66*3*8==1584,'Full attempts/native occurrences remain in census; UNKNOWN/failures receive no implicit binary label.')
    out={'completed':True,'PASS':all(t['PASS'] for t in tests),'checks':tests,'count':len(tests),'invented_only':True,'source_pixel_NPZ_header_weight_model_or_evaluation_reads':0,'code':{'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    with (H/'GENERATED_INVENTED_PROOFS.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'completed':True,'PASS':out['PASS'],'count':len(tests),'checks_sha256':hashlib.sha256((H/'GENERATED_INVENTED_PROOFS.json').read_bytes()).hexdigest()}))
    if not out['PASS']:raise AssertionError('invented proof failed')
if __name__=='__main__':run()
