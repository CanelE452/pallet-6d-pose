"""Only invented scalar coordinates; frozen B pure math compared to A oracle."""
import hashlib,importlib.util,json,math
from pathlib import Path
H=Path(__file__).absolute().parent;V=H.parent
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=module(H/'sampled_domain_toy_proof.py','independent_A_toy')
Bpath=V/'end_locality_contract_b/finite_end_locality_contract.py'
assert hashlib.sha256(Bpath.read_bytes()).hexdigest()=='4948913619d23b8d8105c7d582511d939a4ec724062b3fee5ce2a982a91bb6bd'
B=module(Bpath,'frozen_B_pure_only')
cases=[('L384_holes',[256.,256.],384.,[512,512],[[255.,255.,261.,261.]]),
       ('unsampled_alternative',[256.,256.],384.,[512,512],[[255.,255.,261.,261.],[256.,256.,256.,256.]]),
       ('outside_extent',[256.,256.],384.,[512,512],[[447.,255.,447.,261.]]),
       ('fractional_positive',[256.25,256.25],384.,[512,512],[[258.5,258.5,258.6,258.6]]),
       ('padding_and_valid_taps',[-.25,-.25],12.,[31,29],[[0.,0.,1.,1.]]),
       ('closed_boundary',[256.,256.],384.,[512,512],[[66.,66.,66.,66.]]),
       ('F64_last_sample_not_symmetric_rounding',[256.125,256.875],191.9,[512,512],[[256.125+(95.5/96-.5)*191.9,256.875,256.125+(95.5/96-.5)*191.9,256.875]])]
checks=[]
for name,q,L,hw,boxes in cases:
    original_samples=A.support(q,L,hw)
    xx=[q[0]+((i+.5)/96-.5)*L for i in range(96)];yy=[q[1]+((i+.5)/96-.5)*L for i in range(96)]
    extent=[min(xx),min(yy),max(xx),max(yy)];rows=[]
    for box in boxes:
        e=A.raw_enclosure(box);rows.append({'inside':extent[0]<=box[0]<=box[2]<=extent[2] and extent[1]<=box[1]<=box[3]<=extent[3],
                                          'count':sum(any(p in e and w>0 for p,w in s.items()) for s in original_samples),
                                          'inframe':all(0<=x<hw[1] and 0<=y<hw[0] for x,y in e)})
    result=B.sampled_candidate_relation(q,L,hw,boxes)
    ok=result['original96_sample_coordinate_extent_xyxy']==extent and all(r['continuous_enclosure_inside_original96_sample_extent']==a['inside'] and r['positive_valid_RGB96_sample_intersections']==a['count'] and r['candidate_raw_footprint_in_frame']==a['inframe'] for r,a in zip(result['candidates'],rows))
    assert ok,name
    checks.append({'fixture':name,'PASS':True,'independent_extent':extent,'per_candidate':rows})
report={'completed':True,'PASS':True,'invented_only':True,'count':len(checks),'checks':checks,
        'frozen_B_code':{'path':str(Bpath),'sha256':hashlib.sha256(Bpath.read_bytes()).hexdigest()},
        'oracle_code':{'path':str(H/'sampled_domain_toy_proof.py'),'sha256':hashlib.sha256((H/'sampled_domain_toy_proof.py').read_bytes()).hexdigest()},
        'source_pixel_header_array_asset_hash_weights_models_or_evaluation_reads':0,
        'formula_note':'Operational extent uses exact min/max of original F64 q+((i+.5)/96-.5)*L; mathematical +/-47.5L/96 notation need not be numerically symmetric at the final ULP.'}
out=H/'GENERATED_FROZEN_B_SCALAR_COMPARISON.json'
with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'completed':True,'PASS':True,'count':len(checks),'receipt_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
