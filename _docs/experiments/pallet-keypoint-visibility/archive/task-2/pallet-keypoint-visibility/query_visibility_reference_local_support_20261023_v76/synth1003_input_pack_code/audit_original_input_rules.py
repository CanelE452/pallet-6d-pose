"""Input-only post-build exact legacy-box/validity QA, AST code extraction.

Does not import the private adapter or open any current inputs/GT/RGB/weights.
Does not change a frozen pack field or source selection.
"""
from pathlib import Path
import ast,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=ROOT/'synth1003_input_pack'
ADAPTER=ROOT.parent/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p))
def load(p):return json.loads(p.read_text())
def check(b):assert sha(b['path'])==b['sha256']
def pure(path,name):
    n=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)and x.name==name)
    env={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),env)
    return env[name],hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()

entry=load(HERE/'INPUT_PACK_BUILD_ENTRYLOCK.json');check(entry['protocol']);check(entry['code'])
protocol=load(Path(entry['protocol']['path']))
for b in protocol['bindings']:check(b)
manifest=load(OUT/'PACK_MANIFEST.json');seal=load(OUT/'PACK_SEAL.json');check(seal['manifest']);check(seal['lineage'])
rows=[json.loads(x)for x in Path(protocol['bindings'][0]['path']).read_text().splitlines()]
boxrule,boxast=pure(ADAPTER,'initial_query_box');validrule,validast=pure(ADAPTER,'query_validity')
context,ctxast=pure(Path(protocol['bindings'][3]['path']),'input_context_features')
framecount=0;modelvalid=0;cachedvalid=0;different=0;no_box=0
for c in manifest['chunks']:
    check(c['file'])
    with np.load(c['file']['path'],allow_pickle=False)as z:
        for j in range(c['end']-c['start']):
            i=c['start']+j;r=rows[i];q=np.asarray(r['unchanged_initial_query9'],float);hw=r['raw_RGB']['raw_hw']
            box=boxrule(q,hw);hasbox=box is not None;expected=np.zeros(4,float)if box is None else box
            assert np.array_equal(expected,z['query_box_xyxy'][j])and hasbox==bool(z['target_context_valid'][j])
            center=(expected[:2]+expected[2:])/2 if hasbox else np.zeros(2,float)
            side=float(1.25*np.max(expected[2:]-expected[:2]))if hasbox else 0.
            assert np.array_equal(center,z['target_crop_center_xy'][j])and side==float(z['target_crop_side'][j])
            ctx=np.stack([context(q[:8],k,expected,hw)for k in range(8)])
            assert np.array_equal(ctx,z['context42'][j])
            valid,pixels=validrule(q,hw,z['query_valid_cached'][j])
            # Native center must be inside the cached native9 ROI. Allactual
            # 1003 frames satisfy this, so cached-any adds no extra exclusion.
            assert np.array_equal(valid,z['query_center_valid'][j])
            assert np.array_equal(valid,z['query_valid_cached'][j]&z['pure_continuous_query_valid'][j]&z['nearest_pixel_valid'][j])
            finite=np.isfinite(q[:8]).all(1)&~(q[:8]==-1).all(1)
            purecontinuous=finite&(q[:8,0]>=0)&(q[:8,0]<hw[1])&(q[:8,1]>=0)&(q[:8,1]<hw[0])
            assert np.array_equal(purecontinuous,z['pure_continuous_query_valid'][j])and np.array_equal(purecontinuous,z['ROI_center_valid'][j])
            assert np.array_equal(z['ROI_anyvalid'][j],z['query_valid_cached'][j])
            framecount+=1;no_box+=not hasbox;modelvalid+=int(valid.sum());cachedvalid+=int(z['query_valid_cached'][j].sum())
            different+=int(np.sum(valid!=z['query_valid_cached'][j]))
assert framecount==1003
out=OUT/'ORIGINAL_BOX_AND_QUERY_VALIDITY_QA_ADDENDUM.json';assert not out.exists()
out.write_text(json.dumps(dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=entry['protocol'],pack_seal=bind(OUT/'PACK_SEAL.json'),
    legacy_adapter_code_only=bind(ADAPTER),ASTfunctions={'initial_query_box':boxast,'query_validity':validast},input_context_AST=ctxast,
    functions_extracted_without_import_or_current_inputs=True,all1003_box_side_center_context42_exact=True,frames=framecount,missing_target_boxes=no_box,
    original_cached_ROI_anyvalid_slots=cachedvalid,legacy_model_query_valid_slots=modelvalid,cached_vs_legacy_center_definition_different=different,
    legacy_model_query_validity_equals_pack_query_center_valid=True,ROI_center_equals_purecontinuous=True,ROI_any_equals_originalcached=True,
    clarification='V2fromoriginal8q means sameV74box minmax then clip lo>=0/hi<=W-1,H-1,span>=2; finiteoutsideq retained beforeboxclip. NoqueryorLclip/recenter. PackoriginalcachedROIanyvalid is separate fromcentervalid.',
    frozen_pack_or_entry_modified=False,RGB_GT_mask_annotation_reference_weight_reads=0,new_model_forward_fit_current_PnP=0),indent=2)+'\n')
print(json.dumps(bind(out)))
