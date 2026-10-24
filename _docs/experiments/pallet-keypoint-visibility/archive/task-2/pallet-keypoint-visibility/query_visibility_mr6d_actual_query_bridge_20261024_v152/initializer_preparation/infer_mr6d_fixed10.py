"""V152 inert fixed10 R0 adapter: exact V146 loop, count/identity bindings only."""
from pathlib import Path
from collections import Counter
import argparse,ast,difflib,hashlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_POLICY_BEFORE_MODEL_OR_REFERENCE_BRIDGE.json';POLICY_SHA='a941442d93e0a49554be86716fd324a4ad3bed01d629abb2d2ed77212bce7270'
OLD=P/'query_visibility_public_native_source_expansion_20261024_v146/initializer_preparation'
OLD_RUN=OLD/'EFFECTIVE_PUBLIC64_RUN.py';OLD_RUN_SHA='83358868f3c98ac0de5b196bb8cdd859049b750b1b5dc1ce7b25e87700a8d335'
BASE=P/'query_visibility_public_native_initializer_association_20261024_v141/initializer_preparation/infer_public_fixed16.py'
BASE_SHA='5537924777a5398b7ec19bb7ed9f414b27f25a1c66103f415a7c23ebea70a16e'
TRANSPORT=V/'canonical_RGB_transport_v1/TRANSPORT_COMPLETE.json';TRANSPORT_SHA='fe5baa392860728ad9a9399ae07e65ff6ff36c35b43857d19f4b965462f4f33f'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FORWARD.json';INPUTS=H/'INPUTS10_RGB_ONLY.jsonl';GENERATED=H/'GENERATED_BOUNDARY_AND_LOSSLESS_CHECKS.json'
OUT=V/'actual_mr6d_initializer_v1'
INPUT_KEYS={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder','canonical_RGB_NPY'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)==BASE_SHA and sha(OLD_RUN)==OLD_RUN_SHA
spec=importlib.util.spec_from_file_location('v152_exact_R0_helpers',BASE);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
bind,read,check,write,lines,same=b.bind,b.read,b.check,b.write,b.lines,b.same

def validate_transport(e,p,rows):
    assert e['canonical_transport_seal']==p['canonical_transport_seal'] and Path(e['canonical_transport_seal']['path'])==TRANSPORT
    seal=read(check(e['canonical_transport_seal']));assert seal['completed']and seal['PASS']and seal['frames']==10 and seal['lossless']
    assert seal['policy']==p['policy']and seal['selection']==p['selection']and seal['canonical_decoder']=='PINNED_CV2_IMREAD_COLOR_BGR_TO_RGB'
    manifest=read(check(seal['manifest']));assert manifest['rows']==rows and len(rows)==10
    assert lines(check(seal['model_inputs']))==rows

def array_to_bgr(rgb,row):
    assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']
    return np.ascontiguousarray(rgb[:,:,::-1])

ORIGINAL=OLD_RUN.read_text();tree=ast.parse(ORIGINAL);changes=Counter()
class Counts(ast.NodeTransformer):
    def visit_keyword(self,node):
        node.arg={'all64_and512_retained':'all10_and80_retained','all64_frames512_native_slots_preserved':'all10_frames80_native_slots_preserved'}.get(node.arg,node.arg)
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if type(node.value)is int and node.value in(64,512,66):
            changes[node.value]+=1;return ast.copy_location(ast.Constant({64:10,512:80,66:12}[node.value]),node)
        if isinstance(node.value,str):node.value=node.value.replace('ALL64','ALL10').replace('QUERY512','QUERY80').replace('v146_exact_R0_factory','v152_exact_R0_factory')
        return node
EFFECTIVE=ast.unparse(ast.fix_missing_locations(Counts().visit(tree)))+'\n'
assert set(changes)=={64,512,66}
ns=dict(vars(b));ns.update(__file__=__file__,H=H,V=V,P=P,POLICY=POLICY,POLICY_SHA=POLICY_SHA,PROTOCOL=PROTOCOL,GENERATED=GENERATED,
    INPUTS=INPUTS,INPUT_KEYS=INPUT_KEYS,OUT=OUT,validate_transport=validate_transport,array_to_bgr=array_to_bgr)
exec(compile(EFFECTIVE,str(H/'EFFECTIVE_MR6D10_RUN.py'),'exec'),ns);run=ns['run']

def generated():
    # Pure original native_grid/query serializer; invented arrays only, no RGB/model.
    base=np.array([[90.+11*k,120.+3*k]for k in range(9)],np.float64)
    def candidate(score,q):return dict(candidate_index=0,score=score,box_xyxy=np.array([1.,2.,3.,4.]),keypoints_xy=q,keypoints_conf=np.full(9,.75))
    cases=0
    for xy in([-.5,0.],[-.25,10.],[0.,0.],[1279.49,719.49],[1279.5,10.],[1279.75,10.],[-100.,-100.],[1280.,720.]):
        q=base.copy();q[0]=xy;z=dict(candidates=[candidate(.8,q),candidate(.8,q+30)],selected_index=0,input_shape=(512,640),canvas_shape=(920,1480),added_border=100)
        qa=b.query_arrays(z,[720,1280]);assert np.array_equal(qa['query_xy'],q[:8])
        finite=np.isfinite(q[:8]).all(1)&~(q[:8]==[-1,-1]).all(1)
        expectedL=np.clip(.25*np.ptp(q[:8][finite],axis=0).max(),16,1280);assert (qa['L']==expectedL).all()
        assert qa['continuous_valid'][0]==(0<=xy[0]<1280 and 0<=xy[1]<720)
        assert qa['nearest_valid'][0]==(0<=np.floor(xy[0]+.5)<1280 and 0<=np.floor(xy[1]+.5)<720)
        assert np.array_equal(qa['nearest_valid'],qa['RGBcentre_valid']);cases+=1
    no=b.query_arrays(dict(candidates=[],selected_index=None),[720,1280]);assert np.isnan(no['query_xy']).all()and np.isnan(no['L']).all()and not no['native_grid_valid'].any()
    original=b.query_arrays(dict(candidates=[candidate(.8,base)],selected_index=0),[720,1280]);q=base.copy();q[8]=[1e8,-1e8]
    cent=b.query_arrays(dict(candidates=[candidate(.8,q)],selected_index=0),[720,1280]);assert all(np.array_equal(original[k],cent[k],equal_nan=True)for k in original)
    q=base.copy();q[0]=[np.nan,15.];q[1]=[-1,-1];q[2]=[np.inf,3.]
    bad=b.query_arrays(dict(candidates=[candidate(.8,q)],selected_index=0),[720,1280]);assert not bad['finite_non_sentinel'][:3].any()and not bad['native_grid_valid'][:3].any()
    assert b.clean(q)[0]==[None,15.]and b.clean(q)[2]==[None,3.]
    rgb=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3);row=dict(raw_hw=[4,6],decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    assert array_to_bgr(rgb,row)[:,:,::-1].tobytes()==rgb.tobytes();altered=rgb.copy();altered[0,0,0]^=1
    try:array_to_bgr(altered,row)
    except AssertionError:pass
    else:raise AssertionError('Altered canonical byte accepted')
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_only=True,boundary_cases=cases,firstargmax_tie_preserved=True,
        centroid_excluded_from_q_L=True,no_detection_all8_and_nonfinite_sentinel_OOB_preserved=True,continuous_nearest_RGBcentre_9x9_distinct=True,
        canonical_RGB_BRG_roundtrip_and_one_byte_guard=True,actual_source_RGB_checkpoint_GT_model_body_reads_or_calls=0))

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(TRANSPORT)==TRANSPORT_SHA
    previous=read(OLD/'PROTOCOL_BEFORE_SOURCE_FORWARD.json');assert previous['effective_run']['sha256']==OLD_RUN_SHA and previous['base_code']['sha256']==BASE_SHA
    transport=read(TRANSPORT);assert transport['completed']and transport['PASS']and transport['frames']==10 and transport['policy']==bind(POLICY)
    rr=read(check(transport['manifest']))['rows'];assert len(rr)==10
    assert all(set(r)==INPUT_KEYS and r['frame_index']==i and r['source']=='MR6D'and r['split']=='SOURCE_DEVELOPMENT'for i,r in enumerate(rr))
    assert len({r['source_id']for r in rr})==10 and not any('/000001:'in r['source_id']for r in rr)
    generated()
    with INPUTS.open('x')as f:
        for row in rr:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    effective=H/'EFFECTIVE_MR6D10_RUN.py';effective.write_text(EFFECTIVE)
    diff=H/'EXACT_V146_TO_MR6D10_DIFF.patch';diff.write_text(''.join(difflib.unified_diff(ORIGINAL.splitlines(True),EFFECTIVE.splitlines(True),fromfile='V146_RUN',tofile='V152_RUN')))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),selection=transport['selection'],generated=bind(GENERATED),inputs=bind(INPUTS),
        **{k:previous[k]for k in('reference_code','reference_protocol','backbone_code','checkpoint','native_grid','runtime')},
        original_run=bind(OLD_RUN),original_protocol=bind(OLD/'PROTOCOL_BEFORE_SOURCE_FORWARD.json'),helper_code=bind(BASE),canonical_transport_seal=bind(TRANSPORT),canonical_manifest=transport['manifest'],
        frames=10,native_slots=80,output=str(OUT),effective_run=bind(effective),run_diff=bind(diff),input_fields=sorted(INPUT_KEYS),
        only_run_differences='Count64/512/66 to10/80/12; output field/file identity; new namespace/policy/input seal. Native_grid/factory/firstargmax/forward operator unchanged.',
        AST_count_constant_replacements={str(k):v for k,v in changes.items()},no_pose_K_reference_labels_masks_depth_or_historic_predictions_to_model=True,
        original_official_VAL_preserved_in_source_id=True,source_development_only=True,no_reselection_or_q_L_correction=True,
        RGB_checkpoint_model_body_reads_during_this_prepare=0,actual_forward=False,automatic_retry=False)
    write(PROTOCOL,p)
    template=dict(approved_actual_public_initializer=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],canonical_transport_seal=p['canonical_transport_seal'],**{k:False for k in b.FORBIDDEN})
    t=H/'ROOT_INITIALIZER_ENTRY_TEMPLATE.json';write(t,template)
    ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),inputs=bind(INPUTS),template=bind(t),effective_run=bind(effective),diff=bind(diff),canonical_transport_seal=bind(TRANSPORT),learned_forward_calls=0)
    write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and a.entry is None:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use inert --prepare or fresh root --entry')
