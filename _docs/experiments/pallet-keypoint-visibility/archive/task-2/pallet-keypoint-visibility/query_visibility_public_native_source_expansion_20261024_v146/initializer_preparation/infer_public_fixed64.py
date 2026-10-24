"""V146 explicit new-phase RGB-only R0 on64 canonical public frames.

Thin immutable V141 loop reuse: counts/output names and lossless image loader
only. Fresh V146 policy/selection/transport/root entry authorize this phase.
No old failed-output/repair policy participates in authorization.
"""
from pathlib import Path
import argparse,ast,difflib,hashlib,importlib.util,inspect,json,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_PUBLIC64_NATIVE_EXPANSION_POLICY_BEFORE_NEW_BODIES.json';POLICY_SHA='592ab0be3c8635cb75cbbd93c3457daf463892e7dfe34c5029c63213746d247e'
SELECTION=V/'FIXED64_SELECTION_BEFORE_NEW_SOURCE_BODIES.json'
SELECTION_SHA='cf33c72e0d5f547d0ecfcf8849fbbfd30beda80724587bfef9972e9227aa26f2'
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OLD=P/'query_visibility_public_native_initializer_association_20261024_v141'
BASE=OLD/'initializer_preparation/infer_public_fixed16.py';BASE_SHA='5537924777a5398b7ec19bb7ed9f414b27f25a1c66103f415a7c23ebea70a16e'
SUCCESS=OLD/'initializer_canonical_transport_preparation/PROTOCOL_BEFORE_SOURCE_FORWARD_V2.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FORWARD.json';INPUTS=H/'INPUTS64_RGB_ONLY.jsonl';GENERATED=H/'GENERATED_BOUNDARY_AND_CANONICAL_CHECKS.json'
OUT=V/'actual_public_initializer_v1';TRANSPORT=V/'canonical_RGB_transport_v1/TRANSPORT_COMPLETE.json'
TRANSPORT_SHA='d9d19aaf1276ecf680ce76cfe773b53898d7e33596e17713e3995c4a2cd5ed17'
INPUT_KEYS={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder','canonical_RGB_NPY'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)==BASE_SHA
spec=importlib.util.spec_from_file_location('v146_immutable_R0_helpers',BASE);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
bind,read,check,write,lines,same=b.bind,b.read,b.check,b.write,b.lines,b.same

def validate_transport(entry,p,rows):
    assert Path(entry['canonical_transport_seal']['path'])==TRANSPORT
    seal=read(check(entry['canonical_transport_seal']));assert seal['completed']and seal['PASS']and seal['frames']==64 and seal['lossless']
    assert same(seal['selection'],p['selection'])and same(seal['policy'],p['policy'])
    manifest=read(check(seal['manifest']));arrays=manifest['rows'];assert len(arrays)==len(rows)==64
    for original,array in zip(rows,arrays):
        for key in('frame_index','id','source','source_id','split','raw_hw'):assert original[key]==array[key]
        assert array==original
    return arrays

def array_to_bgr(rgb,row):
    assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']
    return np.ascontiguousarray(rgb[:,:,::-1])

def adapted():
    before=inspect.getsource(b.run);source=before
    # These are frame/slot/predict-count constants only; native_grid is reused untouched.
    tree=ast.parse(source)
    class Counts(ast.NodeTransformer):
        def visit_keyword(self,node):
            node.arg={'all16_and128_retained':'all64_and512_retained','all16_frames128_native_slots_preserved':'all64_frames512_native_slots_preserved'}.get(node.arg,node.arg)
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if type(node.value)is int and node.value in(16,128,18):return ast.copy_location(ast.Constant({16:64,128:512,18:66}[node.value]),node)
            if isinstance(node.value,str):
                node.value=node.value.replace('ALL16','ALL64').replace('QUERY128','QUERY512').replace('all16_and128_retained','all64_and512_retained').replace('all16_frames128_native_slots_preserved','all64_frames512_native_slots_preserved')
            return node
    source=ast.unparse(ast.fix_missing_locations(Counts().visit(tree)))+'\n'
    old="rows = lines(INPUTS)\n    assert len(rows) == 64 and all((set(r) == INPUT_KEYS and r['frame_index'] == i for (i, r) in enumerate(rows)))"
    assert source.count(old)==1;source=source.replace(old,old+"\n    validate_transport(e,p,rows)")
    tree=ast.parse(source);node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='image')
    old='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
    new="""    def image(row):
        assert sha(row['image'])==row['image_sha256']
        rgb=np.load(check(row['canonical_RGB_NPY']),allow_pickle=False)
        return array_to_bgr(rgb,row)"""
    assert source.count(old)==1;source=source.replace(old,new)
    source=source.replace("'v141_exact_R0_factory'","'v146_exact_R0_factory'")
    old="inputs=p['inputs'], outputs=outputs"
    assert source.count(old)==1;source=source.replace(old,"inputs=p['inputs'], canonical_transport_seal=e['canonical_transport_seal'], outputs=outputs")
    return before,source

ORIGINAL,EFFECTIVE=adapted();ns=dict(vars(b));ns.update(__file__=__file__,H=H,V=V,P=P,POLICY=POLICY,POLICY_SHA=POLICY_SHA,PROTOCOL=PROTOCOL,GENERATED=GENERATED,INPUTS=INPUTS,INPUT_KEYS=INPUT_KEYS,OUT=OUT,validate_transport=validate_transport,array_to_bgr=array_to_bgr)
exec(compile(EFFECTIVE,str(H/'EFFECTIVE_PUBLIC64_RUN.py'),'exec'),ns);run=ns['run']

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(SELECTION)==SELECTION_SHA and sha(TRANSPORT)==TRANSPORT_SHA
    selection=bind(SELECTION);transport=read(TRANSPORT)
    assert transport['completed']and transport['PASS']and transport['frames']==64 and same(transport['selection'],selection)and same(transport['policy'],bind(POLICY))
    manifest=read(check(transport['manifest']));rr=manifest['rows'];assert len(rr)==64
    rows=[dict(r)for r in rr]
    assert all(set(r)==INPUT_KEYS and r['source']=='PALLOX'for r in rows)and len({r['source_id']for r in rows})==64
    assert sorted({r['source_id'].split(':')[0]for r in rows})==sorted(['0.55m_hall','1.23m_hall','0.55m_lab','1.23m_lab'])
    assert all(sum(r['source_id'].startswith(s+':')for r in rows)==16 for s in{r['source_id'].split(':')[0]for r in rows})
    assert {f:sum(r['split']==f for r in rows)for f in('TRAIN_LAB','VAL_HALL')}=={'TRAIN_LAB':32,'VAL_HALL':32}
    rgb=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3);r=dict(raw_hw=[4,6],decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest());assert np.array_equal(array_to_bgr(rgb,r)[:,:,::-1],rgb)
    altered=rgb.copy();altered[0,0,0]^=1
    try:array_to_bgr(altered,r)
    except AssertionError:pass
    else:raise AssertionError('Altered RGB accepted')
    with INPUTS.open('x')as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    effective=H/'EFFECTIVE_PUBLIC64_RUN.py';effective.write_text(EFFECTIVE)
    diff=H/'EXACT_V141_TO_PUBLIC64_DIFF.patch';diff.write_text(''.join(difflib.unified_diff(ORIGINAL.splitlines(True),EFFECTIVE.splitlines(True),fromfile='V141_RUN',tofile='V146_PUBLIC64_RUN')))
    previous=read(SUCCESS)
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),original_boundary_proof=previous['generated'],invented_RGB_roundtrip_and_one_byte_guard=True,
        immutable_query_arrays_native_grid_and_factory=True,framecount64_slotcount512_predictcalls66=True,no_old_failure_repair_authorization=True,actual_RGB_checkpoint_model_or_GT_reads=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),selection=selection,generated=bind(GENERATED),inputs=bind(INPUTS),base_code=bind(BASE),successful_loader_protocol=bind(SUCCESS),
        **{k:previous[k]for k in('reference_code','reference_protocol','backbone_code','checkpoint','native_grid','runtime')},
        canonical_transport_seal=bind(TRANSPORT),canonical_manifest=transport['manifest'],frames=64,native_slots=512,output=str(OUT),effective_run=bind(effective),run_diff=bind(diff),
        input_fields=sorted(INPUT_KEYS),no_pose_K_reference_labels_masks_to_model=True,new_phase_root_policy_only=True,automatic_retry=False,RGB_checkpoint_body_reads_during_prepare=0)
    write(PROTOCOL,p)
    template=dict(approved_actual_public_initializer=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],canonical_transport_seal=p['canonical_transport_seal'],**{k:False for k in b.FORBIDDEN})
    path=H/'ROOT_INITIALIZER_ENTRY_TEMPLATE.json';write(path,template)
    print(json.dumps(dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),inputs=bind(INPUTS),template=bind(path))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and a.entry is None:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use inert --prepare or fresh root --entry')
