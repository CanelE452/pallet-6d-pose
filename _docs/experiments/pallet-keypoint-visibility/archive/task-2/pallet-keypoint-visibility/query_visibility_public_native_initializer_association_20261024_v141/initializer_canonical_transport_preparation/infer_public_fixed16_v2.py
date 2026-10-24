"""Explicit manual V141 revision: only checked canonical lossless RGB loader changes.

V1 failed output stays immutable. A separate root entry must bind a completed
canonical transport seal; this code never retries V1 or redefines its RGB SHAs.
"""
from pathlib import Path
import argparse,ast,difflib,hashlib,importlib.util,inspect,json,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent
BASE=V/'initializer_preparation/infer_public_fixed16.py';BASE_SHA='5537924777a5398b7ec19bb7ed9f414b27f25a1c66103f415a7c23ebea70a16e'
BASE_PROTOCOL=BASE.parent/'PROTOCOL_BEFORE_SOURCE_FORWARD.json'
POLICY=V/'ROOT_SOURCE_INITIALIZER_ASSOCIATION_POLICY_BEFORE_FORWARD.json'
REPAIR=V/'ROOT_SOURCE_TRANSPORT_REPAIR_POLICY_BEFORE_EXPORT.json';REPAIR_SHA='3f74ef6133c429a85e5bbe57ae70c2df5c4a506ccf192690c0ee6387fdc64278'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FORWARD_V2.json';GENERATED=H/'GENERATED_LOSSLESS_LOADER_CHECKS.json'
OUT=V/'actual_public_initializer_canonical_transport_v2'
TRANSPORT=V/'canonical_RGB_transport_v1/TRANSPORT_COMPLETE.json'
TRANSPORT_PROTOCOL=V/'canonical_transport_preparation/PROTOCOL_BEFORE_CANONICAL_RGB_TRANSPORT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)==BASE_SHA
spec=importlib.util.spec_from_file_location('v141_immutable_v1_helpers',BASE);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
bind,read,check,write=b.bind,b.read,b.check,b.write


def validate_transport(entry,p,rows):
    assert sha(REPAIR)==REPAIR_SHA and entry['repair_policy']==p['repair_policy']==bind(REPAIR)
    binding=entry['canonical_transport_seal'];assert Path(binding['path'])==TRANSPORT
    seal=read(check(binding));assert seal['completed']and seal['PASS']and seal['frames']==16 and seal['lossless']and seal['all_original_RGB_expected_SHAs_and_HW_exact']
    assert seal['inputs']==p['inputs']and seal['protocol']==p['canonical_transport_protocol']and seal['failed_v1']==p['failed_v1']and seal['repair_policy']==p['repair_policy']
    manifest=read(check(seal['manifest']));assert manifest['source_inputs']==p['inputs']and manifest['all_expected_SHAs_unchanged']
    arrays=manifest['rows'];assert len(arrays)==len(rows)==16
    for original,array in zip(rows,arrays):
        for key in('frame_index','id','source','source_id','split','raw_hw'):assert original[key]==array[key],key
        assert array['original_image']==dict(path=original['image'],sha256=original['image_sha256'])
        assert array['decoded_RGB_sha256']==array['expected_decoded_RGB_sha256_unchanged']==original['decoded_RGB_sha256']
        check(array['canonical_RGB_NPY'])
    return arrays


def array_to_bgr(rgb,row):
    assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3]and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256']
    return np.ascontiguousarray(rgb[:,:,::-1])


def adapted():
    source=inspect.getsource(b.run);before=source
    old="rows=lines(INPUTS);assert len(rows)==16 and all(set(r)==INPUT_KEYS and r['frame_index']==i for i,r in enumerate(rows))"
    new=old+"\n    transport_rows=validate_transport(e,p,rows)"
    assert source.count(old)==1;source=source.replace(old,new)
    tree=ast.parse(source);function=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='image')
    old='\n'.join(source.splitlines()[function.lineno-1:function.end_lineno])
    new="""    def image(row):
        # Exact original expected RGB values, transported losslessly. No JPEG
        # decoder runs in the model environment; original input SHA unchanged.
        assert sha(row['image'])==row['image_sha256']
        array=transport_rows[row['frame_index']]
        rgb=np.load(check(array['canonical_RGB_NPY']),allow_pickle=False)
        return array_to_bgr(rgb,row)"""
    assert source.count(old)==1;source=source.replace(old,new)
    old="GT_pose_K_reference_semantic_mask_or_historic_score_reads=0,visibility_classifier_fit_private_V112_PnP_q_L_correction_calls=0,association_not_inferred=True,seconds=time.monotonic()-start)"
    new="GT_pose_K_reference_semantic_mask_or_historic_score_reads=0,visibility_classifier_fit_private_V112_PnP_q_L_correction_calls=0,association_not_inferred=True,canonical_transport_seal=e['canonical_transport_seal'],manual_revision_after_decoder_guard=True,expected_RGB_SHA_unchanged=True,seconds=time.monotonic()-start)"
    assert source.count(old)==1;source=source.replace(old,new)
    return before,source


ORIGINAL,EFFECTIVE=adapted();ns=dict(vars(b));ns.update(__file__=__file__,H=H,V=V,PROTOCOL=PROTOCOL,GENERATED=GENERATED,OUT=OUT,validate_transport=validate_transport,array_to_bgr=array_to_bgr)
exec(compile(EFFECTIVE,str(H/'EFFECTIVE_V2_RUN.py'),'exec'),ns);run=ns['run']


def prepare():
    assert sha(POLICY)==b.POLICY_SHA and sha(REPAIR)==REPAIR_SHA
    p=dict(read(BASE_PROTOCOL));assert p['code']==bind(BASE)
    assert (V/'actual_public_initializer_v1/FAILED_EXECUTION.json').exists()
    # Only invented RGB exercises lossless transport/strict value hash and BGR.
    rgb=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3);row=dict(raw_hw=[4,6],decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    assert np.array_equal(array_to_bgr(rgb,row)[:,:,::-1],rgb)
    bad=rgb.copy();bad[0,0,0]^=1
    try:array_to_bgr(bad,row)
    except AssertionError:pass
    else:raise AssertionError('Altered canonical RGB accepted')
    assert b.OUT!=OUT and b.PROTOCOL!=PROTOCOL
    effective=H/'EFFECTIVE_V2_RUN.py';effective.write_text(EFFECTIVE)
    diff=H/'EXACT_V1_TO_V2_DIFF.patch';diff.write_text(''.join(difflib.unified_diff(ORIGINAL.splitlines(True),EFFECTIVE.splitlines(True),fromfile='IMMUTABLE_FAILED_V1_RUN',tofile='CANONICAL_TRANSPORT_V2_RUN')))
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),base_code=bind(BASE),original_boundary_checks=p['generated'],invented_RGB_only=True,
        exact_lossless_RGB_to_BGR_roundtrip=True,one_byte_value_change_rejected=True,model_factory_selection_native_grid_runtime_unchanged=True,parent_globals_and_failed_output_not_modified=True,actual_RGB_checkpoint_model_forward_reads=0))
    p.update(code=bind(__file__),generated=bind(GENERATED),base_code=bind(BASE),base_protocol=bind(BASE_PROTOCOL),repair_policy=bind(REPAIR),output=str(OUT),
        failed_v1=bind(V/'actual_public_initializer_v1/FAILED_EXECUTION.json'),failed_v1_started=bind(V/'actual_public_initializer_v1/STARTED.json'),
        canonical_transport_protocol=bind(TRANSPORT_PROTOCOL),expected_completed_transport_seal=str(TRANSPORT),effective_run=bind(effective),run_diff=bind(diff),
        decoders='No pinned JPEG decode. All16 exact expected canonical RGB values loaded from checked lossless NPY; original inputmanifest/SHAs/folds unchanged.',
        manual_root_approved_revision=True,automatic_retry=False,original_failed_v1_preserved=True,legacy2_comparison_after_forward_only=True)
    write(PROTOCOL,p)
    template=dict(approved_actual_public_initializer=False,code=p['code'],policy=p['policy'],repair_policy=p['repair_policy'],protocol=bind(PROTOCOL),generated=p['generated'],inputs=p['inputs'],
        canonical_transport_seal=dict(path=str(TRANSPORT),sha256='FILL_COMPLETED_TRANSPORT_SHA256'),**{k:False for k in b.FORBIDDEN})
    path=H/'ROOT_INITIALIZER_V2_ENTRY_TEMPLATE.json';write(path,template)
    print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(path),effective_run=bind(effective),diff=bind(diff),actual_RGB_checkpoint_model_forward_reads=0)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use inert --prepare or separate manual root v2 --entry')
