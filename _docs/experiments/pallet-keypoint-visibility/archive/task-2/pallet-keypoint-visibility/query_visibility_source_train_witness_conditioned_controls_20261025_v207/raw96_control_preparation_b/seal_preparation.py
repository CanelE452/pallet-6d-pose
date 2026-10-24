"""Own text artifact sealing only; no source PNG/NPZ access."""
from pathlib import Path
import ast,hashlib,json,os,stat
H=Path(__file__).parent;V=H.parent;P=V.parent
S=V/'ROOT_PROSPECTIVE_RAW96_WEAK_WITNESS_CONTROL_PILOT_SCOPE.json'

def rb(p):
    p=Path(p);assert p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.parent==H or p==S
    assert p.suffix in {'.py','.json'}
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def put(n,d):
    with (H/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    scope=bind(S);assert scope['sha256']=='a61ee52c41e4ad8e8e21d514ec8801cdaf6d32863438cc8c084cc4cbf37cf342'
    names=['raw96_control_geometry.py','build_raw96_witness_controls.py','prepare_pilot_metadata.py','invented_checks.py','invented_loader_checks.py','seal_preparation.py']
    for n in names:ast.parse(rb(H/n))
    bindings={n:bind(H/n)for n in names}
    selection=bind(H/'FIXED8_X_TRAIN10_DONOR80_SELECTION.json');whitelist=bind(H/'EXACT26_PAYLOAD_WHITELIST.json')
    checks=[bind(H/n)for n in ('INVENTED_CHECKS.json','INVENTED_LOADER_CHECKS.json')]
    counts=[]
    for c in checks:
        d=json.loads(rb(c['path']));assert d['completed']and d['PASS']and d['invented_only'];counts.append(d['count'])
    protocol=dict(completed=True,prospective_before_source_PNG_or_compact_NPZ_payloads=True,scope=scope,
        code=bindings['build_raw96_witness_controls.py'],geometry=bindings['raw96_control_geometry.py'],selection=selection,payload_whitelist=whitelist,
        authority='POST_SAMPLER_RAW96_2D_WEAK_VISIBLE_FEATURE_CONTROL_FEASIBILITY_ONLY',
        population='All original SOURCE8 in original order x one stablehash donor per 10 original TRAIN object IDs = all80 attempts. No replacement, shrinking, q/L/native/U change, source/HOLD/score filtering or new labels.',
        payload_boundary='Exactly16 V206 encoded CLEAN96/VALID96 bytecopy PNGs plus10 compact rgb/alpha NPZs. Recorded paths/hashes fixed metadata-only. Never follow source RGB, GT, mask, nativealpha, original cache or fullframe bindings.',
        gate='Root actual entry approvedtrue + actual_payload_access_authorizedtrue + no_new_E_or_training_admissiontrue + exact scope/code/geometry/protocol/selection/whitelist/output_dir/80 bindings before ANY payload filesystem access. False template is non-executable.',
        transform='longside32,40,48,56,64,72,80 x CCW0,90,180,270; preserve aspect; floor(dim*scale+.5), min1. Pillow BILINEAR RGB, NEAREST alpha then false-outside3x3 erosion; rotate both equally. Retain all holes.',
        transform_order='Minimum SHA256 UTF8 V207_RAW96_TRANSFORM_V1|row_id|donor_id|longside|rotation; first transform with both legal COVER and SHAM. No RGB/model/label-quality transform search.',
        obligation='Exact union of both saved inclusive A/C rectangle pixel sets; not enclosing bbox or intersection. No q-center containment or minimumarea gate.',
        exhaustive_geometry='All integer top-left positions with full transformed rectangle inside96 and every positive alpha pixel valid. NumPy float64 FFT correlation counts, residual<1e-5 then exact integer round, scalar-fixture crosscheck; no scipy dependency.',
        placement_order='Each COVER/SHAM independently minimum SHA256 UTF8 V207_RAW96_PLACE_V1|row_id|donor_id|longside|rotation|COVER_or_SHAM|x|y. COVER alpha1 all union, SHAM alpha0 all union. Same RGB/alpha transform/mass, no clipping or alpha-hole filling.',
        actual_checks='Encoded binding, 96shape/dtype, exact recorded cutout array hashes; original invalid RGB zero; positive alpha allvalid; both ROI opacity, SHAM feature actualRGB unchanged, each A/C actual COVER RGB change>0; outside respective alpha exact; same pasted alpha mass. Changed RGB pixel mass recorded separately.',
        no_retry_policy='After first geometrically feasible transform/placements, an actual RGB-change failure is retained; no next-transform or donor retry. Exhausted, UNKNOWN and originalU remain explicit factual cases.',
        outputs='raw96_weak_witness_control_pilot_v1: all80 JSONL/CSV, original8 clean/valid bytecopies, constructed clean COVER/SHAM+alpha PNGs, fixed8 CONTACT and manifest, portable MD and HTML, COMPLETE or preserved FAILURE. No training arrays/targets/model.',
        preview='Per original case minimum V207_RAW96_ASSIGN_V1|row_id|donor_id hash among successfully constructed attempts; else factual plus failure. All8 shown, all80 retained, no outcome/score preference. Display-only A/C boxes and q47.5 cyan do not alter clean files.',
        caveats=['This is RAW96 post-sampler augmentation, separate from fullRGB-first intervention and selected41 synthetic GT diagnosis.',
            'Weak visible-feature rectangles may contain background or adjacent material and are not corrected keypoints, full part masks, physical nativegold or new E labels.',
            'Padding/coarse geometry can make the matched SHAM impossible; factual weakD remains unchanged and no source fit admission follows.'],
        dependencies='Pinned Python environment: numpy and Pillow exercised with invented arrays only; scipy not required.',
        checks=checks,invented_count=sum(counts),production_payload_stat_hash_header_decode=0,actual_intervention_started=False,fit_admission=False,new_native_E_label=False)
    put('PROTOCOL_BEFORE_RAW96_OR_DONOR_PAYLOADS.json',protocol)
    expected=dict(scope=scope,code=bindings['build_raw96_witness_controls.py'],geometry=bindings['raw96_control_geometry.py'],protocol=bind(H/'PROTOCOL_BEFORE_RAW96_OR_DONOR_PAYLOADS.json'),
        selection=selection,payload_whitelist=whitelist,output_dir=str(V/'raw96_weak_witness_control_pilot_v1'))
    put('ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved=False,actual_payload_access_authorized=False,no_new_E_or_training_admission=True,attempt_count=80,**expected))
    put('FINAL_READY.json',dict(completed=True,PASS=True,prepared_code_metadata_only=True,scope=scope,code=expected['code'],geometry=expected['geometry'],
        protocol=expected['protocol'],selection=selection,payload_whitelist=whitelist,source_metadata_preparation=bindings['prepare_pilot_metadata.py'],
        checks=checks,checks_code=[bindings[n]for n in ('invented_checks.py','invented_loader_checks.py')],invented_checks=sum(counts),
        entry_template=bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),cases=8,donors=10,attempts=80,PNG_bindings=16,NPZ_bindings=10,
        actual_payload_stat_hash_header_decode=0,actual_intervention_or_new_E_or_fit_admission=False,
        command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B '+str(H/'build_raw96_witness_controls.py')+' --entry ABSOLUTE_ROOT_APPROVED_ENTRY.json'))
    print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),code=expected['code'],geometry=expected['geometry'],protocol=expected['protocol'],selection=selection,checks=sum(counts))))

if __name__=='__main__':main()
