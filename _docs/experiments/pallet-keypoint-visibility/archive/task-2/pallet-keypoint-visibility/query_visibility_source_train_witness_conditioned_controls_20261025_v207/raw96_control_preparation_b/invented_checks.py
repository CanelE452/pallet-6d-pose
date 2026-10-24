"""Functional checks on invented arrays only; never instantiate source loaders."""
from pathlib import Path
import json,hashlib,copy
import numpy as np
import raw96_control_geometry as G

def run():
    checks=[]
    def ok(name,value):assert value,name;checks.append(name)
    def rejects(name,fn):
        try:fn()
        except (AssertionError,ValueError,TypeError):checks.append(name);return
        raise AssertionError(name)
    a=G.rectangle([1,1,2,2],(8,8));c=G.rectangle([4,4,4,4],(8,8));u=a|c
    ok('Inclusive_true_union_not_enclosing_bbox',int(u.sum())==5 and not u[3,3])
    alpha=np.ones((3,3),bool);alpha[1,1]=False;valid=np.ones((8,8),bool)
    covers,shams=G.feasible_positions(alpha,valid,G.rectangle([1,1,1,1],(8,8)))
    ok('Alpha_hole_is_not_filled',(0,0)not in covers and len(covers)>0)
    ok('SHAM_all_feature_pixels_disjoint',all(not u[y:y+3,x:x+3][alpha].any()for x,y in G.feasible_positions(alpha,valid,u)[1]))
    valid[6:]=False
    cp,sp=G.feasible_positions(np.ones((2,2),bool),valid,G.rectangle([1,1,1,1],(8,8)))
    ok('Padding_rejects_any_positive_alpha',(0,5)not in cp+sp and all(y<5 for x,y in cp+sp))
    rng=np.random.default_rng(37)
    for H,W,h,w in [(8,9,3,2),(6,6,1,1),(9,7,5,6),(4,4,4,4)]:
        im=rng.integers(0,2,(H,W)).astype(bool);k=rng.integers(0,2,(h,w)).astype(bool)
        got=G.correlation_counts(im,k)
        expected=np.array([[int((im[y:y+h,x:x+w]&k).sum())for x in range(W-w+1)]for y in range(H-h+1)])
        ok(f'FFT_all_origins_equal_scalar_{H}_{W}_{h}_{w}',np.array_equal(got,expected))
    solid=np.ones((2,2),bool);allv=np.ones((4,4),bool)
    ok('No_legal_SHAM_retained_no_ROI_shrink',not G.feasible_positions(solid,allv,allv)[1])
    case=dict(original_query_available=True,state_A='YES',state_C='YES',ROI_A=[1,1,2,2],ROI_C=[2,2,3,3],
        original_query_xy=[13.25,27.75],original_L=202.125,native_slot=7)
    original=copy.deepcopy(case);before=np.zeros((96,96,3),np.uint8);tex=np.full((4,4,3),91,np.uint8);al=np.ones((4,4),bool)
    cover,ca=G.paste(before,tex,al,(1,1));sham,sa=G.paste(before,tex,al,(20,20))
    proof=G.verify_actual(before,cover,sham,ca,sa,np.ones((96,96),bool),case)
    ok('Both_ROIs_opaque_SHAM_unchanged_mass_equal',proof['actual_change_PASS']and proof['pasted_alpha_pixels']==16 and np.array_equal(sham[ca],before[ca]))
    ok('Outside_alpha_and_qL_native_immutable',np.array_equal(cover[~ca],before[~ca])and case==original)
    rejects('Clipped_paste_is_rejected',lambda:G.paste(before,tex,al,(94,94)))
    corrupt=sham.copy();corrupt[2,2]=1
    rejects('SHAM_feature_change_rejected',lambda:G.verify_actual(before,cover,corrupt,ca,sa,np.ones((96,96),bool),case))
    cover0,ca0=G.paste(before,np.zeros_like(tex),al,(1,1))
    ok('Actual_change_failure_does_not_select_next_transform',not G.verify_actual(before,cover0,sham,ca0,sa,np.ones((96,96),bool),case)['actual_change_PASS'])
    unknown=copy.deepcopy(case);unknown['ROI_C']=None
    ok('UNKNOWN_no_center_or_ROI_imputation',G.obligation(unknown)==(None,'UNKNOWN_WITNESS'))
    unavailable=copy.deepcopy(case);unavailable['original_query_available']=False
    ok('Original_input_U_not_E',G.obligation(unavailable)==(None,'ORIGINAL_INPUT_U'))
    rgb=np.arange(9*7*3,dtype=np.uint8).reshape(9,7,3);aa=np.ones((9,7),bool);aa[4,3]=False
    rr,ar=G.transform(rgb,aa,32,90)
    ok('Aspect_rounding_rotation_no_hole_filling',rr.shape==(25,32,3)and ar.shape==(25,32)and not ar.all())
    poses=[(x,y)for y in range(3)for x in range(4)]
    p=G.choose_position(poses,'row','donor',32,0,'COVER')
    ok('Stable_position_independent_of_enumeration',p==G.choose_position(poses[::-1],'row','donor',32,0,'COVER'))
    ok('Transform_order_is_full28',len(set(G.ordered_transforms('row','donor')))==28)
    expected=dict(scope={'path':'scope','sha256':'a'*64},code={'path':'code','sha256':'b'*64},payload_whitelist={'path':'list','sha256':'c'*64},output_dir='output')
    entry=dict(approved=True,actual_payload_access_authorized=True,no_new_E_or_training_admission=True,attempt_count=80,**expected)
    ok('Exact_root_gate_accepts',G.validate_gate(entry,expected))
    false=copy.deepcopy(entry);false['approved']=False
    rejects('False_entry_rejected_before_payload',lambda:G.validate_gate(false,expected))
    drift=copy.deepcopy(entry);drift['payload_whitelist']['sha256']='d'*64
    rejects('Payload_whitelist_drift_rejected_before_payload',lambda:G.validate_gate(drift,expected))
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,
        source_payload_stat_hash_header_decode=0,source_observations_or_source_arrays=0,fit_or_model=0)

if __name__=='__main__':
    result=run();p=Path(__file__).parent/'INVENTED_CHECKS.json'
    assert 'v112'not in str(p).casefold()
    with p.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,count=result['count'],checks_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
