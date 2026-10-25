"""Invented arrays only: no source/donor/GT/model filesystem access."""
import hashlib,json
from pathlib import Path
import numpy as np
from compositor import compose_counterfactual_pair,_reference_taps,_resize


def run():
    yy,xx=np.indices((96,96));base=np.stack((xx,yy,(xx+yy)%256)).astype(np.uint8)
    valid=np.ones((96,96),bool);donor=np.zeros((24,42,3),np.uint8);donor[:]=[230,30,180]
    mask=np.zeros((24,42),bool);mask[2:22,3:39]=True
    q=np.array([100.,100.],np.float64);ref=q.copy();L=np.float64(96.)
    originals=[a.copy() for a in [base,valid,donor,mask,q,ref]]
    pair=compose_counterfactual_pair(base,valid,donor,mask,q,L,ref,'invented-centre-D')
    assert pair['success'] and not pair['masked']
    again=compose_counterfactual_pair(base,valid,donor,mask,q,L,ref,'invented-centre-D')
    for role in ['COVER','SHAM']:
        assert np.array_equal(pair[role],again[role]) and np.array_equal(pair['alpha_'+role],again['alpha_'+role])
        check=pair['teacherchecks']['roles'][role]
        assert check['unchanged_outside_alpha'] and check['all_changes_only_original_valid']
    assert all(np.array_equal(x,y) for x,y in zip([base,valid,donor,mask,q,ref],originals))
    assert pair['teacherchecks']['COVER_all_required_taps_opaque'] and pair['teacherchecks']['SHAM_all_required_taps_uncovered'] and pair['teacherchecks']['SHAM_visible_paste_at_least64']
    n,s,m,t=_reference_taps(q,L,np.array([100.5,100.5],np.float64))
    assert np.array_equal(n,[100,100]) and np.array_equal(t[0],[[46,46],[47,46],[46,47],[47,47]])
    oof=compose_counterfactual_pair(base,valid,donor,mask,q,L,np.array([1000.,1000.]),'invented-centre-D')
    assert oof['masked'] and oof['COVER'] is None and oof['proposed_states'] is None
    assert oof['donor_plan']['candidate_positions_sha256']==pair['donor_plan']['candidate_positions_sha256']
    invalid=valid.copy();invalid[47,47]=False
    bad=compose_counterfactual_pair(base,invalid,donor,mask,q,L,ref,'invented-centre-D')
    assert bad['failure_reason']=='REFERENCE_REQUIRED_TAP_ORIGINAL_INVALID'
    empty=compose_counterfactual_pair(base,valid,donor,np.zeros_like(mask),q,L,ref,'invented-centre-D')
    assert empty['failure_reason']=='EMPTY_DONOR_FOREGROUND'
    # Support-only validity permits COVER but makes all SHAM visible masks <64.
    sparse=np.zeros_like(valid);tx=t[:,:,0].reshape(-1);ty=t[:,:,1].reshape(-1);sparse[ty,tx]=True
    no_sham=compose_counterfactual_pair(base,sparse,donor,mask,q,L,ref,'invented-centre-D')
    assert no_sham['masked'] and no_sham['failure_reason']=='SHAM_CANDIDATE_BUDGET_EXHAUSTED' and no_sham['proposed_states'] is None
    constant=np.full_like(base,77);same_donor=np.full_like(donor,77)
    zero=compose_counterfactual_pair(constant,valid,same_donor,mask,q,L,ref,'invented-centre-D')
    assert zero['success'] and all(zero['teacherchecks']['roles'][r]['exact_base_RGB_same'] for r in ['COVER','SHAM'])
    assert zero['proposed_states']=={'COVER':'E','SHAM':'D'}
    return {'completed':True,'PASS':True,'numpy_version':np.__version__,
        'checks':['deterministic_same_pair','input_arrays_and_qL_unchanged','coverage_and_SHAM64','reference_nearest_ties_even_and_four_taps','reference_does_not_change_candidate_pools','out_of_crop_masked_not_E','invalid_support_masked','empty_foreground_masked','SHAM_budget_failure_no_pair_targets','zero_RGB_difference_diagnostic_not_new_gate'],
        'success_example':{'side':pair['donor_plan']['chosen_max_side'],'resized_hw':pair['donor_plan']['resized_hw'],'positions':{r:pair['teacherchecks']['roles'][r]['position_xy'] for r in ['COVER','SHAM']},'teacherchecks_roles':pair['teacherchecks']['roles']},
        'actual_source_donor_GT_model_protected_data_reads':0,'actual_forward_training':0,'invented_array_only':True}


if __name__=='__main__':
    result=run()
    with (Path(__file__).parent/'TOY_RESULT.json').open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))
