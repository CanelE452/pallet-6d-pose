"""Pinned-code + invented arrays/one random-head optimizer step; actual data0."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import train_data_only_counterfactual as trial


def run_toy():
    code={key:{'path':value[0],'sha256':value[1]} for key,value in trial.PINS.items() if key in ['source_trainer','crop_module','prefix_runtime']}
    code['code_whitelist']=list(code.values())
    source,crop,runtime=trial.load_sources(code)
    source.configure_runtime()
    n=80;family=np.repeat(np.array([0,1],np.int8),40);target=np.tile(np.repeat(np.array([0,1],np.int8),20),2)
    fit=np.ones(n,bool);fit[np.r_[16:20,36:40,56:60,76:80]]=False;available=np.ones(n,bool);available[0]=False
    data={'family':family,'target':target,'fit':fit,'available':available}
    draws,eligible=source.fixed_draws(data);saved=draws.copy();parents=np.flatnonzero(eligible&(target==0)).astype(np.int64);success=np.ones(len(parents),bool);success[[0,5,len(parents)-1]]=False
    attempt,variant,actual=trial.build_augmented_draws(draws,data,eligible,parents,success)
    second=trial.build_augmented_draws(draws,data,eligible,parents,success)
    assert np.array_equal(saved,draws) and all(np.array_equal(a,b) for a,b in zip((attempt,variant,actual),second))
    assert eligible[actual].all() and (family[actual]==family[draws]).all()
    for offset in [0,16]:
        assert (attempt[:,offset:offset+4]==-1).all() and (attempt[:,offset+8:offset+12]==-1).all()
        assert (variant[:,offset+12:offset+16]==0).all()
        dpositions=draws[:,offset+4:offset+8];dactual=actual[:,offset+4:offset+8]
        assert np.array_equal(dpositions,dactual)
        assert (variant[:,offset+4:offset+8]==-1).any() and (variant[:,offset+4:offset+8]==1).any()
    assert np.array_equal(attempt[:,12:16]>=0,np.ones((1000,4),bool))
    assert (target[actual[attempt>=0]]==0).all()
    fit_indices=np.flatnonzero(eligible);fit_map=np.full(n,-1,np.int64);fit_map[fit_indices]=np.arange(len(fit_indices))
    good=np.flatnonzero(success);compact=np.full(len(parents),-1,np.int64);compact[good]=np.arange(len(good))
    original_features=torch.randn(len(fit_indices),1186);pair_features=torch.randn(len(good),2,1186)
    x=trial.mix_draw_features(original_features,fit_map,pair_features,compact,draws[0],attempt[0],variant[0],'cpu')
    for p in range(32):
        expected=original_features[fit_map[draws[0,p]]] if attempt[0,p]<0 else pair_features[compact[attempt[0,p]],variant[0,p]]
        assert torch.equal(x[p],expected)
    head=crop.CropFamilyHead();initial=source.cpu_copy(head.state_dict());clone=crop.CropFamilyHead();clone.load_state_dict(initial)
    assert source.tensor_state_sha(head.state_dict())==source.tensor_state_sha(clone.state_dict()) and sum(p.numel() for p in head.parameters())==76098
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    loss,parts=crop.family_loss(head(x),torch.as_tensor(target[draws[0]]),torch.as_tensor(family[draws[0]]))
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters());optimizer.step()
    assert len(optimizer.state)==4 and all(float(state['step'])==1 for state in optimizer.state.values())
    assert source.tensor_state_sha(clone.state_dict())==source.tensor_state_sha(initial) and source.tensor_state_sha(head.state_dict())!=source.tensor_state_sha(initial)
    rejected=[]
    for path in ['/invented/V112/no-open','/invented/cache.pyc','/invented/__pycache__/x']:
        try:trial.safe(path)
        except ValueError:rejected.append(path)
        else:raise AssertionError(path)
    try:trial.verify({'path':'/invented/never-read','sha256':'0'*64},[{'path':'/invented/never-read','sha256':'1'*64}])
    except ValueError:rejected.append('wrong_SHA_before_FS')
    else:raise AssertionError('wrong_SHA')
    original_read=trial.read
    try:
        trial.read=lambda ignored:json.dumps({'approved_data_only_training':False}).encode()
        try:trial.run('invented-false-entry')
        except ValueError:rejected.append('false_entry_before_payload')
        else:raise AssertionError('falseentry')
    finally:trial.read=original_read
    return {'completed':True,'PASS':True,'checks':['original_PCG17_ledger_unchanged','PCG235_augmentation_reproducible','only_FIT_available_known_all_routes','family8D8E_preserved','first4D_E_original','last4D_same_parent_SHAM_or_original_failure_fallback','last4E_successful_same_family_D_parent_COVER','compact_pair_index_and_variant_gather_exact','initial76098_head_cloned_exact','one_random_head_optimizer_step_four_states','forbidden_wrongSHA_falseentry_before_payload'],
        'original_draws_sha256':hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest(),'augmented_draw_sha256':trial.augmentation_draw_sha(attempt,variant,actual),'rejections':rejected,
        'toy_optimizer_steps':1,'actual_SOURCE_RGB_NPZ_prefix_weights_GT_or_protected_payload_reads':0,'actual_DINO_instantiation_or_forward':0,'actual_SOURCE_training':0}


if __name__=='__main__':
    result=run_toy()
    with (Path(__file__).parent/'TOY_RESULT.json').open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))
