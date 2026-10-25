"""Random tensor/state fixtures only. No SOURCE/model/cache/old checkpoint reads."""
import copy, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import crop_modules as new
import train_crop_adaptation as trainer

HERE=Path(__file__).absolute().parent
OLD=HERE.parent/'V233_CROP_ADAPT_TRAINER_A/crop_modules.py'
assert 'v112' not in str(OLD).casefold()
for p in reversed((OLD,*OLD.parents)):assert not p.is_symlink()
raw=OLD.read_bytes();assert hashlib.sha256(raw).hexdigest()=='a3aaa1df13660595cc4878e0c808a409571c2d2f0d1015a6ad59a4b5fc239225'
spec=importlib.util.spec_from_file_location('v233_old_crop_toy',OLD);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ADAPTER=HERE.parent/'V232_ADAPTER_PREPARATION_A/adapter_modules.py'
assert 'v112' not in str(ADAPTER).casefold()
for p in reversed((ADAPTER,*ADAPTER.parents)):assert not p.is_symlink()
raw=ADAPTER.read_bytes();assert hashlib.sha256(raw).hexdigest()=='9558f48279891f192eb9664968b41e9c03022a475b9666b5dc95229195fa8b9a'
spec=importlib.util.spec_from_file_location('v233_pinned_adapter_toy',ADAPTER);OldGrid=importlib.util.module_from_spec(spec);spec.loader.exec_module(OldGrid)

def byte_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def fixtures():
    trainer.configure_runtime();assert torch.cuda.is_available(),'Explicit CUDA random-tensor proof required'
    cases=[]
    for device in ['cpu','cuda']:
        generator=torch.Generator(device=device).manual_seed(233)
        patch=torch.randn((32,256,384),generator=generator,device=device,dtype=torch.float32)
        cls=torch.randn((32,384),generator=generator,device=device,dtype=torch.float32)
        centre=torch.randn((32,384),generator=generator,device=device,dtype=torch.float32)
        mask=(torch.arange(32,device=device)%3)!=0
        fractions=torch.ones(32,2,device=device);fractions[0,0]=0;fractions[1,1]=0
        condition=torch.zeros(32,32,device=device)
        full={'x_norm_clstoken':cls,'x_norm_patchtokens':patch};central={'x_norm_clstoken':centre}
        with torch.no_grad():
            original=old.crop_features(full,central,fractions,mask,condition,OldGrid)
            repaired=new.crop_features(full,central,fractions,mask,condition,None)
        assert torch.equal(original,repaired),(device,float((original-repaired).abs().max()))
        assert byte_sha(original)==byte_sha(repaired)
        patches=patch.clone().requires_grad_(True)
        result=new.fixed_center_q_read(patches,mask)
        multiplier=torch.randn(result.shape,generator=generator,device=device)
        (result*multiplier).sum().backward()
        assert patches.grad is not None and torch.isfinite(patches.grad).all() and torch.count_nonzero(patches.grad)>0
        assert torch.count_nonzero(patches.grad[~mask])==0
        spatial=patches.grad.reshape(32,16,16,384);away=spatial.clone();away[:,7:9,7:9]=0
        assert torch.count_nonzero(away)==0
        second=patch.clone().requires_grad_(True)
        (new.fixed_center_q_read(second,mask)*multiplier).sum().backward()
        assert torch.equal(patches.grad,second.grad) and byte_sha(patches.grad)==byte_sha(second.grad)
        # Positive/negative zero forward bytes remain the original kernel bytes.
        zeros=torch.zeros_like(patch);zeros[:,7*16+7]=-0.0
        zero_full={'x_norm_clstoken':cls,'x_norm_patchtokens':zeros}
        with torch.no_grad():
            old_zero=old.crop_features(zero_full,central,fractions,mask,condition,OldGrid)
            new_zero=new.crop_features(zero_full,central,fractions,mask,condition,None)
        assert byte_sha(old_zero)==byte_sha(new_zero)
        assert torch.are_deterministic_algorithms_enabled()
        cases.append({'device':device,'random_rows':32,'full1186_feature_forward_bitexact':True,'forward_byte_SHA256':byte_sha(repaired),'repeat_backward_gradient_byte_SHA256':byte_sha(patches.grad),'repeated_backward_gradients_bitexact':True,'signed_zero_forward_bytes_exact':True,'max_abs_forward_difference':0.0,'deterministic_backward_PASS':True,'masked_gradient_zero':True,'gradient_only_four_tokens':True})
    torch.manual_seed(17);head=new.CropFamilyHead();initial=trainer.cpu_copy(head.state_dict());initial_sha=trainer.tensor_state_sha(initial)
    data={'fit':np.ones(8,bool),'available':np.ones(8,bool),'family':np.repeat(np.array([0,1],np.int8),4),'target':np.tile(np.array([0,0,1,1],np.int8),2)}
    draws,eligible=trainer.fixed_draws(data);drawsha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    official={'path':'/invented/official.pt','sha256':'a'*64};entry={'path':'/invented/root_entry.json','sha256':'b'*64}
    e={'official_state':official,'reuse_control':{'original_entry':entry}}
    cp={'arm':'CONTROL_LOCAL_CROP','updates':1000,'use_lora':False,'head_parameters':76098,'adapter_parameters':0,'adapter_state_dict':{},'initial_adapter_state_dict':{},'head_state_dict':initial,'initial_head_state_dict':initial,'initial_head_sha256':initial_sha,'draws_sha256':drawsha,'base_state_sha256_before_after':'c'*64,'official_base':official,'vendor_revision':trainer.VENDOR_REVISION,'entry':entry,'adapter_seed':17,'rank':4,'scale':1,'optimizer_state_dict':{'param_groups':[{'params':list(range(4)),'lr':1e-3,'weight_decay':1e-4}],'state':{i:{'step':torch.tensor(1000.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)} for i,p in enumerate(head.parameters())}}}
    records=[{'update':i+1,'sample_indices':idx.tolist(),'draws_sha256':drawsha,'family_counts':[16,16],'all_fit_available_known':True,'family0_BCE':.5,'family1_BCE':.75,'total_loss':.625} for i,idx in enumerate(draws)]
    assert trainer.validate_control_reuse(cp,records,initial,initial_sha,draws,drawsha,'c'*64,e)
    rejected=[]
    for name,modify in [('wrong_updates',lambda c:c.update(updates=999)),('wrong_draw_sha',lambda c:c.update(draws_sha256='d'*64)),('wrong_original_entry',lambda c:c.update(entry={'path':entry['path'],'sha256':'d'*64})),('wrong_base',lambda c:c.update(base_state_sha256_before_after='d'*64)),('adapter_present',lambda c:c.update(use_lora=True))]:
        bad=copy.deepcopy(cp);modify(bad)
        try:trainer.validate_control_reuse(bad,records,initial,initial_sha,draws,drawsha,'c'*64,e)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    bad=copy.deepcopy(records);bad[21]['sample_indices'][0]=999
    try:trainer.validate_control_reuse(cp,bad,initial,initial_sha,draws,drawsha,'c'*64,e)
    except ValueError:rejected.append('wrong_log_draw')
    else:raise AssertionError('wrong_log_draw')
    return {'completed':True,'PASS':True,'torch_version':torch.__version__,'CUDA_device':torch.cuda.get_device_name(0),'deterministic_algorithms':True,'feature_and_backward_checks':cases,'synthetic_CONTROL_reuse_accept':True,'synthetic_reuse_rejections':rejected,'actual_SOURCE_DINO_cache_weights_checkpoint_logs_reads':0,'actual_optimizer_steps':0,'actual_training_or_forward':0,'random_tensor_backward_only':True,'old_grid_backward_not_executed':'Actual V1 failure is preserved by root; only old forward compared here.'}

if __name__=='__main__':
    result=fixtures()
    with (HERE/'TOY_DETERMINISTIC_FIX_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
