"""Nonzero-output whole/flat and masked-pixel behavior, generated inputs only."""
from pathlib import Path
import hashlib,importlib.util,json,torch
R=Path(__file__).resolve().parent
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
p=json.loads((R/'MIXED_FIT_PROTOCOL.json').read_text());s=importlib.util.spec_from_file_location('m',R/'query_center_support_head.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1);torch.manual_seed(20261023);model=m.NativeContextSupportHead();m.initialize_from_proxy(model,torch.load(p['bindings']['pretrain']['path'],weights_only=True,map_location='cpu'))
torch.nn.init.normal_(model.classifier[2].weight,std=.1)
v=dict(p3_roi=torch.randn(2,8,65,9,9),local_rgb96=torch.randint(0,256,(2,8,3,96,96),dtype=torch.uint8),local_valid96=torch.ones(2,8,1,96,96,dtype=torch.bool),target_rgb128=torch.randint(0,256,(2,3,128,128),dtype=torch.uint8),target_valid128=torch.ones(2,1,128,128,dtype=torch.bool),context42=torch.randn(2,8,42));v['p3_roi'][:,:,64]=1;v['p3_roi'][:,:,64,:3,:3]=0;v['local_valid96'][:,:,:,:20,:20]=False;v['target_valid128'][:,:,:24,:24]=False
flat={k:(x.repeat_interleave(8,0)if k in ('target_rgb128','target_valid128')else x.reshape(16,*x.shape[2:]))for k,x in v.items()}
a=model(**v)['support_logit'];a.sum().backward();ga={k:x.grad.clone()for k,x in model.named_parameters()};model.zero_grad(set_to_none=True);c=model(**flat)['support_logit'];c.sum().backward();assert torch.equal(a.flatten(),c) and c.abs().sum()>0
assert all(torch.equal(ga[k],x.grad)for k,x in model.named_parameters())
w={k:x.clone()for k,x in v.items()};w['local_rgb96'][:,:,:,:20,:20]=255-w['local_rgb96'][:,:,:,:20,:20];w['target_rgb128'][:,:,:24,:24]=255-w['target_rgb128'][:,:,:24,:24];w['p3_roi'][:,:,:64,:3,:3]=100
with torch.no_grad():assert torch.equal(a,model(**w)['support_logit'])
out=R/'GENERATED_NONZERO_FORWARD_ADDENDUM.json';assert not out.exists();out.write_text(json.dumps(dict(PASS=True,code=b(Path(__file__)),model=b(R/'query_center_support_head.py'),protocol=b(R/'MIXED_FIT_PROTOCOL.json'),nonzero_whole_flat_output_exact=True,nonzero_whole_flat_parameter_gradient_exact=True,invalid_RGB_target_P3_values_do_not_change_output=True,source_or_current_arrays_read=0,optimizer_updates=0,PnP=0),indent=2)+'\n');print('NONZERO_FORWARD_PASS')
