"""Pure generated preprocessing/descriptor math, no backbone model/weights/forward."""
from pathlib import Path
import argparse,importlib.util,json
import torch
H=Path(__file__).resolve().parent;CODE_PATH=H/'prepare_frozen_rgb_features.py';s=importlib.util.spec_from_file_location('generated_c8',CODE_PATH);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main(output):
 torch.set_num_threads(1);rgb=torch.arange(2*3*96*96,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(2,3,96,96);v=torch.ones(2,1,96,96,dtype=torch.bool);v[0]=False;v[1,:,:,0:16]=False
 views,f=m.preprocess_views(rgb,v);assert views.shape==(4,3,224,224)and f.shape==(2,2)and torch.isfinite(views).all();assert views[0].abs().max()<1e-6 and views[2].abs().max()<1e-6 and torch.equal(f,torch.tensor([[0.,0.],[80/96,1.]]))
 # Fixedcentral32isexactsource[32:64], directresize224, no newcrop/recenter.
 mean=torch.tensor(m.MEAN)[None,:,None,None];std=torch.tensor(m.STD)[None,:,None,None];central=rgb[1:2,:,32:64,32:64].float()/255
 expected=(torch.nn.functional.interpolate(central,(224,224),mode='bilinear',align_corners=False,antialias=True)-mean)/std;assert torch.equal(views[3:4],expected)
 raw=torch.arange(4*512,dtype=torch.float32).reshape(4,512);features=m.assemble_features(raw,f);assert features.shape==(2,1026)and torch.equal(features[:,-2:],f)and features[0,:1024].count_nonzero()==0
 assert torch.allclose(features[1,:512].norm(),torch.tensor(1.),atol=1e-6)and torch.allclose(features[1,512:1024].norm(),torch.tensor(1.),atol=1e-6)
 try:m.preprocess_views(rgb.float(),v);raise RuntimeError('wrongdtypeaccepted')
 except ValueError:pass
 result=dict(completed=True,PASS=True,code=m.bind(CODE_PATH),test_code=m.bind(__file__),torch=torch.__version__,checks=dict(meanfill_zero_normalized=True,fixedcentral32_direct224_exact=True,L2_per_scale_and_fractions=True,zero_support_zero_safe=True,strict_RGB_valid_schema=True),actual_backbone_construct_or_weights_load_or_model_forward=0,source_pixels_or_tensors_read=0,new_fit_private_current_PnP=0)
 with Path(output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(PASS=True,proof=m.bind(output))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
