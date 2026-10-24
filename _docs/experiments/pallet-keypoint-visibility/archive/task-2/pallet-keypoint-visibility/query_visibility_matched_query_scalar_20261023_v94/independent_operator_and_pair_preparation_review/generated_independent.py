"""Independent invented tensors/math only. No source loader or model assets."""
import importlib.util
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V=HERE.parent

def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def rect_brute(a):
 h,w=a.shape;rects=[]
 for y0 in range(h):
  for x0 in range(w):
   for y1 in range(y0+1,h+1):
    for x1 in range(x0+1,w+1):
     if a[y0:y1,x0:x1].all():rects.append(((x1-x0)*(y1-y0),(x0,y0,x1,y1)))
 return sorted(rects,key=lambda z:(-z[0],z[1]))[0][1] if rects else None

def main():
 torch.set_num_threads(1);torch.manual_seed(9817)
 op=module(V/'operator_preparation/query_center_scalar.py','independent_op')
 pair=module(V/'paired_source_preparation/v2_geometry_mass_proposal/geometry_mass_v2.py','independent_v2')
 # Own analytic loss/gradient oracle, including unknown NaN and known unavailable.
 z=torch.tensor([-.2,.8,1.,-2.,.3],dtype=torch.float64,requires_grad=True)
 y=torch.tensor([0.,1.,1.,float('nan'),-1.],dtype=torch.float64)
 known=torch.tensor([True,True,True,False,False]);avail=torch.tensor([True,True,False,True,True])
 loss=op.known_binary_loss(dict(query_logit=z,query_available=avail),y,known)
 zz=z.detach().numpy();yy=np.array([0.,1.]);expected=np.logaddexp(0,zz[:2])-yy*zz[:2]
 loss_error=abs(loss['loss'].item()-expected.mean());assert loss_error<1e-15
 loss['loss'].backward();gradient=np.zeros(5);gradient[:2]=(1/(1+np.exp(-zz[:2]))-yy)/2
 gradient_error=float(np.max(abs(z.grad.numpy()-gradient)));assert gradient_error<1e-15
 assert np.array_equal(z.grad.numpy()[2:],np.zeros(3))
 # Full-model centre-invalid rows have zero input gradient after output unmasked,
 # with nonzero readout weights to avoid the default zero-output vacuity.
 m=op.QueryCenterScalar()
 with torch.no_grad():m.readout[-1].weight.fill_(.2);m.readout[-1].bias.fill_(.7)
 p=torch.randn(3,65,9,9);p[:,64]=1;p[1,64,4,4]=0;p.requires_grad_()
 rgb=torch.randint(0,256,(3,3,96,96),dtype=torch.uint8)
 valid=torch.ones(3,1,96,96,dtype=torch.bool);valid[2,0,48,48]=False
 out=m(p,rgb,valid);assert out['query_available'].tolist()==[True,False,False]
 assert out['query_logit'][1:].tolist()==[0.,0.]
 one=op.known_binary_loss(out,torch.tensor([0.,1.,1.]),torch.ones(3,dtype=torch.bool))
 one['loss'].backward();assert p.grad[1:].abs().sum()==0 and p.grad[0,:64].abs().sum()>0
 assert op.decode_binary(out)[1:].tolist()==[-1,-1]
 # Unknown empty graph zero and strict malformed-known label rejection.
 n=torch.randn(2,requires_grad=True)
 empty=op.known_binary_loss(dict(query_logit=n,query_available=torch.ones(2,dtype=torch.bool)),torch.full((2,),float('nan')),torch.zeros(2,dtype=torch.bool))['loss']
 empty.backward();assert empty.item()==0 and n.grad.abs().sum()==0
 try:op.known_binary_loss(dict(query_logit=n,query_available=torch.ones(2,dtype=torch.bool)),torch.tensor([2.,0.]),torch.ones(2,dtype=torch.bool))
 except ValueError:pass
 else:raise AssertionError('Known nonbinary accepted')
 # Exhaustively test all3x3 alpha configurations with independent rectangle enumeration.
 for bits in range(512):
  alpha=np.array([(bits>>i)&1 for i in range(9)],bool).reshape(3,3)
  assert pair.largest_opaque_rectangle(alpha)==rect_brute(alpha)
 # Independent separable scalar bilinear sampling, three fractional/boundary cases.
 rng=np.random.default_rng(910)
 errors=[]
 for shape,q,L in [((11,13),(.1,10.2),18.3),((30,40),(21.7,13.3),29.1),((7,7),(0.,0.),4.)]:
  alpha=rng.integers(0,2,shape).astype(bool);h,w=shape
  u=(np.arange(96,dtype=np.float64)+.5)/96-.5
  xs=q[0]+u*L;ys=q[1]+u*L;expected=0.
  for y in ys:
   for x in xs:
    if not(-.5<=x<w-.5 and -.5<=y<h-.5):continue
    ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy
    for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
     if 0<=ix+ax<w and 0<=iy+ay<h:expected+=int(alpha[iy+ay,ix+ax])*weight
  errors.append(abs(pair.alpha_mass96(alpha,q,L)-expected))
 assert max(errors)<2e-10
 # Scale impossible, empty protection, empty/1px-thin donor remain unusable U.
 drgb=np.zeros((30,40,3),np.uint8);donor=np.zeros((30,40),bool);donor[2:28,2:38]=True
 protected=np.zeros((96,96),bool);protected[20:60,20:60]=True
 assert pair.support_scaled_cutout(drgb,donor,protected,16.)[0]is None
 assert pair.support_scaled_cutout(drgb,donor,np.zeros_like(protected),64.)[0]is None
 thin=np.zeros_like(donor);thin[:,10]=1
 assert pair.support_scaled_cutout(drgb,thin,protected,64.)[0]is None
 # Strict relative mass is declared prospectively, not learned from source scores.
 assert pair.M96_RELATIVE_TOLERANCE==.01 and pair.M96_ABSOLUTE_NUMERICAL_TOLERANCE==1e-9
 report=dict(completed=True,PASS=True,independent_loss_error=loss_error,
  analytic_gradient_max_error=gradient_error,full_model_unavailable_input_gradient_exact0=True,
  empty_unknown_NaN_graphzero=True,malformed_known_rejected=True,
  exhaustive_3x3_rectangle_patterns=512,scalar_alpha_sample_cases=3,scalar_alpha_sample_count=27648,
  scalar_alpha_mass_errors=errors,impossible_empty_thin_retained_U=True,
  numeric_comparison_tolerances=dict(loss_gradient=1e-15,scalar_sum=2e-10),
  actual_source_array_pixel_weight_current_model_forward_fit_PnP=0,
  generated_random_model_CPU_forward_only=True)
 out=HERE/'INDEPENDENT_GENERATED_MATH.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
