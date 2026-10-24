"""Invented selection/array math only. No actual source RGB or teacher arrays."""
from pathlib import Path
import importlib.util,json,sys
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v113_generated_render',H/'render_fixed_material_cases.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def scalar(rgb,x,y):
 h,w=rgb.shape[:2];ix=int(np.floor(x));iy=int(np.floor(y));dx=x-ix;dy=y-iy;out=np.zeros(3,float);valid=True
 for ox,oy,weight in((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  xx,yy=ix+ox,iy+oy;inside=0<=xx<w and 0<=yy<h
  if inside:out+=rgb[yy,xx]*weight
  valid&=inside or weight==0
 return out,valid
def main():
 inputs=[dict(split='TRAIN'),dict(split='VALIDATION'),dict(split='TRAIN')]
 evidence=[dict(slot_index=j,object_id=16 if j<16 else 21,raw_surface_teacher_state='U')for j in range(24)]
 for j,state in((0,'D'),(1,'E'),(8,'D'),(16,'D'),(17,'S')):evidence[j]['raw_surface_teacher_state']=state
 chosen,missing=m.choose(evidence[::-1],inputs)
 assert [x['slot_index']for x in chosen]==[0,1,8,16,17,2]
 assert missing==[dict(object_id=16,split='VALIDATION',state='E'),dict(object_id=21,split='TRAIN',state='E')]
 crop,context=m.samplers();rgb=(np.arange(9*13*3).reshape(9,13,3)%256).astype(np.uint8);q=np.array([.25,8.]);before=q.copy();L=16.
 a,valid=crop(rgb,q,L,96)
 checks=0
 for iy in(0,47,48,95):
  for ix in(0,47,48,95):
   x=q[0]+((ix+.5)/96-.5)*L;y=q[1]+((iy+.5)/96-.5)*L;value,_=scalar(rgb,x,y)
   v=(-.5<=x<rgb.shape[1]-.5)and(-.5<=y<rgb.shape[0]-.5)
   expected=np.floor(np.clip(value,0,255)+.5).astype(np.uint8)if v else np.zeros(3,np.uint8)
   assert np.array_equal(a[:,iy,ix],expected)and valid[0,iy,ix]==v;checks+=1
 box=np.array([-1.,0.,13.,10.]);ctx,cv=context(rgb,box,(224,224))
 for iy in(0,111,223):
  for ix in(0,111,223):
   x=box[0]+(ix+.5)*(box[2]-box[0])/224;y=box[1]+(iy+.5)*(box[3]-box[1])/224;expected,v=scalar(rgb,x,y)
   assert np.array_equal(ctx[iy,ix],expected.astype(np.float32))and cv[iy,ix]==v;checks+=1
 raw=Image.fromarray(a.transpose(1,2,0));snapshot=np.asarray(raw).copy();m.mark(raw,np.array([47.5,47.5]),np.array([51.,43.]))
 assert np.array_equal(np.asarray(raw),snapshot)and np.array_equal(q,before)
 proof=dict(completed=True,PASS=True,code=m.bind(H/'render_fixed_material_cases.py'),test_code=m.bind(__file__),crop_code=m.bind(m.CROP),context_code=m.bind(m.CONTEXT),
            selection='Fixed strata/original row/minimum/missing no substitution PASS',scalar_crop_context_checks=checks,marker_leaves_raw_unchanged=True,
            actual_source_RGB_reads=0,actual_teacher_array_reads=0,model_fit_private_V112_PnP=False)
 m.write(H/'GENERATED_RENDER_TESTS.json',proof);print(json.dumps(dict(PASS=True,proof=m.bind(H/'GENERATED_RENDER_TESTS.json'))))
if __name__=='__main__':main()
