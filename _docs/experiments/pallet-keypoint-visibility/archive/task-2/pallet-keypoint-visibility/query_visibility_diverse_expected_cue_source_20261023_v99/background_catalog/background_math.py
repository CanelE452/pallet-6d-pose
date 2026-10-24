"""Public photometric background only. No geometry/GT/model logic."""
import hashlib,math
import numpy as np
from PIL import Image
import PIL
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def resize_geometry(raw_hw,canvas_hw=(640,640)):
 h,w=map(int,raw_hw);dh,dw=map(int,canvas_hw)
 if min(h,w,dh,dw)<=0:raise ValueError('Positive image dimensions required')
 gain=max(dw/w,dh/h)
 nw=max(dw,math.ceil(w*gain));nh=max(dh,math.ceil(h*gain))
 left=(nw-dw)//2;top=(nh-dh)//2
 return dict(raw_hw=[h,w],canvas_hw=[dh,dw],scale_float64=gain,resized_hw=[nh,nw],crop_xyxy=[left,top,left+dw,top+dh],padding=0)
def prepare_background(rgb,canvas_hw=(640,640)):
 if PIL.__version__!='12.2.0':raise ValueError('Pinned Pillow12.2.0 required')
 if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Native uint8 HWC RGB required')
 facts=resize_geometry(rgb.shape[:2],canvas_hw)
 h,w=facts['resized_hw'];im=Image.fromarray(np.ascontiguousarray(rgb)).resize((w,h),resample=Image.Resampling.BILINEAR,reducing_gap=None)
 out=np.array(im.crop(tuple(facts['crop_xyxy'])),dtype=np.uint8,copy=True)
 assert out.shape==tuple(facts['canvas_hw'])+(3,)
 facts.update(decoded_native_RGB_sha256=array_sha(rgb),canvas_RGB_sha256=array_sha(out),interpolation='Pillow12.2.0 BILINEAR reducing_gap=None',format='RGB_UINT8_HWC')
 return out,facts
