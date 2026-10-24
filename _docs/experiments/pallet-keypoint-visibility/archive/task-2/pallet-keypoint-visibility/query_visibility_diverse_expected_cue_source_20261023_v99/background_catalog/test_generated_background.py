from pathlib import Path
import json,hashlib
import numpy as np
import PIL
from background_math import resize_geometry,prepare_background,array_sha
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def main():
 assert PIL.__version__=='12.2.0'
 a=resize_geometry([480,640]);assert a['resized_hw']==[640,854]and a['crop_xyxy']==[107,0,747,640]and a['padding']==0
 b=resize_geometry([640,480]);assert b['resized_hw']==[854,640]and b['crop_xyxy']==[0,107,640,747]
 c=resize_geometry([640,640]);assert c['resized_hw']==[640,640]and c['crop_xyxy']==[0,0,640,640]
 rgb=np.zeros((3,5,3),np.uint8);rgb[:]=[17,83,249]
 out,f=prepare_background(rgb,(12,12));assert np.all(out==[17,83,249])and f['resized_hw']==[12,20]and f['crop_xyxy']==[4,0,16,12]
 gradient=np.arange(3*5*3,dtype=np.uint8).reshape(3,5,3)
 x,fx=prepare_background(gradient,(12,12));y,fy=prepare_background(gradient,(12,12));assert np.array_equal(x,y)and fx==fy
 assert fx['decoded_native_RGB_sha256']==array_sha(gradient)and fx['canvas_RGB_sha256']==array_sha(x)
 proof=dict(completed=True,PASS=True,math=bind(H/'background_math.py'),generated_code=bind(__file__),Pillow=PIL.__version__,landscape_portrait_square_crop_bounds_exact=True,uniform_RGB_exact=True,deterministic_bilinear_and_hashes=True,actual_public_RGB_reads=0,models_fit_current_PnP=0)
 with(H/'GENERATED_BACKGROUND_TESTS.json').open('x')as f:json.dump(proof,f,indent=2);f.write('\n')
 print(json.dumps(proof))
if __name__=='__main__':main()
