"""Measured grayscale/derivative observations, not visible-boundary labels."""
from context25 import *
import cv2

def actual_image(row):
    im=PREP.raw_image(row)
    if row.get('mask') is not None:
        a,u,b,v=row['mask'];im[u:v,a:b]=row['fill_bgr']
    x,y=row['origin'];h,w=row['image_hw'];im=im[y:y+h,x:x+w].copy()
    assert list(im.shape[:2])==row['image_hw']
    if 'transformed_sha256' in row:assert __import__('hashlib').sha256(im.tobytes()).hexdigest()==row['transformed_sha256']
    return im

def edges_for_manifest(m,key):
    path=RUN/'edge_cache'/f'{key}.pt'
    if path.exists():return torch.load(path,weights_only=True,map_location='cpu')
    cv2.setNumThreads(1);parts=[]
    for row in m['rows']:
        if not row['detected']:continue
        im=actual_image(row);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        dx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REPLICATE);dy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REPLICATE)
        a,u,b,v=row['box'];xx,yy=np.meshgrid(np.linspace(a,b,96,dtype=np.float32),np.linspace(u,v,96,dtype=np.float32))
        parts.append(torch.from_numpy(np.stack([cv2.remap(p,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0) for p in (gray,dx,dy)])))
    out=torch.stack(parts);path.parent.mkdir(parents=True,exist_ok=True);torch.save(out,path)
    write(path.with_suffix('.json'),dict(rows=[r['id'] for r in m['rows'] if r['detected']],manifest_digest=digest(json.dumps(m,sort_keys=True)),sha256=sha(path),description='Grayscale and signed Sobel derivatives sampled in original pixel coordinates,96ROI. No GT, segmentation label, or hallucinated occluded boundary.'))
    print('PIXEL EVIDENCE',key,len(out),flush=True);return out

