"""Canonical Python3.13/Pillow12.0 source-RGB SHA replay; no model/cv2."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import PIL
from PIL import Image
H=Path(__file__).resolve().parent;V=H.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
assert sys.version_info[:2]==(3,13)and PIL.__version__=='12.0.0'
inputs=V/'initializer_preparation/INPUTS16_RGB_ONLY.jsonl';records=[]
for row in[json.loads(l)for l in inputs.read_text().splitlines()]:
    assert sha(row['image'])==row['image_sha256']
    rgb=np.asarray(Image.open(row['image']).convert('RGB'));s=hashlib.sha256(rgb.tobytes()).hexdigest()
    records.append(dict(frame_index=row['frame_index'],id=row['id'],image=row['image'],original_raw_SHA_exact=True,expected_decoded_SHA=row['decoded_RGB_sha256'],decoded_SHA=s,
                        SHA_match=s==row['decoded_RGB_sha256'],HW_match=list(rgb.shape)==row['raw_hw']+[3],dtype=str(rgb.dtype)))
path=H/'CANONICAL_PY313_PIL_ONLY_RESULT.json';assert not path.exists()
result=dict(completed=True,PASS=all(r['SHA_match']and r['HW_match']and r['dtype']=='uint8'for r in records),code=bind(__file__),inputs=bind(inputs),
    runtime=dict(python=sys.executable,Python=sys.version,NumPy=np.__version__,Pillow=PIL.__version__,jpeglib=getattr(Image.core,'jpeglib_version',None),libjpeg_turbo=getattr(Image.core,'libjpeg_turbo_version',None)),rows=records,
    all16_original_raw_file_SHAs_exact=True,source_RGB_decode_only=True,model_checkpoint_forward_fit_private_V112_PnP=0,no_image_output_created=True)
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=bind(path),PASS=result['PASS'],runtime=result['runtime'])))
