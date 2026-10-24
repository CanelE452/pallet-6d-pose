"""Independent source-RGB decoder diagnostic; never imports a model."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import PIL
from PIL import Image
H=Path(__file__).resolve().parent;V=H.parent
INPUTS=V/'initializer_preparation/INPUTS16_RGB_ONLY.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main(tag):
    import cv2
    records=[]
    for row in [json.loads(l)for l in INPUTS.read_text().splitlines()]:
        assert sha(row['image'])==row['image_sha256']
        pil=np.asarray(Image.open(row['image']).convert('RGB'))
        cv=np.ascontiguousarray(cv2.imread(row['image'],cv2.IMREAD_COLOR)[:,:,::-1])
        selected=pil if row['decoder']=='CANONICAL_PIL_RGB_TO_BGR'else cv
        record={k:row[k]for k in('frame_index','id','source','image','image_sha256','raw_hw','decoder','decoded_RGB_sha256')}
        for label,rgb in(('PIL_RGB',pil),('CV2_RGB',cv),('declared_decoder_RGB',selected)):
            record[label]=dict(decoded_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest(),dtype=str(rgb.dtype),shape=list(rgb.shape),
                expected_SHA_match=hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_RGB_sha256'],expected_HW_match=list(rgb.shape)==row['raw_hw']+[3])
        delta=pil.astype(np.int16)-cv.astype(np.int16)
        record['PIL_vs_CV2_changed_pixels']=int(np.any(delta,axis=2).sum());record['PIL_vs_CV2_max_abs_channel_delta']=int(abs(delta).max())
        records.append(record)
    failed=[r['frame_index']for r in records if not(r['declared_decoder_RGB']['expected_SHA_match']and r['declared_decoder_RGB']['expected_HW_match'])]
    path=H/(tag+'_DECODER_ONLY_RESULT.json');assert not path.exists()
    result=dict(completed=True,model_import_forward_or_fit=0,private_V112_PnP=0,code=bind(__file__),inputs=bind(INPUTS),
        runtime=dict(python=sys.executable,Python=sys.version,NumPy=np.__version__,Pillow=PIL.__version__,cv2=cv2.__version__,jpeglib=getattr(Image.core,'jpeglib_version',None),libjpeg_turbo=getattr(Image.core,'libjpeg_turbo_version',None)),
        all16_original_raw_file_SHAs_exact=True,rows=records,declared_decoder_mismatch_indices=failed,first_failure_index=None if not failed else min(failed),
        successful_source_forward_prefix_in_original_order_inferred_from_first_decode_failure=None if not failed else min(failed),
        no_actual_inference_retry=True,no_image_output_created=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(result=bind(path),mismatch_indices=failed,runtime=result['runtime'])))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);main(parser.parse_args().tag)
