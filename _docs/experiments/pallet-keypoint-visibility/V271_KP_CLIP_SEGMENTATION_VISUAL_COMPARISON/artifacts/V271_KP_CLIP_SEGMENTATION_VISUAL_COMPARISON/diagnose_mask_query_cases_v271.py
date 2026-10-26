"""Sample saved selected mask logits at existing q for visual case explanation.

This is numerical postprocessing of frozen maps, not a model forward. Native
human D/E stays a native label; no pixel GT or segmentation IoU is inferred.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
IDS=(2,370,800,840,841,842)
ARMS=('FULL_CLIP','FULL_CLIP_SURGERY_ARCH','EVA_STANDARD','EVA_CSA','EVA_DECLIP')


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(t in str(p).lower() for t in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def main():
    torch.set_num_threads(1)
    csvpath=safe(HERE/'kp_comparison/POINT_MODEL_COMPARISON.csv')
    qrows={int(r['global_index']):r for r in csv.DictReader(csvpath.open())}
    casepath=safe(HERE/'segmentation_atlas/CASES.json');cases={r['frame_index']:r for r in json.loads(casepath.read_text())}
    mappath=safe(HERE/'segmentation_atlas/SELECTED_RAW32_LOGITS.npz')
    with np.load(mappath,allow_pickle=False)as z:maps={k:z[k] for k in z.files}
    records=[]
    for index in IDS:
        r=qrows[index];fi=int(r['frame_index']);j=int(np.flatnonzero(maps['frame_index']==fi)[0]);h,w=cases[fi]['raw_hw']
        q=np.asarray([float(r['original_q_x']),float(r['original_q_y'])],np.float32)
        grid=torch.from_numpy((2*(q+.5)/np.asarray([w,h],np.float32)-1).reshape(1,1,1,2))
        for arm in ARMS:
            x=torch.from_numpy(maps[arm+'_logit'][j].copy())[None,None]
            logit=float(F.grid_sample(x,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0,0,0])
            p=float(torch.sigmoid(torch.tensor(logit,dtype=torch.float32)))
            prediction='D' if logit>=0 else 'E';assert prediction==r[arm],(index,arm,prediction,r[arm])
            records.append({'global_index':index,'frame_index':fi,'native_slot':int(r['native_slot']),
                'human_native_label':r['truth'],'q_x':float(q[0]),'q_y':float(q[1]),
                'method':arm,'sampled_target_logit':logit,'sampled_P_target':p,'native_mask_presence_proxy':prediction,
                'native_label_match':prediction==r['truth'],'pixel_segmentation_GT_available':False})
    out=safe(HERE/'MASK_QUERY_CASE_DIAGNOSIS.csv');assert not out.exists()
    with out.open('x')as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    result={'completed':True,'PASS':True,'source_bindings':{'point_comparison':binding(csvpath),'case_metadata':binding(casepath),'saved_raw_logits':binding(mappath)},
        'selected_original_global_indices':list(IDS),'points':len(IDS),'methods':list(ARMS),'records':records,
        'sample_rule':'Frozen raw32logit bilinear at normalized2*(q+.5)/[W,H]-1,align_cornersFalse,border; sigmoid after sample.',
        'interpretation':'Human D/E is native visibility, not pixel truth. Mask-sample agreement/disagreement is not segmentation IoU.',
        'model_forwards':0,'optimizer_steps':0,'real_pixel_GT_used':0,'query_moved':False}
    p=safe(HERE/'MASK_QUERY_CASE_DIAGNOSIS.json');assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'csv':binding(out),'json':binding(p),'records':records},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
