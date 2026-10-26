"""Disambiguate current dynamic native slots from SYN asset canonical corners.

No teacher-value/coordinate change. Preserve v1 as prior terminology evidence.
"""
from pathlib import Path
import hashlib
import io
import json
import numpy as np

HERE=Path(__file__).absolute().parent


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def savej(path,x):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    previous=safe(HERE/'raw_model_targets9_v1/COMPLETE.json')
    assert binding(previous)['sha256']=='bb0687d174201caf7d0fa29f35d67e9df1e3470bd229fdf076790d3c324c09b5'
    c=json.loads(previous.read_text());b=c['outputs']['RAW_MODEL_TARGETS9.npz'];assert binding(b['path'])==b
    with np.load(safe(b['path']),allow_pickle=False)as z:a={k:z[k] for k in z.files}
    prior_native=a['model_channel_to_native'].copy()
    identity=np.tile(np.arange(9,dtype=np.int8),(1003,1))
    a['model_channel_to_asset_canonical9']=prior_native
    a['model_channel_to_native']=identity
    a['model_channel_to_current_native']=identity.copy()
    assert np.array_equal(prior_native[:,:8],a['model_channel_to_canonical_native8'])
    out=safe(HERE/'raw_model_targets9_v2');assert not out.exists(),out;out.mkdir()
    np.savez_compressed(out/'RAW_MODEL_TARGETS9.npz',**a)
    savej(out/'RAW_CHANNEL_CONTRACT.json',{'completed':True,'PASS':True,'source_v1':binding(previous),
        'model_channel':'Original R0 raw camera_dynamic_0123_v4 channel0..7,centroid8.',
        'current_ledger_native_slot':'Legacy R0 raw dynamic channel index, not canonical asset xyz. model_channel_to_native/current_native is identity9.',
        'SYN_asset_canonical':'model_channel_to_canonical_native8=storedperm_v4; model_channel_to_asset_canonical9=perm_v4+centroid8. This is only geometry provenance.',
        'do_not_apply_SYN_perm_to_current_predictions':True,'do_not_reapply_perm_to_existing_dynamic_xy_or_E':True,
        'v1_terminology_issue':'V1 model_channel_to_native denoted asset canonical corner; v1 contract was explicit but field name could be confused with current ledger native_slot. v2 separates both.',
        'label_values_and_all_coordinates_unchanged':True,'GT_best_permutation_selection':False,
        'real_input_mapping_evidence':'Root independent audit INPUT_CHANNEL_NATIVE_IDENTITY_BEFORE_GT.json, separate from this SYN-only preparation.',
        'current223_reads':0,'RGB_reads':0,'model_forwards':0})
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'outputs':{n:binding(out/n) for n in ['RAW_MODEL_TARGETS9.npz','RAW_CHANNEL_CONTRACT.json']},
        'supersedes_transport_terminology_only':binding(previous),
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in a.items()},
        'unchanged_teacher_arrays':['gt_xy9_raw','gt_xy9_raw01','coordvalid9','E9','Emask9','fit','raw_hw','frame_index'],
        'model_channel_to_current_native_identity':True,'current223_reads':0,'RGB_reads':0,'model_forwards':0})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)


if __name__=='__main__':main()
