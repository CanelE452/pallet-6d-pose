"""Explicit raw detector-channel9 targets and canonical-native transport.

Raw channels equal annotation projected_cuboid[0:8]+centroid, not a guessed
per-frame GT permutation. Stored perm_v4 is independently verified against
canonical cuboid geometry; it is recorded as transport, not applied to labels.
"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import numpy as np

HERE=Path(__file__).absolute().parent
GT=Path('/home/minjae/Documents/pallet-research/pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(bound):
    assert binding(bound['path'])==bound,bound
    return safe(bound['path']).read_bytes()


def savej(path,obj):
    p=safe(path);assert not p.exists(),p
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    source=HERE/'syn_joint_targets9_v1';c=json.loads(safe(source/'COMPLETE.json').read_text())
    assert c['PASS'] and c['completed']
    with np.load(io.BytesIO(checked(c['outputs']['TARGETS9.npz'])),allow_pickle=False)as z:a={k:z[k] for k in z.files}
    native=HERE/'syn_joint_targets_v1';nc=json.loads(safe(native/'COMPLETE.json').read_text())
    rows=[json.loads(x) for x in checked(nc['outputs']['FRAME_TARGETS1003.jsonl']).decode().splitlines()]
    gtbytes=safe(GT).read_bytes();assert hashlib.sha256(gtbytes).hexdigest()=='578a24d67db0959a0b014ece9802279c803ee05c9845c2fbe38e8e8c389bdeb6'
    gt={x['id']:x for x in map(json.loads,gzip.decompress(gtbytes).decode().splitlines())}
    assert a['xy_px'].shape==(1003,9,2) and a['loc_valid'].shape==(1003,9)
    E9=np.full((1003,9),-1,np.int8);E9[:,:8]=a['visibility_y'];Emask9=(E9>=0)
    transport=np.full((1003,9),8,np.int8);transport[:,:8]=a['perm_v4_provenance']
    for i,r in enumerate(rows):
        g=gt[r['id']];w,h,d=np.asarray(g['physical_dimensions_xyz_m'],np.float64)/2
        canonical=np.asarray([[-w,h,d],[w,h,d],[w,-h,d],[-w,-h,d],[-w,h,-d],[w,h,-d],[w,-h,-d],[-w,-h,-d]],np.float64)
        assert np.array_equal(np.asarray(g['native_GT_xyz'],np.float64),canonical[transport[i,:8]])
        assert np.array_equal(a['xy_px'][i,:8],np.asarray(r['projected_native_dynamic8_xy'],np.float32))
    out=safe(HERE/'raw_model_targets9_v1');assert not out.exists(),out;out.mkdir()
    arrays={'gt_xy9_raw':a['xy_px'],'gt_xy9_raw01':a['xy01'],'coordvalid9':a['loc_valid'],
        'E9':E9,'Emask9':Emask9,'model_slot':np.tile(np.arange(9,dtype=np.int8),(1003,1)),
        'model_channel_to_native':transport,'model_channel_to_canonical_native8':transport[:,:8],
        'fit':a['fit'],'raw_hw':a['raw_hw'],'frame_index':a['frame_index']}
    np.savez_compressed(safe(out/'RAW_MODEL_TARGETS9.npz'),**arrays)
    savej(out/'RAW_CHANNEL_CONTRACT.json',{'code':binding(Path(__file__).absolute()),
        'native8_complete':binding(native/'COMPLETE.json'),'coordinate9_complete':binding(source/'COMPLETE.json'),
        'source_gt':binding(GT),'model_channels':'Annotation.objects[0].projected_cuboid[0:8], followed by projected_cuboid_centroid at8.',
        'native_teacher_channels':'occlusion_fraction[0:8] and source self_visible(world cuboid) use same stored camera_dynamic_0123_v4 order.',
        'canonical_native_definition':'[-W/2,+H/2,+D/2],[+W/2,+H/2,+D/2],[+W/2,-H/2,+D/2],[-W/2,-H/2,+D/2], then corresponding -D/2 four points; renderer physical frame.',
        'transport':'model_channel_to_native[k]=stored perm_v4[k] for first8; centroid8->8. Verified canonical_xyz[perm_v4] equals saved native_GT_xyz exactly for all1003. This is provenance transport, not GT-selected best permutation.',
        'applied_permutations_to_xy_or_E':0,'raw_annotation_channel_matches':1003,'canonical_transport_exact_matches':1003,
        'E9':'-1 unknown/ignored,0clear external ray proxy,1blocked external ray proxy; centroid always-1 and EmaskFalse.',
        'preprocess':'Coordinates remain raw. Apply same image affine as detector input once; this file performs no padding/letterbox.',
        'proxy_not_human_native_visibility_gold':True,'current223_reads':0,'RGB_reads':0,'model_forwards':0})
    savej(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),
        'outputs':{n:binding(out/n) for n in ['RAW_MODEL_TARGETS9.npz','RAW_CHANNEL_CONTRACT.json']},
        'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
        'centroid_VIS_mask_count':int(Emask9[:,8].sum()),'native_VIS_counts_train':{'D':int((E9[a['fit']]==0).sum()),'E':int((E9[a['fit']]==1).sum())},
        'native_VIS_counts_dev':{'D':int((E9[~a['fit']]==0).sum()),'E':int((E9[~a['fit']]==1).sum())},
        'current223_reads':0,'RGB_reads':0,'model_forwards':0})
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)


if __name__=='__main__':main()
