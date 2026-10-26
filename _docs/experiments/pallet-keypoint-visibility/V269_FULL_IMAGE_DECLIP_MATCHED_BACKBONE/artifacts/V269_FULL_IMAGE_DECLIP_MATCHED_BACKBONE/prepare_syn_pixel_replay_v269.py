"""Prepare tiny portable saved-SYN replay fixtures; no new-model scores."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).absolute().parent
OLD=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
V268=HERE.parent/'V268_FULL_IMAGE_BOUNDARY_REFINEMENT'
FIXED=(768,769,770,776,792,900)


def safe(v):
    p=Path(v);assert p.is_absolute()and not any(x in str(p).lower()for x in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p


def binding(v):
    p=safe(v);x=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(x).hexdigest(),size_bytes=len(x))


def checked(b):
    assert binding(b['path'])==b
    return safe(b['path'])


def rel(b):return dict(relative_path=str(Path(b['path']).relative_to(HERE)),sha256=b['sha256'],size_bytes=b['size_bytes'])


def main():
    out=safe(HERE/'syn_pixel_replay_v1');assert not out.exists();out.mkdir()
    p=safe(V268/'syn_pixel_replay_v1/SYN_DEV_M4_PACKBITS.npz')
    assert binding(p)['sha256']=='1619a1d6876d2ce243066a2c4ac18840ec932eb3cf322b69a7441a2c10562b0e'
    with safe(out/p.name).open('xb')as f:f.write(p.read_bytes())
    packet=binding(out/p.name)
    old=json.loads(safe(OLD/'syn_training_v1/COMPLETE.json').read_bytes());assert old['PASS']
    old_bound={}
    for label,key in [('OPENAI_FULL_CLIP','PLAIN_CLIP'),('V267_CS','CLIP_SURGERY_ARCH')]:
        b=old['outputs'][key+'_SYN_DEV_MASKS.npz']
        with np.load(checked(b),allow_pickle=False)as z:ids=np.array(z['frame_ids'],copy=True);logit=np.array(z['mask_logits'],copy=True)
        assert logit.shape==(235,1,32,32)and np.array_equal(ids,np.arange(768,1003))
        p=safe(out/(label+'_DEV_LOGITS.npz'));np.savez_compressed(p,frame_ids=ids,mask_logits=logit);old_bound[label]=rel(binding(p))
    pairs=json.loads(safe(V268/'syn_selected_raw_inputs_v1/SOURCE_RGB_M4_PAIRS.json').read_bytes());assert pairs['PASS']
    rawdir=safe(out/'raw6');rawdir.mkdir();raw={i:{}for i in FIXED}
    for row in pairs['files']:
        i=row['frame_index'];assert i in FIXED
        p=safe(V268/'syn_selected_raw_inputs_v1'/row['output']);b=binding(p);assert b['sha256']==row['sha256']and b['size_bytes']==row['size_bytes']
        q=safe(rawdir/row['output'])
        with q.open('xb')as f:f.write(p.read_bytes())
        label='RGB'if row['output'].startswith('RAW_RGB')else'M4';raw[i][label]=rel(binding(q))
    assert all(set(r)=={'RGB','M4'}for r in raw.values())
    manifest=dict(completed=True,PASS=True,frames=235,packed_M4=rel(packet),old_logits=old_bound,raw6={str(k):v for k,v in raw.items()},
        new_arms=['EVA_STANDARD','EVA_CSA','EVA_DECLIP'],fixed_visual_frame_indices=list(FIXED),
        source_packbits=binding(V268/'syn_pixel_replay_v1/SYN_DEV_M4_PACKBITS.npz'),old_training=binding(OLD/'syn_training_v1/COMPLETE.json'),
        source_raw_pairs=binding(V268/'syn_selected_raw_inputs_v1/SOURCE_RGB_M4_PAIRS.json'),
        no_new_model_scores_read=True,new_fits=0,new_model_forwards=0,current_reads=0)
    p=safe(out/'REPLAY_INPUTS.json');p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    complete=dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),manifest=binding(p),
        copied_binary_M4_exact=True,old_logits_lossless=True,raw12_encoded_files_copied_exact=True,new_model_scores_read=0,current_reads=0,new_fits=0)
    q=safe(out/'PREPARATION_COMPLETE.json');q.write_text(json.dumps(complete,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
