"""Recompute scores and replay checkpoints using observation-only inputs."""
from core import *

def main():
    torch.set_num_threads(2)
    frozen=read(OUT/'TRAIN_SOURCE_LOCK.json')
    assert sha(OUT/'PROTOCOL.json')==frozen['protocol_sha256']
    for name,digest in frozen['sources'].items():assert sha(KIT/name)==digest,name
    for split,digest in frozen['cache_sha256'].items():assert sha(OUT/f'{split}.pt')==digest
    tr=read(OUT/'TRAINING_COMPLETE.json');assert tr['optimizer_steps']==12000
    data=load_cache('val');records=read(OUT/'val_records.json')
    obs={k:data[k] for k in ['features','dims_whd','level','anchor']}
    rows=read(OUT/'PER_FRAME.json');saved=np.load(OUT/'predictions.npz');h=base_model().model[-1]
    replay={};score_error=0.
    for arm,(dims,sym) in ARMS.items():
        for seed in [1,2,3]:
            tag=f'{arm}_seed{seed}';path=OUT/'models'/tag/'final.pt'
            assert sha(path)==tr['cells'][tag]['checkpoint_sha256']
            model=OutputAdapter(h,seed);model.load_state_dict(torch.load(path,weights_only=True)['state'])
            with torch.no_grad():pred=model(obs,dims)
            xy=pred[:,:,:2].numpy()*data['stride'].numpy()[:,None,None]
            xy=(xy-data['letterbox_offset'].numpy()[:,None])/data['letterbox_gain'].numpy()[:,None,None]-data['pad'].numpy()[:,None,None]
            replay[tag]=float(np.max(np.abs(xy-saved[tag])))
            assert replay[tag]<.0001
            for i,(r,out) in enumerate(zip(records,rows[tag])):
                a=np.array(r['box_raw']);b=np.array(r['gt_box_raw'])
                intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection
                matched=bool(r['confidence']>=.25 and intersection/max(union,1e-9)>=.5)
                assert matched==out['matched']
                diag=np.hypot(*r['raw_hw']);gt=np.array(r['gt_raw'])[:8];mask=np.array(r['valid'])[:8]
                choices=[np.arange(8)]
                if r['order']==2:choices.append(np.array(r['perm'])[:8])
                means=[np.linalg.norm(xy[i,:8]-gt[p],axis=1)[mask[p]].mean() if mask.any() else diag for p in choices]
                value=min(means) if matched else diag
                expected=min(value/diag,1)
                score_error=max(score_error,abs(expected-out['normalized']))
    assert score_error<1e-7,score_error
    hashes=read(OUT/'CACHE_AUDIT.json')['image_label_sha256'];groups={};images={}
    for split in ['train','val']:
        rr=read(OUT/f'{split}_manifest.json')['records']
        groups[split]={r['pair_group_id'] for r in rr}
        images[split]={hashes[r['image_path']] for r in rr}
    assert not groups['train']&groups['val'] and not images['train']&images['val']
    assert read(ROOT/'data/pallet/results/gpu_recovery_20260911/no_reboot_recovery.json')['PASS']
    write(OUT/'AUDIT.json',dict(PASS=True,checkpoint_replay_max_abs_px=replay,
        independent_primary_max_abs_error=score_error,observation_only_replay=True,
        train_val_group_overlap=0,train_val_image_hash_overlap=0,optimizer_steps=12000,
        runtime_recovered_without_reboot=True,summary_sha256=sha(OUT/'SUMMARY.json')))
    print('Independent audit PASS')

if __name__=='__main__':main()
