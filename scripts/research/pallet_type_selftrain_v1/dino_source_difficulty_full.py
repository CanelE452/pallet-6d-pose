"""Independent source-array geometry and full existing-source tail inventory."""
import numpy as np
from . import dino_mid_feature as U


def measure(arrays,rows):
    box=np.asarray(arrays['boxes'][rows],np.float64);q=np.asarray(arrays['points'][rows,:8],np.float64)
    gt=np.asarray(arrays['gt_points'][rows,:8],np.float64);gain=np.asarray(arrays['gain'][rows],np.float64).reshape(-1)
    valid=np.asarray(arrays['point_valid'][rows,:8],bool)&np.isfinite(q).all(-1)
    mask=np.asarray(arrays['gt_valid'][rows,:8],bool)&np.isfinite(gt).all(-1)
    assert valid.any(-1).all() and (gain>0).all()
    wh=box[:,2:]-box[:,:2];assert (wh>0).all()
    # Source input-canvas -> wide crop; translation offsets cancel relative to box.
    scale=288/(np.maximum(wh[:,0],wh[:,1]*.75)*1.25)
    target=(gt-(box[:,:2]+box[:,2:])[:,None]/2)*scale[:,None,None]+[288,384]
    mask&=(target>=0).all(-1)&(target[:,:,0]<576)&(target[:,:,1]<768)
    error=np.linalg.norm(q-gt,axis=-1)/gain[:,None]
    distance=np.linalg.norm(gt[:,:,None,:]-q[:,None,:,:],axis=-1)/gain[:,None,None]
    nearest=np.where(valid[:,None,:],distance,np.inf).min(-1)
    return dict(corners=mask.sum(-1),good5=((error<5)&mask).sum(-1),hard20=((error>20)&mask).sum(-1),
        far40=((nearest>40)&mask).sum(-1))


def main():
    p=U.verify();source=U.P.SourceData();data=source.data;prior=U.C.read(U.DOC/'SOURCE_DIFFICULTY_AUDIT.json')
    for b in prior['evidence']:U.C.verify(b)
    known={r['row']:r for r in prior['rows']};summary={};records=[]
    for name,rows in [('train',source.train_rows),('held',source.held_rows)]:
        rr=[]
        for offset in range(0,len(rows),1024):
            ii=rows[offset:offset+1024];values=measure(data.arrays,ii)
            for j,row in enumerate(ii):
                original=data.source['records'][int(data.indices[row])]
                item=dict(row=int(row),id=original['id'],partition=name,scenario=original['scenario_id'],source=original['source'],
                    **{k:int(v[j]) for k,v in values.items()})
                if int(row) in known:
                    for k in values:assert item[k]==known[int(row)][k],(int(row),k,item[k],known[int(row)][k])
                rr.append(item)
        summary[name]=dict(images=len(rr),images_with_far40=sum(r['far40']>0 for r in rr),**{k:sum(r[k] for r in rr) for k in ['corners','good5','hard20','far40']})
        records.extend(rr)
    assert {r['row'] for r in records}>=set(known)
    U.C.freeze(U.DOC/'FULL_SOURCE_DIFFICULTY_AUDIT.json',dict(status='SOURCE_ONLY_INVENTORY_NOT_SAMPLE_SELECTION',
        summary=summary,records=records,selected8256_counts_independently_reproduced=True,new_training_run=False,
        source_masks='Reconstruct exact wide support from original cached input-canvas GT, R0 boxes and gain; no image decode/crop-target reuse.',
        no_real_GT=True,no_labels_changed=True,no_pool_changed=True,
        evidence=[U.C.bound(__file__),U.C.bound(U.DOC/'SOURCE_DIFFICULTY_AUDIT.json'),
            U.C.bound(data.directory/'CACHE_MANIFEST.json'),U.C.bound(data.directory/'CACHE_COMPLETE.json'),U.C.bound(data.run_dir/'SOURCE_MANIFEST.json')]))
    print('FULL_SOURCE_DIFFICULTY',summary,flush=True)


if __name__=='__main__':main()
