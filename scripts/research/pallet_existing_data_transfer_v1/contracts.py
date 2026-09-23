import numpy as np
from collections import Counter
from . import common as E
from .train import Frozen

def main():
    E.immutable();ds=[Frozen(a) for a in E.ARMS];counts=Counter();c0=set();different=0
    for occ,r in enumerate(ds[0].records):
        for d in ds:d.epoch=occ//1024
        samples=[d[occ%1024] for d in ds]
        if r['role']!='REPLACEMENT':
            E.verify(r['cache'])
            for k in ('img','keypoints','bboxes','cls'):
                assert all(np.array_equal(samples[0][k].numpy(),s[k].numpy()) for s in samples[1:])
            if r['role']=='C0':c0.add(r['old']['target_index'])
        else:
            for b in r['cache'].values():E.verify(b)
            for k in ('img','bboxes','cls','batch_idx'):assert np.array_equal(samples[1][k].numpy(),samples[2][k].numpy())
            for s in samples:assert np.array_equal(samples[0]['keypoints'][:,:,2],s['keypoints'][:,:,2])
            mask=samples[1]['keypoints'][0,:,2]==2
            different+=int(not np.array_equal(samples[1]['keypoints'][0,mask,:2],samples[2]['keypoints'][0,mask,:2]))
            assert (samples[0]['keypoints'][0,~mask,2]==1).all()
        counts[r['role']]+=1
    assert len(c0)==10 and counts['C0']==counts['REPLACEMENT']==1280 and counts['SYNTH']==2560 and different>0
    assert E.read(E.C.DOC/'LOSS_TEST.json')['passed']
    E.save(E.DOC/'PRETRAIN_TESTS.json',dict(passed=True,counts=dict(counts),C0_unique=len(c0),manual_target_changed_occurrences=different,all_5120_occurrences_checked=True,T1_T2_RGB_bbox_class_exact=True,all_three_support_exact=True,C0_synthetic_all_exact=True,ignored_sentinel_restored=True,TrueIgnore_test_reused=E.bind(E.C.DOC/'LOSS_TEST.json'),numeric_tolerance=dict(atol=1e-5,rtol=1e-4),new_optimizer_steps=0))
    print('PRETRAIN_TESTS_PASS',counts,different,flush=True)

if __name__=='__main__':main()
