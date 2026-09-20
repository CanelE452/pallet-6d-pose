"""Verify optimized block6 extraction against the pinned official API."""
import numpy as np
import torch
from . import dino_mid_feature as U


def main():
    p=U.verify();U.N.setup();print('GPU',U.N.E.gpu(),flush=True)
    source=U.P.SourceData();backbone,_=U.D.A.load();records=[]
    train=sorted(r['row'] for r in p['source_records'] if r['partition']=='train')
    cache={r['row']:r for r in U.C.read(U.X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'}
    for row in train[:2]+p['source_held_rows'][:2]:
        x=U.W.source_item(source,row)
        with torch.no_grad():
            tensor=U.image_tensor([x]);reference=backbone.get_intermediate_layers(tensor,n=[5,11],norm=True)
            fast=U.M.block_features(backbone,tensor,False)[0];both=U.M.block_features(backbone,tensor,True)
        torch.testing.assert_close(fast,reference[0],atol=0,rtol=0)
        for a,b in zip(both,reference):torch.testing.assert_close(a,b,atol=0,rtol=0)
        with np.load(U.C.ROOT/cache[row]['cache']['path']) as z:np.testing.assert_array_equal(U.array(reference[1])[0],z['feature'])
        records.append(dict(row=row,mid_sha=U.N.array_sha(U.array(fast)[0]),late_sha=U.N.array_sha(U.array(reference[1])[0])))
    U.C.freeze(U.DOC/'EXTRACTOR_PREFLIGHT.json',dict(passed=True,source_images=4,official_block_indices=[5,11],
        exact_FP32_token_match=True,exact_FP16_parent_late_match=True,records=records,
        source=U.C.bound(__file__),protocol=U.C.bound(U.DOC/'PROTOCOL.json')))
    print('MID_EXTRACTOR_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
