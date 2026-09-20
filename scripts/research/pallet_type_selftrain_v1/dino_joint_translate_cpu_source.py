"""CPU-only source calibration while another user experiment owns the GPU.

Same frozen weights, cached features, proposal function and thresholds. This
does not disable the GPU guard or terminate any other process. Real inference
remains pending until GPU access is available.
"""
import numpy as np
import torch
from . import dino_joint_translate as T


@torch.no_grad()
def main():
    C=T.C;p=T.verify();torch.set_num_threads(2)
    assert not (T.RAW/'SOURCE_PROPOSALS.json').exists()
    models={}
    for a in T.D.ARMS:
        f=C.read(T.X.DOC/f'FIT_{a}.json');C.verify(f['checkpoint']);m=T.V.V.Head()
        m.load_state_dict(torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)['model']);models[a]=m.eval()
        assert all(x.device.type=='cpu' for x in m.parameters())
    side=np.load(T.SIDECAR);src=T.P.SourceData();np.testing.assert_array_equal(side['record_index'],src.data.indices)
    cache={r.get('row'):r for r in C.read(T.X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'};records=[]
    for i,row in enumerate(p['calibration_rows']+p['test_rows']):
        meta=cache[row];C.verify(meta['cache'])
        with np.load(C.ROOT/meta['cache']['path']) as z:r=dict(meta,**{k:np.array(z[k]) for k in z.files if k!='protocol_sha256'})
        feature=torch.as_tensor(r['feature']).float()[None];q0=torch.as_tensor(r['points'])[None];v0=torch.as_tensor(r['valid'])[None]
        zz={a:m(feature,q0,v0) for a,m in models.items()};lp=T.M.logmass(zz['SYN'],zz['MIX'])
        perms=side['permutations'][row,:int(side['order'][row])]
        for j in range(5):
            q,v=T.view(r,j);proposal=T.M.propose(lp,torch.as_tensor(q),torch.as_tensor(v))
            after=T.M.restore(q,proposal,np.eye(3));err=T.J.canonical_errors(np.stack([q,after]),r['target'],r['target_valid'],perms,r['matrix'][0,0])
            records.append(dict(row=row,view=j,proposal=proposal,errors=[[None if not np.isfinite(v) else float(v) for v in line] for line in err]))
        if (i+1)%8==0:print('CPU_REGISTER_SOURCE',i+1,'/64',flush=True)
    assert not torch.cuda.is_initialized()
    C.freeze(T.RAW/'SOURCE_PROPOSALS.json',dict(records=records,GT_used_only_for_diagnostic_labels=True,inputs_include_no_GT=True,
        execution_backend='CPU_FLOAT32_THREADS2',adapter=C.bound(__file__),CUDA_initialized=False))
    C.freeze(T.DOC/'CPU_SOURCE_ADAPTER.json',dict(code=C.bound(__file__),source_proposals=C.bound(T.RAW/'SOURCE_PROPOSALS.json'),
        reason='GPU guard detected foreign PID5296 pallet_direct_dimension_v1.evaluate synthetic before source work. Preserve foreign job;CPU calibration only.',
        mathematical_protocol_changed=False,bit_exact_CPU_GPU_claim=False,real_inference_still_requires_GPU_guard=True))
    T.fit();print('CPU_REGISTER_CALIBRATION_COMPLETE',flush=True)


if __name__=='__main__':main()
