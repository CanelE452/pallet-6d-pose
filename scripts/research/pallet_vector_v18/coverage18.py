"""Posthoc check: can nearby line samples actually see retained image pixels?"""
from context18 import *
from model18 import geometry

def main():
    setup();rows={}
    for condition,typ in [('clean','all'),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        m,O=load('real',condition);ts=targets('real',condition);det=np.array([r['detected'] for r in m['rows']]);gt=np.array([t['points'] for t in ts])[det];valid=np.array([t['valid'] for t in ts])[det]
        if typ!='all':valid&=np.array([t[typ] for t in ts])[det]
        for seed in SEEDS:
            rep=repcache('real',condition,seed);sample=geometry(O,rep,rep['base'][:,:,None])['sample'][:,:,0];bb=O['box'];wh=(bb[:,2:]-bb[:,:2]).clamp_min(1);sg=2*(sample-bb[:,None,None,None,None,:2])/wh[:,None,None,None,None]-1;hw=O['image_hw'].flip(-1)[:,None,None,None,None]
            inside=(((sample>=0)&(sample<hw)).all(-1)&((sg>=-1)&(sg<=1)).all(-1)).numpy();fraction=inside.mean(axis=(-1,-2,-3));empty=fraction==0;pred=np.load(RUN/'predictions/real'/condition/f'line_vector_seed{seed}.npz');base=pred['base'][det];new=pred['points'][det];delta=np.linalg.norm(new-gt,axis=-1)-np.linalg.norm(base-gt,axis=-1)
            stats={}
            for name,mask in [('all',valid),('no_valid_line_samples',valid&empty),('has_valid_line_samples',valid&~empty)]:
                stats[name]=dict(points=int(mask.sum()),mean_error_change_px=float(delta[mask].mean()) if mask.any() else None,harmful_points=int((mask&(delta>1e-4)).sum()),mean_valid_sample_fraction=float(fraction[mask].mean()) if mask.any() else None)
            rows[f'{condition}/seed{seed}']=stats
    write(RUN/'COVERAGE.json',dict(rows=rows,scope='Posthoc diagnostic, no tuning. Geometric in-image and in-ROI sample coverage, not physical line correctness or visibility under occlusion. Zero valid samples still leaves line geometry/association tokens and parent context available. Correlation does not establish a cause.'))
    print('COVERAGE COMPLETE',flush=True)

if __name__=='__main__':main()
