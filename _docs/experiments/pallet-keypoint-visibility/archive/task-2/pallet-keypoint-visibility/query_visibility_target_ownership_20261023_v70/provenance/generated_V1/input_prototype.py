"""GT-free, current-target, leave-one-ROI-out prototype from cached inputs.

Grouping represents the current input target/query set, not a semantic slot.
Build on ALL included ROIs before any V/E/O/U training label filtering. Never
mix original and composite target groups, or use another frame as a prototype.
"""
import numpy as np

MASS_EPS=1e-8  # Predeclared denominator stability only; never tuned on VAL.


def input_prototypes(x,sam,box_grid,context,target_index):
    n=len(x)
    for array,shape,name in ((x,(n,65,9,9),'x'),(sam,(n,2,9,9),'sam'),
        (box_grid,(n,4,9,9),'box_grid'),(context,(n,64),'context'),(target_index,(n,),'target_index')):
        if array.shape!=shape:raise ValueError(name+': '+str(shape))
    if target_index.dtype!=np.int64:raise ValueError('Input target grouping int64 required')
    if not all(np.isfinite(a).all() for a in (x,sam,box_grid,context)):raise ValueError('Finite cached inputs required')
    targets,inverse=np.unique(target_index,return_inverse=True);t=len(targets)
    roi_sum=np.zeros((n,64),np.float64);mass=np.zeros(n,np.float64);sites=np.zeros(n,np.float64)
    for start in range(0,n,512):
        sl=slice(start,min(n,start+512));valid=np.clip(x[sl,64].astype(np.float64),0,1)
        within=(np.abs(box_grid[sl,0])<=1)&(np.abs(box_grid[sl,1])<=1)
        raw=np.clip(sam[sl,0].astype(np.float64)*8,-32,32)
        probability=1/(1+np.exp(-raw))
        weight=valid*within*probability
        roi_sum[sl]=np.einsum('nchw,nhw->nc',x[sl,:64].astype(np.float64),weight,optimize=False)
        mass[sl]=weight.sum((1,2));sites[sl]=(valid*within).sum((1,2))
    target_sum=np.zeros((t,64),np.float64);target_mass=np.zeros(t,np.float64);target_sites=np.zeros(t,np.float64)
    np.add.at(target_sum,inverse,roi_sum);np.add.at(target_mass,inverse,mass);np.add.at(target_sites,inverse,sites)
    count=np.bincount(inverse,minlength=t);other_count=count[inverse]-1
    if np.any(count>8):raise ValueError('This pilot requires at most8 initially supplied ROIs per current target')
    other_sum=target_sum[inverse]-roi_sum
    other_mass=np.maximum(target_mass[inverse]-mass,0)
    other_sites=np.maximum(target_sites[inverse]-sites,0)
    fallback=(other_mass<=MASS_EPS)|(other_count==0)
    prototype=np.array(context,np.float32,copy=True)
    good=~fallback;prototype[good]=(other_sum[good]/other_mass[good,None]).astype(np.float32)
    # Info is input-derived reliability, not GT support or a prediction veto.
    info=np.stack((other_count/7,np.divide(other_mass,np.maximum(other_sites,1)),fallback.astype(np.float64),
                   other_sites/np.maximum(other_count*81,1)),1).astype(np.float32)
    if not np.isfinite(prototype).all() or not np.isfinite(info).all():raise ValueError('Finite input-only prototypes required')
    audit=dict(recipe='leave-one-entire-ROI-out; other same-current-target initial ROIs only',
        mass_epsilon=MASS_EPS,n_queries=n,n_current_targets=t,other_ROI_count_min=int(other_count.min()) if n else None,
        other_ROI_count_max=int(other_count.max()) if n else None,fallback_count=int(fallback.sum()),
        fallback='Provided same-current-target cached input context, with explicit flag; context may include current ROI.',
        weighting='clip(P3valid,0,1) * rough-box-inside * sigmoid(8*normalizedMobileSAMexact)',
        pooling_uses_GT_or_VE_subset=False,query_ROI_self_matching_excluded=True,
        overlap_limit='Other initial ROIs can overlap in native space; this is not a certified clean target mask.',
        multiquery_deployment_contract='All available same-current-target ROIs (1..8), not canonical native semantic IDs.',
        info_columns=['other_ROI_count/7','other_softmask_mass/other_valid_box_sites','context_fallback_flag','other_valid_box_sites/(other_ROI_count*81)'])
    return prototype,info,audit
