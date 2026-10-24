"""Pure all-row mask adapter. Importing this module never opens an asset."""
import numpy as np


def require(value,message):
    if not bool(value):raise AssertionError(message)


def fit_mask(metadata,split_rows):
    """Keep full caches/targets intact; filter only optimization eligibility."""
    require(len(metadata)==len(split_rows),'Full original row count')
    result=np.zeros(len(metadata),bool)
    for i,(original,row) in enumerate(zip(metadata,split_rows)):
        require(row['cache_index']==i and row['row_id']==original['row_id'],'Exact original row/order')
        require(row['role'] in ('TRAIN','HOLDOUT'),'Frozen component role')
        for name in ('original_global_index','native_slot','original_query_available'):
            require(row[name]==original[name],'Original identity/availability '+name)
        for name in ('query_xy','L'):
            a=np.asarray([np.nan,np.nan] if name=='query_xy' and original[name] is None else np.nan if original[name] is None else original[name],np.float64)
            b=np.asarray([np.nan,np.nan] if name=='query_xy' and row[name] is None else np.nan if row[name] is None else row[name],np.float64)
            require(np.array_equal(a,b,equal_nan=True),'Exact original F64 '+name)
        result[i]=row['role']=='TRAIN'
    return result


def masked_old_pools(caches,train_masks,original_pools_for):
    """First retain inherited full-cache guards, then apply the same split everywhere.

    The supplied frozen original pools_for validates full253/full314 authority
    and all original soft masses. No row-normalization or partial-known drop.
    """
    full=original_pools_for(caches)
    domains=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816')
    for domain in domains:
        mask=train_masks[domain]
        require(mask.dtype==bool and mask.shape==(len(caches[domain]['x']['original_query_available']),),'All-row mask layout')
    synth,public,cf=(train_masks[d] for d in domains)
    support=[[a[synth[a]] for a in group] for group in full['support']]
    conditional=[]
    class_mass=[]
    for probability in full['conditional']:
        require(probability.shape==synth.shape and np.isfinite(probability).all() and (probability>=0).all(),'Original class-conditional probabilities')
        weights=np.where(synth,probability,0.)
        total=weights.sum(dtype=np.float64);class_mass.append(float(total))
        conditional.append(weights/total if total>0 else None)
    hard_public=[a[public[a]] for a in full['public']]
    hard_cf=[a[cf[a]] for a in full['CF']]
    counts=dict(support=[[len(a) for a in group] for group in support],synthetic_class_probability_retained=class_mass,public=[len(a) for a in hard_public],CF=[len(a) for a in hard_cf])
    feasible=all(len(a)>0 for group in support for a in group) and all(a is not None for a in conditional) and all(len(a)>0 for group in (hard_public,hard_cf) for a in group)
    return dict(feasible=feasible,pools=dict(support=support,conditional=conditional,public=hard_public,CF=hard_cf) if feasible else None,counts=counts,no_fallback_or_rebalance=True)


def masked_direct_pools(available,target,train_mask):
    require(available.dtype==train_mask.dtype==bool and available.shape==target.shape==train_mask.shape,'Direct all-row layout')
    require(np.isin(target,[-1,0,1]).all() and (target[~available]==-1).all(),'Original direct UNKNOWN/U')
    pools=[np.flatnonzero(available&train_mask&(target==i)) for i in (0,1)]
    return dict(feasible=all(len(a)>0 for a in pools),pools=pools,counts=[len(a) for a in pools],no_fallback_or_rebalance=True)


def assert_draws_train(draws,direct_draws,train_masks):
    mapping={'visible':'SYNTH_AUG65536','foreign':'SYNTH_AUG65536','synthetic_owner':'SYNTH_AUG65536','public_owner':'PUBLIC_TRAIN1480','CF_owner':'CF3816'}
    for name,indices in draws.items():require(train_masks[mapping[name]][indices].all(),'No HOLDOUT optimization row '+name)
    require(train_masks['DIRECT_CONTROL'][direct_draws].all(),'No HOLDOUT direct optimization row')
