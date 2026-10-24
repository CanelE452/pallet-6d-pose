"""Pure invented-array math. No source, model, GT or score file I/O."""
import hashlib
import numpy as np

def fixed_auc(score,target,cohort):
    score=np.asarray(score,np.float64);target=np.asarray(target);cohort=np.asarray(cohort,bool)
    known=cohort&np.isin(target,[0,1]);defined=np.isfinite(score);used=known&defined
    s=score[used];y=target[used];npos=int((y==1).sum());nneg=int((y==0).sum());auc=None
    if npos and nneg:
        order=np.argsort(s,kind='stable');ss=s[order];ranks=np.empty(len(s),np.float64);lo=0
        while lo<len(ss):
            hi=lo+1
            while hi<len(ss)and ss[hi]==ss[lo]:hi+=1
            ranks[order[lo:hi]]=(lo+1+hi)/2;lo=hi
        auc=float((ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    return dict(AUC_fixed_higher_score_positive=auc,original_rows=int(cohort.sum()),target_known=int(known.sum()),target_unknown=int((cohort&~np.isin(target,[0,1])).sum()),input_defined_all=int((cohort&defined).sum()),input_undefined_all=int((cohort&~defined).sum()),ranked_positive=npos,ranked_negative=nneg,
        unranked_positive=int((known&~defined&(target==1)).sum()),unranked_negative=int((known&~defined&(target==0)).sum()),one_or_zero_ranked_classes=not(npos and nneg))

def train_input_quintiles(score,train):
    score=np.asarray(score,np.float64);train=np.asarray(train,bool);finite=score[train&np.isfinite(score)];bins=np.full(len(score),-1,np.int8)
    if len(finite):
        edges=np.quantile(finite,[.2,.4,.6,.8]);valid=np.isfinite(score);bins[valid]=np.searchsorted(edges,score[valid],side='right')
        digest=hashlib.sha256(np.asarray(edges,dtype='<f8').tobytes()).hexdigest();distinct=int(len(np.unique(edges)))
    else:edges=np.empty(0,np.float64);digest=hashlib.sha256(b'').hexdigest();distinct=0
    # Numerical cutpoints stay in memory. Outputs carry bins/hash only.
    proof=dict(TRAIN_finite_input_N=int(len(finite)),TRAIN_all_input_N=int(train.sum()),edge_sha256=digest,distinct_quantile_edges=distinct,
        train_bin_counts={str(i):int((train&(bins==i)).sum())for i in(-1,0,1,2,3,4)},numeric_boundaries_exported=False,side='right',GT_or_VAL_used=False)
    return bins,proof

def target_bin_counts(bins,target,cohort):
    out={};target=np.asarray(target);cohort=np.asarray(cohort,bool)
    for b in(-1,0,1,2,3,4):
        group=cohort&(bins==b);known=group&np.isin(target,[0,1]);positive=int((known&(target==1)).sum());total=int(known.sum())
        out['UNDEFINED'if b<0 else'Q'+str(b+1)]=dict(original_rows=int(group.sum()),target_known=total,target_unknown=int((group&~np.isin(target,[0,1])).sum()),positive=positive,negative=int((known&(target==0)).sum()),positive_rate=float(positive/total)if total else None)
    return out

def generated_checks():
    allrows=np.ones(4,bool)
    assert fixed_auc([1,2,2,3],[0,0,1,1],allrows)['AUC_fixed_higher_score_positive']==.875
    assert fixed_auc([3,2,2,1],[0,0,1,1],allrows)['AUC_fixed_higher_score_positive']==.125
    assert fixed_auc([1,1,1,1],[0,0,1,1],allrows)['AUC_fixed_higher_score_positive']==.5
    missing=fixed_auc([0,np.nan,1,np.nan],[0,1,1,-1],allrows)
    assert missing['AUC_fixed_higher_score_positive']==1 and missing['unranked_positive']==1 and missing['target_unknown']==1 and missing['original_rows']==4
    assert fixed_auc([1,2],[1,1],[1,1])['AUC_fixed_higher_score_positive']is None
    train=np.array([1]*10+[0]*4,bool);s=np.arange(14,dtype=float);b,p=train_input_quintiles(s,train);s2=s.copy();s2[~train]=[-999,999,np.nan,500]
    b2,p2=train_input_quintiles(s2,train);assert p==p2 and np.array_equal(b[train],b2[train])
    tied,proof=train_input_quintiles(np.zeros(8),np.ones(8,bool));assert proof['distinct_quantile_edges']==1 and np.array_equal(tied,np.full(8,4))
    absent,proof=train_input_quintiles(np.full(3,np.nan),np.ones(3,bool));assert np.array_equal(absent,np.full(3,-1))and proof['TRAIN_finite_input_N']==0
    bins,proof=train_input_quintiles(np.arange(10,dtype=float),np.ones(10,bool));counts=target_bin_counts(bins,np.array([0,1]*5),np.ones(10,bool));assert sum(x['original_rows']for x in counts.values())==10
    return dict(PASS=True,invented_arrays_only=True,exact_tiedrank_AUC_fixed_polarity_not_flipped=True,undefined_input_and_unknown_target_denominators_preserved=True,
        oneclass_AUC_null=True,TRAIN_input_quintiles_unchanged_by_VAL_values_or_missingness=True,duplicate_edges_empty_bins_kept_no_jitter=True,no_numerical_bin_cutvalues_exported=True,no_source_GT_score_model_reads=True)
