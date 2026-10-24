"""Read label/valid/source/identity arrays only; no model or score dependency.

Inventory compatible-label mass and replay the registered NumPy row sampler.
Uniform-six-logit gradients below are analytical derivatives with respect to
independent output logits. They are NOT trained parameter gradients, feature
effects, branch causality, or an explanation proved from private performance.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'source_audit'
KEYS=('dense_allowed','dense_valid','dense_exact_state','dense_radius2_state',
      'global_allowed','global_valid','global_exact_state','global_radius2_state',
      'source','global_source','target_index','sample_id','global_id')
CARD=np.array([int(a).bit_count()for a in range(64)],np.int8)
G=np.zeros((64,6),np.float64);LOSS=np.zeros(64,np.float64)
for a in range(1,64):
    k=CARD[a];G[a]=1/6-np.array([bool(a&(1<<j))for j in range(6)])/k
    LOSS[a]=np.log(6/int(k))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def ratio(a,b):return float(a)/int(b)if b else None
def array_hash(a):
    h=hashlib.sha256((a.dtype.str+':'+repr(a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')

def load_labels(path,seal):
    with np.load(path,allow_pickle=False)as z:d={k:z[k]for k in KEYS}
    hit=next(b for b in seal['outputs']if(ROOT/b['path']).resolve()==path.resolve())
    n=len(d['source']);t=len(d['global_source'])
    assert d['dense_allowed'].shape==(n,9,9)and d['global_allowed'].shape==(t,64,64)
    assert d['target_index'].dtype==np.int64 and np.array_equal(d['global_source'][d['target_index']],d['source'])
    assert len(set(d['sample_id']))==n and len(set(d['global_id']))==t
    for prefix in ['dense','global']:
        a,v=d[prefix+'_allowed'],d[prefix+'_valid']
        assert a.dtype==np.uint8 and v.dtype==bool and np.isin(a,np.arange(1,64)).all()and(a[~v]==63).all()
        for suffix,classbits in [('exact_state',[1,6,56]),('radius2_state',[11,20,32])]:
            possible=np.stack([(a&b)!=0 for b in classbits],axis=1)
            expected=np.where(possible.sum(1)==1,possible.argmax(1),-1).astype(np.int8);expected[~v]=-1
            assert np.array_equal(d[prefix+'_'+suffix],expected),'Hard GT/compatible marginal mismatch'
    binding=dict(path=str(path),archive_sha256_from_existing_seal=hit['sha256'],
        archive_hash_not_recomputed_to_avoid_reading_feature_members=True,
        decoded_arrays={k:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=array_hash(a))for k,a in d.items()})
    return d,binding

def summary(split,source,view,a,v,hard,targets):
    valid=a[v];info=valid[valid!=63];hist=np.bincount(info,minlength=64)
    cards={str(k):int((CARD[valid]==k).sum())for k in range(1,7)}
    n=len(info);positive=int((valid==1).sum());negative=int(((valid&1)==0).sum());unknown=len(valid)-positive-negative
    up=int(((info&1)!=0).sum());down=int(((info&1)==0).sum())
    mean=(hist@G/n).tolist()if n else None
    record=dict(kind='label_inventory',split=split,source=source,view=view,targets=int(targets),total_cells=int(a.size),native_valid_cells=int(v.sum()),
        native_invalid_cells=int((~v).sum()),informative_cells=n,valid_uninformative63=int((valid==63).sum()),
        informative_label_cardinality_counts={str(k):int(hist[CARD==k].sum())for k in range(1,6)},valid_label_cardinality_counts=cards,
        compatible_mask_counts={str(k):int(c)for k,c in enumerate(np.bincount(valid,minlength=64))if c},
        exactV_confirmed_positive=positive,exactV_confirmed_negative=negative,exactV_unknown=unknown,
        exactV_unknown_but_informative=int(((info&1)!=0).sum()-(info==1).sum()),
        hard_exact_state_counts={str(k):int((hard[v]==k).sum())for k in [-1,0,1,2]},
        uniform_VV_up_cells=up,uniform_VV_down_cells=down,uniform_VV_no_pressure_valid=int((valid==63).sum()),
        uniform_VV_up_fraction_informative=ratio(up,n),uniform_VV_down_fraction_informative=ratio(down,n),
        uniform_mean_logit_gradient_joint_order=mean,uniform_mean_VV_logit_gradient=mean[0]if mean else None,
        uniform_mean_partial_NLL=float(hist@LOSS/n)if n else None,
        meaning='Negative dL/dzVV means gradient descent raises independent VV logit at uniform6 logits; not parameter/branch causality.')
    return record

def row_stats(a,v):
    info=v&(a!=63);count=info.reshape(len(a),-1).sum(1)
    def total(value):return(value*info).reshape(len(a),-1).sum(1)
    return dict(count=count,gVV=total(G[a,0]),loss=total(LOSS[a]),
        pos=total((a==1).astype(float)),neg=total(((a&1)==0).astype(float)),
        unknown=total((((a&1)!=0)&(a!=1)).astype(float)),up=total(((a&1)!=0).astype(float)),down=total(((a&1)==0).astype(float)))

def replay(d,protocol,candidate):
    sources=sorted(protocol['candidates'][candidate]['training_sources']);updates=protocol['updates'];batch=protocol['batch_size']
    pools={s:np.flatnonzero((d['source']==s)&d['dense_valid'][:,4,4]&(d['dense_allowed'][:,4,4]!=63))for s in sources}
    assert all(len(x)for x in pools.values())
    rng=np.random.default_rng(protocol['seed']);visits=np.zeros(len(d['source']),np.int64)
    ca=d['dense_allowed'][:,4,4];local=row_stats(d['dense_allowed'],d['dense_valid']);glob=row_stats(d['global_allowed'],d['global_valid'])
    sums={s:{branch:Counter()for branch in ['center','local_dense','global_dense']}for s in sources}
    allocation={s:Counter()for s in sources};global_active_source_counts=Counter();expected_unique={s:[]for s in sources}
    targetcounts=np.bincount(d['target_index'][np.concatenate(list(pools.values()))],minlength=len(d['global_source']))
    for update in range(1,updates+1):
        q,r=divmod(batch,len(sources));chunks=[]
        for j,s in enumerate(sources):
            take=q+int((j-(update-1))%len(sources)<r);allocation[s][take]+=1
            chunks.append(rng.choice(pools[s],size=take,replace=True))
        ix=np.concatenate(chunks);rng.shuffle(ix) # Preserve exact RNG consumption.
        np.add.at(visits,ix,1);active=0;global_sources_this_step=[]
        for s in sources:
            si=ix[d['source'][ix]==s];a=ca[si];n=len(si);cc=sums[s]['center']
            cc['represented_steps']+=1;cc['sampled_rows_total']+=n
            cc['gVV_sum_of_step_source_means']+=float(G[a,0].mean());cc['loss_sum_of_step_source_means']+=float(LOSS[a].mean())
            for name,value in [('pos',a==1),('neg',(a&1)==0),('unknown',((a&1)!=0)&(a!=1)),('up',(a&1)!=0),('down',(a&1)==0)]:cc[name+'_fraction_sum']+=float(value.mean())
            lc=sums[s]['local_dense'];den=int(local['count'][si].sum());assert den>=n
            lc['represented_steps']+=1;lc['informative_cells_total_with_row_replacement']+=den
            for name in ['gVV','loss']:lc[name+'_sum_of_step_source_means']+=float(local[name][si].sum()/den)
            for name in ['pos','neg','unknown','up','down']:lc[name+'_fraction_sum']+=float(local[name][si].sum()/den)
            lc['center_logit_extra_local_to_center_coefficient_ratio_sum']+=.25*n/den
            ts=np.unique(d['target_index'][si]);good=ts[glob['count'][ts]>0];gc=sums[s]['global_dense']
            gc['selected_unique_targets_total']+=len(ts);gc['ignored_zero_global_info_unique_targets_total']+=len(ts)-len(good)
            if len(good):
                active+=1;global_sources_this_step.append(s);gc['represented_steps']+=1;gc['informative_unique_targets_total']+=len(good)
                for name in ['gVV','loss']:gc[name+'_sum_of_step_source_means']+=float(np.mean(glob[name][good]/glob['count'][good]))
                for name in ['pos','neg','unknown','up','down']:gc[name+'_fraction_sum']+=float(np.mean(glob[name][good]/glob['count'][good]))
        global_active_source_counts[active]+=1
        for s in global_sources_this_step:sums[s]['global_dense']['outer_source_weight_sum']+=1/active
    result=[]
    for s in sources:
        pool=pools[s];tm=np.bincount(d['target_index'][pool],minlength=len(d['global_source']));source_targets=np.flatnonzero(d['global_source']==s)
        active_targets=source_targets[tm[source_targets]>0]
        draws=int(visits[pool].sum());n_by=allocation[s]
        expected=float(sum(freq*np.sum(1-(1-tm[active_targets]/len(pool))**take)for take,freq in n_by.items())/updates)
        for branch in ['center','local_dense','global_dense']:
            a=sums[s][branch];steps=int(a['represented_steps']);coef=1 if branch=='center'else .25
            record=dict(kind='sampler_replay',candidate=candidate,source=s,view=branch,steps=updates,source_count=len(sources),
                nominal_source_macro_weight=1/len(sources),actual_mean_outer_source_weight=a['outer_source_weight_sum']/updates if branch=='global_dense'else 1/len(sources),
                nominal_branch_coefficient=coef,center_pool_rows=len(pool),
                actual_row_draws=draws,expected_draws_per_pool_row=draws/len(pool),unique_pool_rows_seen=int((visits[pool]>0).sum()),
                draws_per_pool_row_min=int(visits[pool].min()),draws_per_pool_row_median=float(np.median(visits[pool])),draws_per_pool_row_max=int(visits[pool].max()),
                batch_allocation_histogram={str(k):int(v)for k,v in sorted(n_by.items())},source_targets=len(source_targets),
                center_reachable_targets=len(active_targets),expected_unique_targets_per_step=expected,represented_steps=steps,
                uniform_VV_gradient_step_source_mean=a['gVV_sum_of_step_source_means']/steps if steps else None,
                uniform_partial_NLL_step_source_mean=a['loss_sum_of_step_source_means']/steps if steps else None,
                fractions_step_source_mean={k:a[k+'_fraction_sum']/steps for k in ['pos','neg','unknown','up','down']}if steps else None,
                normalization='center:row mean; local:all informative cells in sampled rows; global:informative-cell mean per UNIQUE sampled target then equal target mean; outer equal-source mean',
                replay_uses_no_model_parameters_or_outputs=True)
            if branch=='local_dense':record['extra_local_center_logit_to_center_term_coefficient_ratio_mean']=a['center_logit_extra_local_to_center_coefficient_ratio_sum']/steps
            if branch=='global_dense':
                record['selected_unique_targets_per_step']=a['selected_unique_targets_total']/updates
                record['informative_unique_targets_per_step']=a['informative_unique_targets_total']/updates
                record['zero_info_unique_targets_ignored_total']=int(a['ignored_zero_global_info_unique_targets_total'])
            result.append(record)
    return dict(candidate=candidate,sources_sorted_as_trainer=sources,batch=batch,seed=protocol['seed'],sampler_iterations_replayed=updates,
        global_active_source_count_histogram={str(k):int(v)for k,v in sorted(global_active_source_counts.items())},records=result)

def main():
    out=HERE/'SOURCE_TRAINING_PRESSURE_AUDIT.json';assert not out.exists();start=time.monotonic()
    pp=ROOT/'FIT_PROTOCOL_V2.json';trainer=ROOT/'model/fit_partial_head.py';protocol=json.loads(pp.read_text())
    expected=next(b['sha256']for b in protocol['bindings']if b['path']==str(trainer));assert sha(trainer)==expected
    assert protocol['batch_size']==256 and protocol['updates']==2000 and protocol['loss']['center']==1 and protocol['loss']['local_dense']==protocol['loss']['global_dense']==.25
    sealpath=ROOT/'joined_source_cache/SEAL.json';seal=json.loads(sealpath.read_text());assert seal['complete']
    inventories=[];target_coverage=[];bindings=[];train=None;all_sources=None
    for split in ['TRAIN','VAL']:
        path=ROOT/'joined_source_cache'/('CONTROLLED_PLUS_PALLOX_'+split+'.npz');d,b=load_labels(path,seal);bindings.append(b)
        if split=='TRAIN':train=d
        sources=sorted(set(d['source']));assert sources==['CONTROLLED_YCB','ORIGINAL_YCB','PALLOX','SYNTH']
        pool=d['dense_valid'][:,4,4]&(d['dense_allowed'][:,4,4]!=63)
        counts=np.bincount(d['target_index'][pool],minlength=len(d['global_source']))
        global_info=d['global_valid']&(d['global_allowed']!=63);gcount=global_info.reshape(len(global_info),-1).sum(1)
        for s in sources:
            qi=d['source']==s;pi=qi&pool;ti=d['global_source']==s;reachable=ti&(counts>0)
            for view,a,v,h,nt in [
                ('center_all_rows',d['dense_allowed'][qi,4,4],d['dense_valid'][qi,4,4],d['dense_exact_state'][qi,4,4],ti.sum()),
                ('local_all_rows',d['dense_allowed'][qi],d['dense_valid'][qi],d['dense_exact_state'][qi],ti.sum()),
                ('local_center_informative_rows',d['dense_allowed'][pi],d['dense_valid'][pi],d['dense_exact_state'][pi],reachable.sum()),
                ('global_all_targets',d['global_allowed'][ti],d['global_valid'][ti],d['global_exact_state'][ti],ti.sum()),
                ('global_center_reachable_targets',d['global_allowed'][reachable],d['global_valid'][reachable],d['global_exact_state'][reachable],reachable.sum())]:
                inventories.append(summary(split,s,view,a,v,h,nt))
            target_coverage.append(dict(split=split,source=s,query_rows=int(qi.sum()),center_informative_pool=int(pi.sum()),global_targets=int(ti.sum()),
                eligible_center_count_per_target_histogram={str(k):int(v)for k,v in zip(*np.unique(counts[ti],return_counts=True))},
                center_reachable_targets=int(reachable.sum()),targets_with_no_center_info=int((ti&~reachable).sum()),
                center_unreachable_targets_with_global_information=int((ti&~reachable&(gcount>0)).sum()),
                informative_global_cells_on_center_unreachable_targets=int(gcount[ti&~reachable].sum()),
                local_info_cells_on_center_uninformative_rows=int((d['dense_valid'][qi&~pool]&(d['dense_allowed'][qi&~pool]!=63)).sum())))
        print('PRESSURE_LABELS',split,len(d['source']),len(d['global_source']),flush=True)
    replays=[replay(train,protocol,c)for c in ['CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX']]
    examples=[]
    for a in [1,4,32,10,31,62,63]:
        examples.append(dict(allowed=a,joints=[['VV','EV','EE','OV','OE','OO'][j]for j in range(6)if a&(1<<j)],cardinality=int(CARD[a]),
            uniform_partial_NLL=float(LOSS[a]),uniform_dL_dzVV=float(G[a,0]),
            descent_direction_VV='UP'if G[a,0]<0 else'DOWN'if G[a,0]>0 else'NONE'))
    report=dict(PASS=True,schema='SOURCE_LABEL_MASS_AND_SAMPLER_PRESSURE_AUDIT_V1',protocol=ref(pp),trainer=ref(trainer),cache_seal=ref(sealpath),code=ref(__file__),
        input_arrays_decoded_exactly=list(KEYS),source_cache_label_array_bindings=bindings,label_inventories=inventories,target_sampling_coverage=target_coverage,sampler_replays=replays,
        analytic_definition=dict(uniform_joint_probability='1/6 each; exact pV=1/6 and radius2 pV=3/6, so not a neutral binary-visible prior',
            partial_NLL='log(6/k), k=number of compatible states',logit_gradient='dL/dz_j=1/6-1[j in A]/k',
            VV='Exact visible is only jointVV: mask1 means positive, noVVbit means confirmed visible-negative, VV plus other states means exact visibility unknown.',
            UNKNOWN='Class E/O ambiguity can still certify visible-negative; all63 is uninformative and has exactly0 loss/gradient. UNKNOWN is not uniformly identical to all63.',
            pressure='For informative cells, VV included => negative derivative/raiseVV under descent; VV excluded => positive derivative/lowerVV. Direction of independent-logit derivative is not a learned parameter gradient.'),
        analytical_examples=examples,
        implementation_findings=[],normalization_review=dict(source_balanced=True,no_class_weights_or_class_balancing=True,center_pool_only=True,
            local_uses_repeated_sampled_rows_and_cell_weighting=True,global_deduplicates_targets_then_target_equal_mean_within_source=True,
            global_population_target_weights_not_uniform='Target inclusion probability depends on eligible center count m_t:1-(1-m_t/pool)^n. Unique-target reduction is per sampled minibatch, not equal over all source population targets.',
            center_ineligible_rows_not_recovered_by_neighbor_or_global_information=True,
            nominal_loss_coefficients=[1,.25,.25],empty_global_targets_ignored_as_registered=True),
        limitations=['This is structural label/sampler analysis. It does not read trained logits or compute actual parameter gradients.',
            'Actual gradients depend on current logits, input features, shared parameters and coupling between global/dense branches. Uniform aggregate direction cannot establish a branch or private-transfer cause.',
            'CONTROLLED_ONLY replay uses the identical first3 source blocks in the sealed PLUS label table, preserving row indices; PALLOX is appended in merge candidate order.',
            'No observed-score NLL decomposition is performed; another agent owns that saved-score analysis.'],
        new_model_forward_or_training_or_parameter_gradient=0,models_weights_features_current_private_arrays_read=0,GPU_used=False,PnP_calls=0,
        source_winner_thresholds_input_normalization_unchanged=True,runtime=dict(python=sys.executable,numpy=np.__version__),seconds=time.monotonic()-start)
    flat=[]
    for r in inventories:
        x={k:v for k,v in r.items()if not isinstance(v,(dict,list))}
        for k,v in r['valid_label_cardinality_counts'].items():x['valid_cardinality_'+k]=v
        flat.append(x)
    for rr in replays:
        for r in rr['records']:flat.append({k:v for k,v in r.items()if not isinstance(v,(dict,list))})
    fields=list(dict.fromkeys(k for r in flat for k in r));cp=HERE/'SOURCE_TRAINING_PRESSURE_AUDIT.csv'
    with cp.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    report['CSV']=ref(cp);write(out,report)
    print('SOURCE_TRAINING_PRESSURE_PASS',sha(out),sha(cp),'seconds',report['seconds'],flush=True)

if __name__=='__main__':main()
